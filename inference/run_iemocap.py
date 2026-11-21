#!/usr/bin/env python3
"""Per-tier (T / T+A / T+A+V) inference on IEMOCAP (Session1-4 = calibration, Session5 = test).

6-class: angry, excited, frustrated, happy, neutral, sad
Video is dialog-level, extracted per-utterance using timestamps.
Audio is utterance-level wav files.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import io
import base64
import numpy as np
from PIL import Image
from tqdm import tqdm

IEMOCAP_FULL = Path("data/IEMOCAP_full_release")  # overridden by --iemocap_full
IEMOCAP_LABELS = Path("data/iemocap_splits")  # overridden by --iemocap_labels
EMOTIONS = ["angry", "excited", "frustrated", "happy", "neutral", "sad"]
TIERS = ["T", "T+A", "T+A+V"]


def extract_video_frames(video_path, max_frames=8, max_resolution=512,
                         start_time=None, end_time=None):
    """Extract frames from video, optionally within a time range."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_time is not None and end_time is not None:
        start_frame = int(start_time * fps)
        end_frame = min(int(end_time * fps), total_frames - 1)
    else:
        start_frame = 0
        end_frame = total_frames - 1

    nframes = min(max_frames, max(1, end_frame - start_frame + 1))
    indices = np.linspace(start_frame, end_frame, nframes).round().astype(int).tolist()

    pil_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(frame[:, :, ::-1])
            w, h = img.size
            if max(w, h) > max_resolution:
                scale = max_resolution / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            pil_frames.append(img)
    cap.release()
    return pil_frames


def encode_frame_to_base64(frame, quality=85):
    buf = io.BytesIO()
    frame.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_audio_from_wav(wav_path):
    """Read wav file and return base64 encoded WAV string."""
    import soundfile as sf
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def parse_emo_evaluation(session_path):
    """Parse EmoEvaluation files to get timestamps per utterance."""
    timestamps = {}
    emo_dir = session_path / "dialog" / "EmoEvaluation"
    for txt_file in emo_dir.glob("*.txt"):
        if txt_file.name.startswith("._"):
            continue
        with open(txt_file, errors="ignore") as f:
            for line in f:
                m = re.match(
                    r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(\S+)\s+(\w+)",
                    line.strip())
                if m:
                    start, end, utt_id, emo = m.groups()
                    timestamps[utt_id] = (float(start), float(end))
    return timestamps


def load_iemocap():
    """Load IEMOCAP with dev (Session1-4) / test (Session5) split."""
    samples = {"dev": [], "test": []}

    # Parse timestamps from all sessions
    all_timestamps = {}
    for sess in range(1, 6):
        sess_path = IEMOCAP_FULL / f"Session{sess}"
        ts = parse_emo_evaluation(sess_path)
        all_timestamps.update(ts)

    for split, csv_name in [("dev", "dev_6class.csv"),
                            ("test", "test_6class.csv")]:
        csv_path = IEMOCAP_LABELS / csv_name
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                wav_id = row["wav_id"]
                session = row["session"]
                dialog = wav_id.rsplit("_", 1)[0]
                emotion = row["emotion"]
                text = row["text"]

                wav_path = (IEMOCAP_FULL / session / "sentences" / "wav"
                            / dialog / f"{wav_id}.wav")
                avi_path = (IEMOCAP_FULL / session / "dialog" / "avi"
                            / "DivX" / f"{dialog}.avi")

                start_time, end_time = all_timestamps.get(wav_id, (None, None))

                samples[split].append({
                    "id": wav_id,
                    "ground_truth": emotion,
                    "utterance": text,
                    "wav_path": str(wav_path),
                    "avi_path": str(avi_path),
                    "start_time": start_time,
                    "end_time": end_time,
                })
    return samples


def build_mcq_messages(modality, utterance=None, frames_b64=None,
                       audio_b64=None):
    parts = set(modality.split("+"))
    content = []
    hints = []

    if "V" in parts and frames_b64:
        for b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",
                },
            })
        hints.append(f"{len(frames_b64)} video frames from the conversation")

    if "A" in parts and audio_b64:
        content.append({
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"},
        })
        hints.append("the audio clip of the utterance")

    if "T" in parts and utterance:
        hints.append(f'the text transcript: "{utterance}"')

    letter_map = {chr(65 + i): e for i, e in enumerate(EMOTIONS)}
    last = chr(65 + len(EMOTIONS) - 1)
    options = "\n".join(f"{chr(65+i)}. {e}" for i, e in enumerate(EMOTIONS))
    instruction = (
        f"Classify the emotion into exactly one of the following:\n"
        f"{options}\n\nReply with ONLY the letter (A-{last}), nothing else."
    )

    ctx = ""
    if hints:
        ctx = "Given " + ", and ".join(hints) + ".\n\n"
    content.append({"type": "text", "text": ctx + instruction})

    return [
        {"role": "system",
         "content": "You are an expert in emotion recognition."},
        {"role": "user", "content": content},
    ], letter_map


def extract_probs(top_logprobs, letter_map):
    raw = {}
    for item in top_logprobs:
        token = item.token.strip()
        if token in letter_map and token not in raw:
            raw[token] = item.logprob
    for letter in letter_map:
        if letter not in raw:
            raw[letter] = -100.0
    letters = sorted(raw.keys())
    logps = np.array([raw[l] for l in letters])
    logps -= logps.max()
    exps = np.exp(logps)
    probs = exps / exps.sum()
    return {letter_map[l]: float(p) for l, p in zip(letters, probs)}


def run_sample(client, model, sample, modality, max_frames=8):
    result = {
        "id": sample["id"],
        "ground_truth": sample["ground_truth"],
        "modality": modality,
    }
    parts = set(modality.split("+"))

    try:
        utterance = sample["utterance"] if "T" in parts else None
        frames_b64 = None
        audio_b64 = None

        if "V" in parts:
            avi_path = Path(sample["avi_path"])
            if avi_path.exists() and sample["start_time"] is not None:
                try:
                    frames = extract_video_frames(
                        avi_path, max_frames=max_frames,
                        start_time=sample["start_time"],
                        end_time=sample["end_time"])
                    if frames:
                        frames_b64 = [encode_frame_to_base64(f) for f in frames]
                except Exception as e:
                    result["prediction"] = "unknown"
                    result["error"] = f"video failed: {e}"
                    return result
            else:
                result["prediction"] = "unknown"
                result["error"] = f"video not found or no timestamp"
                return result

        if "A" in parts:
            wav_path = Path(sample["wav_path"])
            if wav_path.exists():
                try:
                    audio_b64 = extract_audio_from_wav(wav_path)
                except Exception as e:
                    result["prediction"] = "unknown"
                    result["error"] = f"audio failed: {e}"
                    return result
            else:
                result["prediction"] = "unknown"
                result["error"] = f"wav not found"
                return result

        messages, letter_map = build_mcq_messages(
            modality, utterance, frames_b64, audio_b64)

        _t_start = time.time()
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=1, temperature=0.0,
                    logprobs=True, top_logprobs=20, timeout=60)
                break
            except Exception as e:
                if attempt == 2:
                    result["prediction"] = "unknown"
                    result["error"] = f"API error: {e}"
                    return result
                time.sleep(2 ** attempt)

        raw_text = resp.choices[0].message.content or ""
        probs = {}
        if (resp.choices[0].logprobs
                and resp.choices[0].logprobs.content
                and resp.choices[0].logprobs.content[0].top_logprobs):
            probs = extract_probs(
                resp.choices[0].logprobs.content[0].top_logprobs, letter_map)
        else:
            pred_letter = raw_text.strip().upper()
            if pred_letter in letter_map:
                probs = {e: (1.0 if e == letter_map[pred_letter] else 0.0)
                         for e in EMOTIONS}
            else:
                probs = {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}

        result["prediction"] = max(probs, key=probs.get)
        result["probs"] = probs
        result["raw_output"] = raw_text
        result["prompt_tokens"] = getattr(resp.usage, "prompt_tokens", 0)
        result["inference_time"] = time.time() - _t_start

    except Exception as e:
        result["prediction"] = "unknown"
        result["error"] = str(e)

    return result


def run_split(client, model, split_samples, split_name, out_dir,
              max_workers=4):
    for mod in TIERS:
        tag = mod.replace("+", "_")
        out_path = out_dir / f"{split_name}_{tag}_probs.jsonl"

        done_ids = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass

        todo = [s for s in split_samples if s["id"] not in done_ids]
        if not todo:
            print(f"  [{split_name}/{mod}] All done ({len(done_ids)})")
            continue

        print(f"  [{split_name}/{mod}] Todo: {len(todo)}, "
              f"Resume: {len(done_ids)}")

        with open(out_path, "a") as out_f:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(run_sample, client, model, s, mod): s
                    for s in todo
                }
                pbar = tqdm(total=len(todo), desc=f"[{split_name}/{mod}]")
                for fut in as_completed(futures):
                    r = fut.result()
                    out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    out_f.flush()
                    pbar.update(1)
                pbar.close()



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="qwen2.5-omni-7b")
    p.add_argument("--output_dir", default="outputs/iemocap_qwen25")
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--iemocap_full", default="data/IEMOCAP_full_release",
                   help="IEMOCAP_full_release directory (Session1-5)")
    p.add_argument("--iemocap_labels", default="data/iemocap_splits",
                   help="directory with dev_6class.csv / test_6class.csv (scripts/prepare_iemocap.py)")
    args = p.parse_args()
    global IEMOCAP_FULL, IEMOCAP_LABELS
    IEMOCAP_FULL = Path(args.iemocap_full)
    IEMOCAP_LABELS = Path(args.iemocap_labels)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_iemocap()
    print(f"IEMOCAP 6-class: dev={len(data['dev'])}, test={len(data['test'])}")

    # Check availability
    for split in ["dev", "test"]:
        wav_miss = sum(1 for s in data[split]
                       if not Path(s["wav_path"]).exists())
        avi_miss = sum(1 for s in data[split]
                       if not Path(s["avi_path"]).exists())
        ts_miss = sum(1 for s in data[split]
                      if s["start_time"] is None)
        print(f"  {split}: wav_miss={wav_miss}, avi_miss={avi_miss}, "
              f"ts_miss={ts_miss}/{len(data[split])}")

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    try:
        models = [m.id for m in client.models.list().data]
        print(f"vLLM models: {models}")
    except Exception as e:
        print(f"Cannot reach vLLM: {e}")
        return

    for split_name in ["dev", "test"]:
        print(f"\n{'='*60}")
        print(f"  {split_name}: {len(data[split_name])} samples")
        run_split(client, args.model, data[split_name], split_name,
                  out_dir, args.max_workers)

    print(f"\nDone. Per-tier probabilities saved to {out_dir}. "
          f"Run router/emorouter.py --pred_dir {out_dir} to route.")


if __name__ == "__main__":
    main()

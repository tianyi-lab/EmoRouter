#!/usr/bin/env python3
"""Per-tier (T / T+A / T+A+V) inference on CMU-MOSEI with official valid/test splits.

Uses:
- Valid set (300 videos) for CP calibration
- Test set (678 videos) for evaluation
- Raw video segments from CMU-MOSEI/Raw/
- Text transcripts from CMU_MOSEI_TimestampedWords.csd
- Labels from CMU_MOSEI_Labels.csd
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import cv2
import io
import base64
import os
import subprocess
import tempfile
from PIL import Image


def extract_video_frames(video_path, max_frames=8, max_resolution=512):
    """Extract frames using cv2 (avoids decord hang)."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nframes = min(max_frames, max(1, total_frames))
    indices = np.linspace(0, total_frames - 1, nframes).round().astype(int).tolist()
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


def extract_audio_from_mp4(video_path, sr=16000, timeout=30):
    """Extract audio with timeout to avoid ffmpeg hang."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path),
             "-vn", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
             tmp_path],
            capture_output=True, check=True, timeout=timeout,
        )
        import soundfile as sf
        data, sr_out = sf.read(tmp_path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, data, sr_out, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

MOSEI_BASE = Path("data/CMU-MOSEI")  # overridden by --mosei_base
EMOTIONS = ["negative", "neutral", "positive"]
TIERS = ["T", "T+A", "T+A+V"]


def load_mosei():
    """Load MOSEI samples with official splits and text transcripts."""
    # Load standard folds
    # Standard folds from the CMU-MultimodalSDK (mmsdk/.../CMU_MOSEI/cmu_mosei_std_folds.py)
    folds = json.load(open(Path(__file__).with_name("mosei_std_folds.json")))
    valid_ids = set(folds["standard_valid_fold"])
    test_ids = set(folds["standard_test_fold"])

    # Load labels
    labels_f = h5py.File(
        MOSEI_BASE / "CMU-MOSEI/labels/CMU_MOSEI_Labels.csd", "r")
    labels_data = labels_f["All Labels"]["data"]

    # Load words
    words_f = h5py.File(
        MOSEI_BASE / "CMU-MOSEI/languages/CMU_MOSEI_TimestampedWords.csd", "r")
    words_data = words_f["words"]["data"]

    raw_dir = MOSEI_BASE / "Raw"

    samples = {"valid": [], "test": []}

    for vid_id in labels_data.keys():
        if vid_id in valid_ids:
            split = "valid"
        elif vid_id in test_ids:
            split = "test"
        else:
            continue

        seg_intervals = labels_data[vid_id]["intervals"][:]
        seg_labels = labels_data[vid_id]["features"][:]

        # Get raw video files sorted by segment ID
        vid_dir = raw_dir / vid_id
        raw_files = sorted(vid_dir.glob("*.mp4"),
                           key=lambda x: int(x.stem)) if vid_dir.exists() else []

        # Sort label segments by start time to match raw files
        time_order = seg_intervals[:, 0].argsort()

        # Get words for this video
        word_feats = None
        word_intervals = None
        if vid_id in words_data:
            word_feats = words_data[vid_id]["features"][:]
            word_intervals = words_data[vid_id]["intervals"][:]

        for rank, seg_idx in enumerate(time_order):
            start, end = seg_intervals[seg_idx]
            sent = seg_labels[seg_idx][0]  # sentiment score

            if sent < 0:
                label = "negative"
            elif sent > 0:
                label = "positive"
            else:
                label = "neutral"

            # Extract text from words
            text = ""
            if word_feats is not None:
                words = []
                for j in range(len(word_feats)):
                    w = word_feats[j][0]
                    if isinstance(w, bytes):
                        w = w.decode()
                    wi = word_intervals[j]
                    if wi[0] >= start - 0.1 and wi[1] <= end + 0.1 and w != "sp":
                        words.append(w)
                text = " ".join(words)

            # Video path: match sorted raw files to time-sorted labels
            if rank < len(raw_files):
                video_path = raw_files[rank]
            else:
                video_path = vid_dir / f"{seg_idx}.mp4"

            seg_name = video_path.stem if video_path.exists() else str(seg_idx)
            samples[split].append({
                "id": f"{vid_id}_{seg_name}",
                "ground_truth": label,
                "utterance": text,
                "video_path": str(video_path),
            })

    labels_f.close()
    words_f.close()
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
        f"Classify the sentiment into exactly one of the following:\n"
        f"{options}\n\nReply with ONLY the letter (A-{last}), nothing else."
    )

    ctx = ""
    if hints:
        ctx = "Given " + ", and ".join(hints) + ".\n\n"
    content.append({"type": "text", "text": ctx + instruction})

    return [
        {"role": "system",
         "content": "You are an expert in sentiment analysis."},
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
    video_path = Path(sample["video_path"])

    if (parts & {"A", "V"}) and not video_path.exists():
        result["prediction"] = "unknown"
        result["error"] = f"video not found: {video_path}"
        return result

    try:
        utterance = sample["utterance"] if "T" in parts else None
        frames_b64 = None
        audio_b64 = None

        if "V" in parts and video_path.exists():
            try:
                frames = extract_video_frames(video_path, max_frames=max_frames)
                frames_b64 = [encode_frame_to_base64(f) for f in frames]
            except Exception:
                result["prediction"] = "unknown"
                result["error"] = f"video decode failed: {video_path.name}"
                return result

        if "A" in parts and video_path.exists():
            try:
                audio_b64 = extract_audio_from_mp4(video_path)
            except Exception:
                result["prediction"] = "unknown"
                result["error"] = f"audio extract failed: {video_path.name}"
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
    p.add_argument("--output_dir", default="outputs/mosei_qwen25")
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--mosei_base", default="data/CMU-MOSEI",
                   help="CMU-MOSEI root (contains CMU-MOSEI/labels, CMU-MOSEI/languages, Raw/)")
    args = p.parse_args()
    global MOSEI_BASE
    MOSEI_BASE = Path(args.mosei_base)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_mosei()
    print(f"MOSEI official splits: valid={len(data['valid'])}, "
          f"test={len(data['test'])}")

    # Check video availability
    for split in ["valid", "test"]:
        missing = sum(1 for s in data[split]
                      if not Path(s["video_path"]).exists())
        total = len(data[split])
        if missing:
            print(f"  {split}: {missing}/{total} videos missing")
        else:
            print(f"  {split}: all {total} videos available")

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    try:
        models = [m.id for m in client.models.list().data]
        print(f"vLLM models: {models}")
    except Exception as e:
        print(f"Cannot reach vLLM: {e}")
        return

    for split_name in ["valid", "test"]:
        print(f"\n{'='*60}")
        print(f"  {split_name}: {len(data[split_name])} samples")
        run_split(client, args.model, data[split_name], split_name,
                  out_dir, args.max_workers)

    print(f"\nDone. Per-tier probabilities saved to {out_dir}. "
          f"Run router/emorouter.py --pred_dir {out_dir} to route.")


if __name__ == "__main__":
    main()

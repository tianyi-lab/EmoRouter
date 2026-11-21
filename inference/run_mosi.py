#!/usr/bin/env python3
"""Per-tier (T / T+A / T+A+V) inference on CMU-MOSI with official valid/test splits.

3-class: negative (score < 0) / neutral (score = 0) / positive (score > 0)
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import io
import os
import base64
import subprocess
import tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm

MOSI_BASE = Path("data/CMU-MOSI")  # overridden by --mosi_base
EMOTIONS = ["negative", "neutral", "positive"]
TIERS = ["T", "T+A", "T+A+V"]


def extract_video_frames(video_path, max_frames=8, max_resolution=512):
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


def load_mosi():
    """Load MOSI samples with official splits (3-class)."""
    samples = {"train": [], "valid": [], "test": []}
    raw_dir = MOSI_BASE / "Raw"

    with open(MOSI_BASE / "label.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row["label"])
            label = "negative" if score < 0 else ("positive" if score > 0 else "neutral")
            split = row["mode"]
            video_path = raw_dir / row["video_id"] / f"{row['clip_id']}.mp4"

            samples[split].append({
                "id": f"{row['video_id']}_{row['clip_id']}",
                "ground_truth": label,
                "utterance": row["text"],
                "video_path": str(video_path),
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
        f"Classify the sentiment into exactly one of the following:\n"
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
            except Exception as e:
                result["prediction"] = "unknown"
                result["error"] = f"video decode failed: {video_path.name}: {e}"
                return result

        if "A" in parts and video_path.exists():
            try:
                audio_b64 = extract_audio_from_mp4(video_path)
            except Exception as e:
                result["prediction"] = "unknown"
                result["error"] = f"audio extract failed: {video_path.name}: {e}"
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
    p.add_argument("--output_dir", default="outputs/mosi_qwen25")
    p.add_argument("--max_workers", type=int, default=32)
    p.add_argument("--mosi_base", default="data/CMU-MOSI",
                   help="CMU-MOSI root (contains label.csv and Raw/)")
    args = p.parse_args()
    global MOSI_BASE
    MOSI_BASE = Path(args.mosi_base)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_mosi()
    print(f"MOSI 3-class: train={len(data['train'])}, "
          f"valid={len(data['valid'])}, test={len(data['test'])}")

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

#!/usr/bin/env python3
"""Build the 6-class IEMOCAP splits used by EmoRouter.

Classes: angry, excited, frustrated, happy, neutral, sad
Split:   Session1-4 -> dev (calibration), Session5 -> test

Usage:
    python scripts/prepare_iemocap.py \
        --iemocap_root data/IEMOCAP_full_release --out_dir data/iemocap_splits
"""

import argparse
import re
from pathlib import Path

import pandas as pd

EMO_MAP = {"ang": "angry", "exc": "excited", "fru": "frustrated",
           "hap": "happy", "neu": "neutral", "sad": "sad"}
LINE_RE = re.compile(r"\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(\S+)\s+(\w+)")
TRANS_RE = re.compile(r"(\S+)\s+\[[\d\.\-]+\]:\s*(.*)")


def parse_session(root, session):
    sess = root / session
    labels = {}
    for f in sorted((sess / "dialog" / "EmoEvaluation").glob("*.txt")):
        if f.name.startswith("._"):
            continue
        for line in open(f, errors="ignore"):
            m = LINE_RE.match(line.strip())
            if m:
                start, end, utt, emo = m.groups()
                labels[utt] = (float(start), float(end), emo)
    texts = {}
    for f in sorted((sess / "dialog" / "transcriptions").glob("*.txt")):
        if f.name.startswith("._"):
            continue
        for line in open(f, errors="ignore"):
            m = TRANS_RE.match(line.strip())
            if m:
                texts[m.group(1)] = m.group(2).strip()

    rows = []
    for utt, (start, end, emo) in sorted(labels.items()):
        if emo not in EMO_MAP or utt not in texts:
            continue
        dialog = utt.rsplit("_", 1)[0]
        wav = sess / "sentences" / "wav" / dialog / f"{utt}.wav"
        avi = sess / "dialog" / "avi" / "DivX" / f"{dialog}.avi"
        if not wav.exists():
            continue
        rows.append({"wav_id": utt, "dialog": dialog, "session": session,
                     "speaker": f"Speaker_{utt.rsplit('_', 1)[1][0]}",
                     "text": texts[utt], "emotion": EMO_MAP[emo],
                     "start_time": start, "end_time": end,
                     "wav_path": str(wav), "avi_path": str(avi)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iemocap_root", default="data/IEMOCAP_full_release")
    p.add_argument("--out_dir", default="data/iemocap_splits")
    args = p.parse_args()
    root, out = Path(args.iemocap_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([r for s in range(1, 6) for r in parse_session(root, f"Session{s}")])
    dev, test = df[df.session != "Session5"], df[df.session == "Session5"]
    dev.to_csv(out / "dev_6class.csv", index=False)
    test.to_csv(out / "test_6class.csv", index=False)
    print(f"dev={len(dev)}  test={len(test)}")
    print(df.groupby("session").emotion.value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EmoRouter: conformal-prediction routing over the T -> T+A -> T+A+V cascade.

Given the per-tier class probabilities produced by `inference/run_*.py`, this script
  1. calibrates split-CP thresholds on the calibration split (dev / valid),
  2. selects the per-tier error budgets (alpha_1, alpha_2) with
     Conformal Set Contraction (CSC), using calibration data only (no labels of the
     routing outcome, no latency measurements),
  3. routes every test utterance: exit at the first tier whose prediction set has
     at most one label, otherwise escalate; T+A+V always answers,
  4. reports wF1, latency reduction rate (LRR), time utility (U_t) and the exit-tier
     distribution.

Example:
    python router/emorouter.py --pred_dir outputs/meld_qwen25 --dataset meld
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

LABELS = {
    "meld": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "iemocap": ["angry", "excited", "frustrated", "happy", "neutral", "sad"],
    "mosei": ["negative", "neutral", "positive"],
    "mosi": ["negative", "neutral", "positive"],
    "chsims": ["Negative", "Neutral", "Positive"],
}
TIERS = ["T", "T_A", "T_A_V"]
ALPHA_GRID = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


# ----------------------------------------------------------------------------- data
def _load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "probs" in r and "inference_time" in r:
                out[r["id"]] = r
    return out


def load_split(pred_dir, split, labels):
    """Return (ids, P, t, y): P[tier] is an (n, K) probability matrix,
    t[tier] the per-utterance latency (s) of that tier, y the gold label indices."""
    pred_dir = Path(pred_dir)
    if split == "cal":
        split = "valid" if (pred_dir / "valid_T_probs.jsonl").exists() else "dev"
    D = {t: _load_jsonl(pred_dir / f"{split}_{t}_probs.jsonl") for t in TIERS}
    ids = [i for i in D["T"] if all(i in D[t] for t in TIERS)
           and D["T"][i]["ground_truth"] in labels]
    P = {t: np.array([[D[t][i]["probs"].get(c, 0.0) for c in labels] for i in ids]) for t in TIERS}
    tm = {t: np.array([D[t][i]["inference_time"] for i in ids]) for t in TIERS}
    y = np.array([labels.index(D["T"][i]["ground_truth"]) for i in ids])
    return ids, P, tm, y


# ------------------------------------------------------------------ conformal prediction
def qhat(P, y, alpha):
    """Split-CP quantile of the nonconformity score s = 1 - p_y."""
    s = np.sort(1.0 - P[np.arange(len(y)), y])
    n = len(s)
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))) - 1, 0), n - 1)
    return s[k]


def set_size(P, q):
    """|C(x)| with C(x) = {y : p_y(x) >= 1 - q}."""
    return (P >= 1.0 - q).sum(1)


def csc_score(s1, s2):
    """Conformal Set Contraction: mean set-size drop from T to T+A over utterances
    not resolved at T, and from T+A to the singleton over utterances not resolved at T+A."""
    n1 = s1 > 1
    d1 = (s1[n1] - s2[n1]).mean() if n1.any() else 0.0
    n2 = n1 & (s2 > 1)
    d2 = (s2[n2] - 1).mean() if n2.any() else 0.0
    return (d1 + d2) / 2


def select_alphas(Pc, yc, grid=ALPHA_GRID):
    """CSC: pick (alpha_1, alpha_2) on the calibration split maximising csc_score."""
    S1 = {a: set_size(Pc["T"], qhat(Pc["T"], yc, a)) for a in grid}
    S2 = {a: set_size(Pc["T_A"], qhat(Pc["T_A"], yc, a)) for a in grid}
    best, arg = -np.inf, None
    for a1 in grid:
        for a2 in grid:
            v = csc_score(S1[a1], S2[a2])
            if v > best + 1e-12:
                best, arg = v, (float(a1), float(a2))
    return arg


def route(Pc, yc, Pt, a1, a2):
    """Exit tier (1/2/3) for each test utterance. A set with <= 1 label exits;
    an empty set exits with the top-1 label of that tier."""
    q1, q2 = qhat(Pc["T"], yc, a1), qhat(Pc["T_A"], yc, a2)
    exit1 = set_size(Pt["T"], q1) <= 1
    exit2 = set_size(Pt["T_A"], q2) <= 1
    tier = np.full(len(exit1), 3)
    tier[exit2] = 2
    tier[exit1] = 1
    return tier


# ------------------------------------------------------------------------- metrics
def wf1(y, pred, K):
    return 100 * f1_score(y, pred, labels=list(range(K)), average="weighted", zero_division=0)


def evaluate(tier, Pt, tt, yt, K):
    """wF1 of the routed predictions, LRR (%) w.r.t. always running T+A+V, and
    time utility U_t = (wF1 - wF1_T) / (1 - LRR)."""
    pred = np.where(tier == 1, Pt["T"].argmax(1),
                    np.where(tier == 2, Pt["T_A"].argmax(1), Pt["T_A_V"].argmax(1)))
    cost = tt["T"] + (tier >= 2) * tt["T_A"] + (tier >= 3) * tt["T_A_V"]
    w = wf1(yt, pred, K)
    lrr = 100 * (1 - cost.mean() / tt["T_A_V"].mean())
    ut = (w - wf1(yt, Pt["T"].argmax(1), K)) / max(1 - lrr / 100, 1e-9)
    return {"wF1": w, "LRR": lrr, "Ut": ut,
            "exit_pct": [100 * float((tier == k).mean()) for k in (1, 2, 3)]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_dir", required=True, help="output dir of inference/run_<dataset>.py")
    p.add_argument("--dataset", required=True, choices=sorted(LABELS))
    p.add_argument("--alpha1", type=float, help="fix alpha_1 instead of CSC selection")
    p.add_argument("--alpha2", type=float, help="fix alpha_2 instead of CSC selection")
    args = p.parse_args()

    labels = LABELS[args.dataset]
    K = len(labels)
    _, Pc, _, yc = load_split(args.pred_dir, "cal", labels)
    _, Pt, tt, yt = load_split(args.pred_dir, "test", labels)

    if args.alpha1 is not None and args.alpha2 is not None:
        a1, a2 = args.alpha1, args.alpha2
    else:
        a1, a2 = select_alphas(Pc, yc)
    tier = route(Pc, yc, Pt, a1, a2)
    r = evaluate(tier, Pt, tt, yt, K)

    print(f"calibration n={len(yc)}  test n={len(yt)}")
    print(f"alpha_1={a1:.3f}  alpha_2={a2:.3f}" + ("" if args.alpha1 is not None else "  (selected by CSC)"))
    print(f"T only : wF1={wf1(yt, Pt['T'].argmax(1), K):.1f}")
    print(f"T+A+V  : wF1={wf1(yt, Pt['T_A_V'].argmax(1), K):.1f}")
    print(f"EmoRouter: wF1={r['wF1']:.1f}  LRR={r['LRR']:.1f}%  U_t={r['Ut']:.1f}")
    print("exit at T / T+A / T+A+V: " + " / ".join(f"{x:.1f}%" for x in r["exit_pct"]))


if __name__ == "__main__":
    main()

"""Single-frame static information probe — "can a conductor be identified from one skeleton frame?"

Background: the shuffle test (V5d_shuffle 0.345) only proves that "a model trained on
motion collapses without motion". The ceiling of a model trained from the start to mine
static cues alone is a separate question — in particular the possibility of memorizing
camera angles (H2). Verdict criteria (pre-registered, agreed 2026-08-20):
frame accuracy <=0.40 negligible / 0.40-0.55 intermediate / >0.55 substantive leakage.

Design (same conditions as the main experiment V5d_both):
  - same snapshot CSV, segment exclusion, session merges, session-group 5-fold (seed 0)
  - features = one frame of build_channels(pos+bone): x, y, conf, bone vector, foreshortening r
    (dr is a temporal derivative and thus excluded — static information only)
  - conditions: full (306-dim) / coords (conf and r removed) / conf only / fore (foreshortening=angle) only
  - models: logistic regression (linear ceiling) + MLP 256-128 (nonlinear ceiling), both sklearn
  - train 32 random frames/run, eval 8 equally spaced frames/run -> probability summation to recording/session

Usage:
    PYTHONPATH=src python scripts/static_frame_probe.py \
        --csv data/experiments/snapshot5_20260812.csv \
        --seg-verdicts data/experiments/segver_20260813.csv \
        --session-merges data/experiments/session_merges_20260813.csv
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import conductor_classifier.channels as CH
from conductor_classifier.segver import load_blacklist, load_video_exclude
from conductor_classifier.sessions import (apply_merge_overrides, group_sessions,
                                           load_duplicate_clusters,
                                           load_merge_overrides)
from train_pilot import RunCache, load_runs

# build_channels(use=("pos","bone")) output channels: [x, y, conf, vecx, vecy, r, dr]
FEATS = {
    "full":   [0, 1, 2, 3, 4, 5],   # all static features — H1 ceiling
    "coords": [0, 1, 3, 4],         # pose + angle (coordinate frame) — conf and r removed
    "conf":   [2],                  # confidence only — video-quality proxy check
    "fore":   [5],                  # foreshortening only — camera-angle check (H2)
}
# Configuration matching the 11-channel main experiment (no depth channel, adopted 2026-08-27):
# only 5 features (x, y, conf, vecx, vecy) remain, so the r-related conditions disappear.
FEATS_NODEPTH = {
    "full":   [0, 1, 2, 3, 4],
    "coords": [0, 1, 3, 4],
    "conf":   [2],
}
N_TRAIN_FRAMES = 32
N_EVAL_FRAMES = 8


def frame_features(cache, run: dict, joints: str,
                   no_depth: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Extract (train frames, eval frames) features from the run's clean segments.

    To keep the feature definition fully identical to the main experiment, we run
    build_channels over the whole segment and then pick frame columns. dr (the last
    channel) is discarded. With joints="body" the 42 hand joints are dropped entirely
    and only up to the wrists remains (applied identically to train and eval — with
    different input definitions no comparison is possible).
    """
    seq = cache[run["run"]]
    cols = []                       # (7, V) per-frame features
    for a, b in run["segments"]:
        use = ("pos", "bonedir") if no_depth else ("pos", "bone")
        n_keep = 5 if no_depth else 6               # for bone, drop the trailing dr
        ch = CH.build_channels(seq[a:b], joints, use=use)
        cols.append(ch[:n_keep].transpose(1, 0, 2))
    allf = np.concatenate(cols)                     # (N,6,V)
    # zlib.crc32: a cross-process stable seed — Python's hash() is salted differently
    # per run, so different frames get sampled each execution and reproducibility
    # breaks (discovered 2026-08-23).
    import zlib
    rng = np.random.default_rng(zlib.crc32(run["run"].encode()))
    tr_idx = rng.choice(len(allf), size=min(N_TRAIN_FRAMES, len(allf)),
                        replace=False)
    ev_idx = np.linspace(0, len(allf) - 1, N_EVAL_FRAMES).round().astype(int)
    return allf[tr_idx], allf[ev_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/experiments/snapshot5_20260812.csv")
    ap.add_argument("--root", default="skeletons_A")
    ap.add_argument("--seg-verdicts", default="data/experiments/segver_20260813.csv")
    ap.add_argument("--session-merges",
                    default="data/experiments/session_merges_20260813.csv")
    ap.add_argument("--dup-drops", default="",
                    help="true-duplicate cluster CSV — keep only one representative (same as the main experiment)")
    ap.add_argument("--ood-exclude", default="",
                    help="training-exclusion video CSV — protects the OOD held-out set")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joints", default="both", choices=["both", "body"],
                    help="body = drop hand joints, keep only up to the wrists (9 joints)")
    ap.add_argument("--no-depth", action="store_true",
                    help="without the depth (foreshortening) channel — same configuration as the 11-channel main experiment")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = f"data/experiments/SF_static_frame_{args.joints}.json" \
            if args.joints != "both" else "data/experiments/SF_static_frame.json"

    seg_bl = load_blacklist(args.seg_verdicts)
    runs = load_runs(Path(args.csv), Path(args.root), 50, (), seg_blacklist=seg_bl)
    # The corpus must be identical to the main experiment for the comparison to hold —
    # apply true-duplicate dropping and OOD exclusion with the same rules as
    # train_pilot (user's point, 2026-08-27).
    if args.dup_drops:
        from train_pilot import _video_quality
        dup = load_duplicate_clusters(args.dup_drops)
        cluster_vids = {(c, v) for c, gs in dup.items() for g in gs for v in g}
        quality = _video_quality(runs, cluster_vids)
        footage: dict = defaultdict(int)
        for r in runs:
            k = (r["conductor"], r["video_id"])
            if k in cluster_vids:
                footage[k] += sum(b - a for a, b in r["segments"])
        drop = set()
        for c, groups in dup.items():
            for g in groups:
                present = [(c, v) for v in g if (c, v) in quality]
                if len(present) < 2:
                    continue
                best = max(present, key=lambda k: (footage[k], quality[k]))
                drop |= {k for k in present if k != best}
        before = len(runs)
        runs = [r for r in runs if (r["conductor"], r["video_id"]) not in drop]
        print(f"True-duplicate drop: {len(drop)} videos excluded (runs {before}->{len(runs)})")
    if args.ood_exclude:
        excl = load_video_exclude(args.ood_exclude)
        before = len(runs)
        runs = [r for r in runs if (r["conductor"], r["video_id"]) not in excl]
        print(f"OOD exclusion: {len(excl)} videos listed -> runs {before}->{len(runs)}")
    classes = sorted({r["conductor"] for r in runs})
    lab = {c: i for i, c in enumerate(classes)}
    recs = sorted({(r["conductor"], r["video_id"]) for r in runs})
    print(f"runs {len(runs)} · recordings {len(recs)} · classes {classes}")

    # Session grouping + folds — must match train_pilot main verbatim for the comparison to hold
    titles = {}
    for m in csv.DictReader(Path("data/recordings_meta.csv").open()):
        titles.setdefault(m["conductor"], {})[m["video_id"].strip()] = m["title"]
    merge_ov = load_merge_overrides(args.session_merges)
    sess_of = {}
    for c in classes:
        vids = {r["video_id"] for r in runs if r["conductor"] == c}
        tmap = {v: titles.get(c, {}).get(v, "") for v in vids}
        g = group_sessions(tmap)
        if c in merge_ov:
            g = apply_merge_overrides(g, merge_ov[c])
        for v, sid in g.items():
            sess_of[(c, v)] = (c, sid)
    sessions = sorted({sess_of[k] for k in recs})
    # Folds — the same stratified group split as train_pilot main (reflects the
    # stratification introduced 2026-08-25): groups (sessions) kept intact + conductor
    # ratios equalized across folds. Arguments and their order must match exactly for
    # the folds to line up with the GCN experiments.
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.array([lab[c] for c, _ in recs])
    groups = np.array([str(sess_of[k]) for k in recs])
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                random_state=args.seed)
    fold_of = {}
    for f, (_, te_idx) in enumerate(sgkf.split(np.arange(len(recs)), y, groups)):
        for i in te_idx:
            fold_of[recs[i]] = f
    print(f"sessions {len(sessions)} · stratified group folds seed {args.seed}")

    print("Extracting features (build_channels pos+bone, dr excluded)…")
    t0 = time.time()
    cache = RunCache(runs, depth=not args.no_depth, depth_anchor="median")
    tr_feats, ev_feats = {}, {}
    for r in runs:
        tr_feats[r["run"]], ev_feats[r["run"]] = frame_features(
            cache, r, args.joints, args.no_depth)
    print(f"  {time.time()-t0:.0f}s")

    def make_xy(rs, feats, chans):
        X = np.concatenate([feats[r["run"]][:, chans, :].reshape(
            len(feats[r["run"]]), -1) for r in rs])
        y = np.concatenate([[lab[r["conductor"]]] * len(feats[r["run"]])
                            for r in rs])
        return X, y

    results = {}
    for fname, chans in (FEATS_NODEPTH if args.no_depth else FEATS).items():
        for mname in ("linear", "mlp"):
            fr_c = fr_n = 0
            cm = np.zeros((len(classes),) * 2, int)
            rec_p = defaultdict(lambda: np.zeros(len(classes)))
            t0 = time.time()
            for f in range(args.folds):
                tr = [r for r in runs if fold_of[(r["conductor"], r["video_id"])] != f]
                te = [r for r in runs if fold_of[(r["conductor"], r["video_id"])] == f]
                Xtr, ytr = make_xy(tr, tr_feats, chans)
                sc = StandardScaler().fit(Xtr)
                if mname == "linear":
                    m = LogisticRegression(max_iter=2000, n_jobs=-1)
                else:
                    m = MLPClassifier((256, 128), max_iter=300, tol=1e-5,
                                      random_state=0)
                m.fit(sc.transform(Xtr), ytr)
                for r in te:
                    P = m.predict_proba(sc.transform(
                        ev_feats[r["run"]][:, chans, :].reshape(N_EVAL_FRAMES, -1)))
                    y = lab[r["conductor"]]
                    for p in P:
                        cm[y, int(np.argmax(p))] += 1
                    fr_c += int((P.argmax(1) == y).sum()); fr_n += len(P)
                    rec_p[(r["conductor"], r["video_id"])] += P.sum(0)
            # frame -> recording -> session (same probability-summation rule as the main experiment)
            rec_ok = sum(int(np.argmax(p)) == lab[c] for (c, v), p in rec_p.items())
            sess_p = defaultdict(lambda: np.zeros(len(classes)))
            for (c, v), p in rec_p.items():
                sess_p[sess_of[(c, v)]] += p / p.sum()
            sess_ok = sum(int(np.argmax(p)) == lab[s[0]] for s, p in sess_p.items())
            f1s = []
            for c in range(len(classes)):
                tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c].sum() - tp
                pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
                f1s.append(2 * pr * rc / max(1e-9, pr + rc))
            key = f"{fname}_{mname}"
            results[key] = {
                "preds": {f"{c}|{v}": {"p": (p / p.sum()).round(4).tolist(),
                                       "y": lab[c]}
                          for (c, v), p in rec_p.items()},
                "frame_acc": round(fr_c / fr_n, 4), "frame_n": fr_n,
                "frame_macro_f1": round(float(np.mean(f1s)), 4),
                "rec_acc": round(rec_ok / len(rec_p), 4), "rec_n": len(rec_p),
                "sess_acc": round(sess_ok / len(sess_p), 4), "sess_n": len(sess_p),
                "cm_frame": cm.tolist(), "minutes": round((time.time()-t0)/60, 1),
            }
            r = results[key]
            print(f"[{key:14s}] frame {r['frame_acc']:.3f} "
                  f"(F1 {r['frame_macro_f1']:.3f}) · recording {r['rec_acc']:.3f} · "
                  f"session {r['sess_acc']:.3f} · {r['minutes']} min")

    frame_chance = max(np.bincount(
        [lab[r["conductor"]] for r in runs for _ in range(N_EVAL_FRAMES)]
    )) / (len(runs) * N_EVAL_FRAMES)
    out = {"classes": classes, "csv": args.csv, "seed": args.seed,
           "joints": args.joints,
           "folds": args.folds, "n_train_frames": N_TRAIN_FRAMES,
           "n_eval_frames": N_EVAL_FRAMES,
           "frame_chance": round(float(frame_chance), 4),
           "rec_chance": round(max(sum(1 for c, _ in recs if c == cc)
                                   for cc in classes) / len(recs), 4),
           "results": results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\nSaved: {args.out} · frame chance {frame_chance:.3f}")


if __name__ == "__main__":
    main()

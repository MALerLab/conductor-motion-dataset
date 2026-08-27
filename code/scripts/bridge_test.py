"""Bridge test -- feed a third conductor into the B-vs-D model to judge whether era confounds the prediction.

B (1959-1989) and D (2007-2014) don't overlap in time, so the two classes alone
can't separate "conducting style" from "recording era." We run Haitink
(1959-2021), who spans both eras, through the model to see whether the
prediction **tracks the recording year**. Haitink's identity label is not needed.

Procedure: train on all of B+D (no holdout needed -- Haitink is an unseen class
by construction) -> compute P(Dudamel) for every Haitink run -> average per
recording -> correlate and AUC against year.

Usage:
    PYTHONPATH=src python scripts/bridge_test.py --third-root skeletons_H \
        --third Haitink --epochs 30
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

from conductor_classifier import bridge as BR
from conductor_classifier import channels as CH
from conductor_classifier import graph as G
from conductor_classifier import windows as W
from conductor_classifier.resgcn import ResGCN
from train_pilot import (DST_FPS, EvalWindows, RunCache, TrainWindows,
                         load_runs, supcon_loss)


def load_third(root: Path, conductor: str, min_seg_frames: int) -> list[dict]:
    """Third-conductor runs -- read directly from the skeletons folder, no review CSV."""
    out = []
    for m in sorted(root.glob(f"{conductor}/*/meta.json")):
        d = m.parent
        npy, qcf = d / "skeleton_norm.npy", d / "qc.json"
        if not (npy.exists() and qcf.exists()):
            continue
        qc = json.loads(qcf.read_text())
        segs = W.rescale_segments([tuple(s) for s in qc["segments"]],
                                  qc["fps"], DST_FPS)
        segs = [(a, b) for a, b in segs if b - a >= min_seg_frames]
        if not segs:
            continue
        meta = json.loads((d / "meta.json").read_text())
        out.append({"conductor": conductor, "video_id": meta["video_id"],
                    "run": d.name, "npy": npy, "fps": qc["fps"], "segments": segs})
    return out


def load_years(meta_csv: Path, conductor: str) -> dict[str, int]:
    """Recording year -- prefer year_manual (human-entered), fall back to year_desc (auto-extracted from description)."""
    out = {}
    for r in csv.DictReader(meta_csv.open()):
        if r["conductor"] != conductor:
            continue
        for k in ("year_manual", "year_desc", "year_title"):
            v = (r.get(k) or "").strip().rstrip("s")
            if v[:4].isdigit():
                out[r["video_id"].strip()] = int(v[:4])
                break
    return out


def train_full(runs, cache, labels, args, device):
    A = G.adjacency(args.joints, coord=getattr(args, "coord", False))
    use = tuple(args.use.split(","))
    C = sum(CH.BRANCH_WIDTH[b] for b in use)
    model = ResGCN(C, len(labels), A).to(device)
    ds = TrainWindows(runs, cache, labels, args.joints, use, args.T,
                      args.windows_per_run, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, persistent_workers=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl)))
    model.train()
    for ep in range(args.epochs):
        tot = n = 0
        for x, y, *_ in dl:
            x, y = x.to(device, non_blocking=True), y.to(device)
            logit, emb = model(x, return_embedding=True)
            loss = Fn.cross_entropy(logit, y) + args.supcon * supcon_loss(emb, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(y); n += len(y)
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep + 1}/{args.epochs} loss {tot / max(1, n):.3f}")
    return model


def score_runs(model, runs, cache, labels, args, device, pos_class: int):
    """For every run window, P(pos_class) -> list of (recording_id, probability)."""
    use = tuple(args.use.split(","))
    ev = EvalWindows(runs, cache, labels, args.joints, use, args.T, args.n_eval,
                     seed=args.seed)
    dl = DataLoader(ev, batch_size=args.batch, num_workers=4)
    out = []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x, vid = batch[0], batch[2]
            p = Fn.softmax(model(x.to(device)), dim=1).cpu().numpy()[:, pos_class]
            out += list(zip(vid, p.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/review/run_verdicts.csv")
    ap.add_argument("--meta", default="data/recordings_meta.csv")
    ap.add_argument("--third-root", default="skeletons_H")
    ap.add_argument("--third", default="Haitink")
    ap.add_argument("--joints", default="both")
    ap.add_argument("--only", nargs="*", default=["Bernstein", "Dudamel"],
                    help="Conductors to train on -- the third party must always be excluded")
    ap.add_argument("--use", default="pos,vel,acc,bone")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--windows-per-run", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--supcon", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cutoff", type=int, default=2000)
    ap.add_argument("--out", default="data/experiments/bridge.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_runs(Path(args.csv), Path("."), args.T, tuple(args.only))
    if args.third in {r["conductor"] for r in base}:
        raise SystemExit(f"[aborted] third party {args.third} is included in the training set -- check --only")
    labels = {c: i for i, c in enumerate(sorted({r["conductor"] for r in base}))}
    third = load_third(Path(args.third_root), args.third, args.T)
    years = load_years(Path(args.meta), args.third)
    third_years = {r["video_id"] for r in third} & set(years)
    print(f"train {len(base)} run / {args.third} {len(third)} run "
          f"({len({r['video_id'] for r in third})} recordings, {len(third_years)} with a year) · {device}")

    t0 = time.time()
    cache = RunCache(base)
    model = train_full(base, cache, labels, args, device)

    # Sanity check on the training data itself (confirms training worked; carries optimistic bias)
    self_scores = BR.aggregate_scores(
        score_runs(model, base, cache, labels, args, device, pos_class=1))
    truth = {r["video_id"]: r["conductor"] for r in base}
    dud = [s for k, s in self_scores.items() if truth[k] == "Dudamel"]
    ber = [s for k, s in self_scores.items() if truth[k] == "Bernstein"]
    print(f"training-set check: P(Dudamel) median -- Dudamel {np.median(dud):.2f} / "
          f"Bernstein {np.median(ber):.2f}")

    cache3 = RunCache(third)
    scores = BR.aggregate_scores(
        score_runs(model, third, cache3, labels, args, device, pos_class=1))
    common = {k: v for k, v in scores.items() if k in years}
    xs = [years[k] for k in common]
    ys = [common[k] for k in common]
    rho = BR.spearman(xs, ys)
    auc = BR.era_auc(common, years, cutoff=args.cutoff)
    old = [v for k, v in common.items() if years[k] < args.cutoff]
    new = [v for k, v in common.items() if years[k] >= args.cutoff]

    # Power check: scores need to be spread out for "independent of year" to mean anything
    sv = np.array(list(common.values()))
    # Positive control: the training set (B) itself spans multiple eras, so check within it too
    b_years = load_years(Path(args.meta), "Bernstein")
    b_scores = {k: v for k, v in self_scores.items()
                if truth.get(k) == "Bernstein" and k in b_years}
    b_auc = BR.era_auc(b_scores, b_years, cutoff=args.cutoff)

    res = {"third": args.third, "joints": args.joints, "use": args.use,
           "only": args.only,
           "third_score_std": float(sv.std()), "third_score_iqr":
               [float(np.percentile(sv, 25)), float(np.percentile(sv, 75))],
           "bernstein_internal_era_auc": b_auc, "n_bernstein_with_year": len(b_scores), "n_rec_scored": len(scores),
           "n_rec_with_year": len(common), "spearman_year_vs_pDudamel": rho,
           "era_auc": auc, "cutoff": args.cutoff,
           "p_dudamel_old_median": float(np.median(old)) if old else None,
           "p_dudamel_new_median": float(np.median(new)) if new else None,
           "n_old": len(old), "n_new": len(new),
           "selfcheck_p_dudamel_median": {"Dudamel": float(np.median(dud)),
                                          "Bernstein": float(np.median(ber))},
           "minutes": (time.time() - t0) / 60}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n=== Bridge test result ({args.third}) ===")
    print(f"{len(common)} recordings with a year (old {len(old)} / recent {len(new)})")
    print(f"P(Dudamel) median -- old {res['p_dudamel_old_median']} / "
          f"recent {res['p_dudamel_new_median']}")
    print(f"year-vs-prediction rank correlation rho = {rho:.3f}   era AUC = {auc:.3f}")
    print(f"score spread: std {sv.std():.3f}, IQR {np.percentile(sv, 25):.2f}~"
          f"{np.percentile(sv, 75):.2f}  (no power if all values are identical)")
    print(f"reference (within training set, optimistic bias): era AUC over {len(b_scores)} "
          f"Bernstein recordings = {b_auc:.3f}")
    print("interpretation: AUC near 1.0 means 'the model is reading era', 0.5 means era-independent")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

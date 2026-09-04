"""Final classifier for public release — trains on all accepted runs and saves the weights.

Trains on the full data without an evaluation split (standard deployment practice).
For performance figures, cite the session-LOSO experiment (V5d_both: session 0.851 /
recording 0.860) — not numbers from these weights.
The training data applies the same segment-contamination exclusion (--seg-verdicts)
as the paper.

Usage: PYTHONPATH=src python scripts/train_final_model.py \
        --csv data/experiments/snapshot5_20260812.csv \
        --seg-verdicts data/experiments/segver_20260813.csv \
        --out data/models/resgcn_5class.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pilot as TP
from conductor_classifier import channels as CH
from conductor_classifier import graph as G
from conductor_classifier.resgcn import ResGCN


def main():
    ap = argparse.ArgumentParser()
    # Defaults = the paper's final configuration (2026-08-28). Changing them makes the
    # released weights diverge from the paper.
    ap.add_argument("--csv", default="data/experiments/snapshot6_20260826.csv")
    ap.add_argument("--joints", default="both")
    ap.add_argument("--use", default="pos,vel,acc,bonedir")
    ap.add_argument("--T", type=int, default=75)
    ap.add_argument("--windows-per-run", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--supcon", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/models/resgcn_5class.pt")
    ap.add_argument("--seg-verdicts",
                    default="data/experiments/segver_20260813.csv",
                    help="segment identity verdict CSV — excludes contaminated segments (same condition as the paper)")
    ap.add_argument("--dup-drops",
                    default="data/experiments/audio_same_20260826.csv",
                    help="drop duplicate videos confirmed via audio (same condition as the paper)")
    ap.add_argument("--ood-exclude",
                    default="data/experiments/rehearsal_ood_20260826.csv",
                    help="exclude OOD videos such as rehearsals (same condition as the paper)")
    ap.add_argument("--pad", default="zeromask", choices=["none", "zero", "zeromask"])
    ap.add_argument("--min-real-frac", type=float, default=0.5,
                    help="minimum fraction of real frames per window — salvages short segments via padding")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from conductor_classifier.segver import load_blacklist, load_video_exclude
    from padding_length_probe import drop_duplicates
    seg_bl = load_blacklist(args.seg_verdicts) if args.seg_verdicts else frozenset()
    min_seg = int(np.ceil(args.T * args.min_real_frac)) if args.pad != "none" else args.T
    runs = TP.load_runs(Path(args.csv), Path("."), min_seg, (), seg_blacklist=seg_bl)
    if args.dup_drops:
        runs = drop_duplicates(runs, args.dup_drops)
    if args.ood_exclude:
        ood = load_video_exclude(args.ood_exclude)
        runs = [r for r in runs if (r["conductor"], r["video_id"]) not in ood]
    labels = {c: i for i, c in enumerate(sorted({r["conductor"] for r in runs}))}
    n_vid = len({(r["conductor"], r["video_id"]) for r in runs})
    print(f"Final training: runs {len(runs)} · videos {n_vid} · {labels} · {device}")
    cache = TP.RunCache(runs, depth=False)

    use = tuple(args.use.split(","))
    A = G.adjacency(args.joints)
    C = sum(CH.BRANCH_WIDTH[b] for b in use)
    model = ResGCN(C, len(labels), A).to(device)
    ds = TP.TrainWindows(runs, cache, labels, args.joints, use, args.T,
                         args.windows_per_run, seed=args.seed, pad=args.pad)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, persistent_workers=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl)))
    model.train()
    use_mask = args.pad == "zeromask"
    for ep in range(args.epochs):
        tot = n = 0
        for x, y, m in dl:
            x, y = x.to(device, non_blocking=True), y.to(device)
            kw = {"mask": m.to(device, non_blocking=True)} if use_mask else {}
            logit, emb = model(x, return_embedding=True, **kw)
            loss = Fn.cross_entropy(logit, y) + args.supcon * TP.supcon_loss(emb, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(y); n += len(y)
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1}/{args.epochs} loss {tot/max(1,n):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "labels": labels,                       # class -> index
        "joints": args.joints, "use": args.use, "T": args.T,
        "input_channels": C,
        "norm_mode": "bone_fore",               # assumes fore-normalized skeleton input
        "pad": args.pad, "min_real_frac": args.min_real_frac,
        "trained_on": args.csv,
        "dup_drops": args.dup_drops, "ood_exclude": args.ood_exclude,
        "seg_verdicts": args.seg_verdicts, "supcon": args.supcon,
        "cv_reference": "5-fold stratified group CV, window 0.743±0.012 (P12_nosupcon) — separate experiment",
    }, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

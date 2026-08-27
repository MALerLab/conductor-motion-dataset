"""Final classifier for public release -- trains on all approved runs and saves the weights.

Trains on the full dataset with no evaluation split (standard practice for a
release). For performance numbers, cite the session-LOSO experiment
(V5d_both: session 0.851 / recording 0.860) -- not any number from this
checkpoint. The training data applies the same segment-contamination exclusion
(--seg-verdicts) used in the paper.

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
    ap.add_argument("--csv", default="data/experiments/snapshot5_20260812.csv")
    ap.add_argument("--joints", default="both")
    ap.add_argument("--use", default="pos,vel,acc,bone")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--windows-per-run", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--supcon", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/models/resgcn_5class.pt")
    ap.add_argument("--seg-verdicts", default="",
                    help="Segment-identity verdict CSV -- excludes contaminated segments (same condition as the paper)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from conductor_classifier.segver import load_blacklist
    seg_bl = load_blacklist(args.seg_verdicts) if args.seg_verdicts else frozenset()
    runs = TP.load_runs(Path(args.csv), Path("."), args.T, (), seg_blacklist=seg_bl)
    labels = {c: i for i, c in enumerate(sorted({r["conductor"] for r in runs}))}
    print(f"final training: {len(runs)} runs · {labels} · {device}")
    cache = TP.RunCache(runs, depth=False)

    use = tuple(args.use.split(","))
    A = G.adjacency(args.joints)
    C = sum(CH.BRANCH_WIDTH[b] for b in use)
    model = ResGCN(C, len(labels), A).to(device)
    ds = TP.TrainWindows(runs, cache, labels, args.joints, use, args.T,
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
        "trained_on": args.csv,
        "loso_reference": "session-LOSO 0.886 (V5_both.json) -- a separate experiment from this checkpoint",
    }, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

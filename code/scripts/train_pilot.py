"""Pilot deep training — clean spans of accepted runs -> small ResGCN -> recording-level evaluation.

Design: docs/superpowers/specs/2026-08-09-pilot-deep-training-design.md

Core principles
  · Training windows are not stored on disk; each epoch takes random crops inside clean
    segments (= phase augmentation). Evaluation uses many equally spaced windows +
    **recording-level majority voting**.
  · Splits are **recording-level group K-fold** — window/run-level splits leak.
  · Joint sets (body/hand/both) and channel branches are swappable via arguments to
    run the ablation experiments.

Usage:
    PYTHONPATH=src python scripts/train_pilot.py --joints both --tag B1
    PYTHONPATH=src python scripts/train_pilot.py --joints body --tag A1
    PYTHONPATH=src python scripts/train_pilot.py --joints both --shuffle-test --tag S1
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from conductor_classifier import channels as CH
from conductor_classifier import graph as G
from conductor_classifier import windows as W
from conductor_classifier.resgcn import ResGCN
from conductor_classifier.pose import BONES, ROOT
from conductor_classifier.segver import load_blacklist, load_video_exclude
from conductor_classifier.sessions import (apply_merge_overrides, group_sessions,
                                           load_duplicate_clusters, load_merge_overrides)
from conductor_classifier.temporal import lowpass_temporal

DST_FPS = 25.0


def load_runs(csv_path: Path, root: Path, min_seg_frames: int,
              only: tuple = (),
              seg_blacklist: frozenset = frozenset()) -> list[dict]:
    """List of accepted runs — normalized-skeleton path, clean segments (rescaled to 25fps), label.

    If only is given, use just those conductors (to fix the class composition).
    seg_blacklist is the (conductor, run, seg_idx) set from segver.load_blacklist() —
    segments judged identity-contaminated are excluded from window extraction
    (indices per qc.json).
    **Caution**: the review UI keeps updating the CSV, so cross-condition comparison
    experiments must be pinned to a snapshot CSV (--csv) — otherwise each condition
    sees different data. The same goes for seg_blacklist: use a copy taken at snapshot time.
    """
    out = []
    n_seg_dropped = 0
    for r in csv.DictReader(csv_path.open()):
        if not (r["verdict"] == "accept" or r["source_verdict"] == "accept"):
            continue
        if only and r["conductor"] not in only:
            continue
        d = Path(r["run_dir"])
        npy, qcf = d / "skeleton_norm.npy", d / "qc.json"
        if not (npy.exists() and qcf.exists()):
            continue
        qc = json.loads(qcf.read_text())
        native = [tuple(s) for i, s in enumerate(qc["segments"])
                  if (r["conductor"], r["run"], i) not in seg_blacklist]
        n_seg_dropped += len(qc["segments"]) - len(native)
        segs = W.rescale_segments(native, qc["fps"], DST_FPS)
        segs = [(a, b) for a, b in segs if b - a >= min_seg_frames]
        if not segs:
            continue
        out.append({"conductor": r["conductor"], "video_id": r["video_id"],
                    "run": r["run"], "npy": npy, "fps": qc["fps"], "segments": segs})
    if seg_blacklist:
        print(f"segver: {n_seg_dropped} contaminated segments excluded")
    return out


def _video_quality(runs: list[dict],
                   video_ids: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """(conductor, video_id) -> mean joint confidence (3rd channel of skeleton.npy).

    Used as a video-quality proxy because there is no resolution metadata (source
    videos were not retained) — same definition as rec_conf() in
    verify_hand_conf_era.py (raw conf, independent of normalization).
    Used only to pick the representative (highest-quality) video within a
    true-duplicate cluster."""
    s: dict = defaultdict(float)
    n: dict = defaultdict(int)
    for r in runs:
        k = (r["conductor"], r["video_id"])
        if k not in video_ids:
            continue
        raw = np.load(Path(r["npy"]).parent / "skeleton.npy")
        s[k] += float(raw[:, :, 2].sum()); n[k] += raw[:, :, 2].size
    return {k: s[k] / n[k] for k in s if n[k]}


def _fore_ratio(run_dir: Path, native_fps: float, tgt_len: int,
                min_conf: float = 0.3, anchor: str = "median") -> np.ndarray:
    """(tgt_len,133) foreshortening ratio r — raw 2D bone length / run median.

    Taylor (2000, CVIU 80): under scaled orthographic projection, on-screen length
    l = s*L*cos(theta). Dividing by the per-frame shoulder width cancels zoom (s), and
    dividing by the run median cancels body proportions (L), leaving only the relative
    magnitude of r ~ cos(theta) = how much the bone tilts out of the image plane
    (toward the camera). This single channel re-injects the depth-direction motion
    that normalization erased.
    The sign (forward/backward) is inherently ambiguous from a single viewpoint
    (same paper), so it is not provided.
    Invalid frames (low confidence / shoulders undetected) get the neutral 1.
    Clipped to [0.2, 2.0].
    """
    raw = np.load(run_dir / "skeleton.npy")
    T = len(raw)
    ls, rs = raw[:, 5, :2], raw[:, 6, :2]
    sw = np.linalg.norm(ls - rs, axis=1)
    sc = (ls + rs) / 2.0
    sw_ok = (sw > 1e-6) & (raw[:, 5, 2] >= min_conf) & (raw[:, 6, 2] >= min_conf)
    ratio = np.ones((T, 133), dtype=np.float32)
    for pj, cj in BONES:
        pxy = sc if pj == ROOT else raw[:, pj, :2]
        l = np.linalg.norm(raw[:, cj, :2] - pxy, axis=1)
        ok = sw_ok & (raw[:, cj, 2] >= min_conf)
        if pj != ROOT:
            ok &= raw[:, pj, 2] >= min_conf
        rel = np.where(ok, l / np.maximum(sw, 1e-6), np.nan)
        # anchor: median = only the relative change vs. the usual pose / p95 = the
        # 'most extended-looking moment' ~ actual length (Taylor: at the image-plane-
        # parallel moment, l -> s*L) -> r approaches absolute cos(theta), preserving
        # even the habitual tilt (static style). The trade-off is that camera angle
        # gets mixed in.
        med = (np.nanpercentile(rel, 95) if anchor == "p95"
               else np.nanmedian(rel))
        if not np.isfinite(med) or med < 1e-6:
            continue
        r = rel / med
        ratio[:, cj] = np.clip(np.where(np.isfinite(r), r, 1.0), 0.2, 2.0)
    if T == tgt_len:
        return ratio
    xo = np.arange(T) / native_fps
    xn = np.arange(tgt_len) / DST_FPS
    out = np.empty((tgt_len, 133), dtype=np.float32)
    for j in range(133):
        out[:, j] = np.interp(xn, xo, ratio[:, j])
    return out


class RunCache:
    """Keeps the normalized skeletons in memory (tens of minutes of footage in total).

    skeleton_norm.npy is **already saved at 25fps** by the normalize step — resampling
    here again would double-compress the time axis and misalign the segment
    coordinates (a bug that actually happened: re-resampling based on the qc fps
    (native) -> the last 17% of windows of a 29.97fps run were mirror-reflected
    garbage, plus all windows played 20% fast).

    With depth=True, the foreshortening ratio is computed from raw coordinates and
    appended as a 4th column. lowpass applies additional low-pass filtering
    (for ablation experiments).
    """

    def __init__(self, runs: list[dict], lowpass: float = 0.0,
                 depth: bool = True, depth_anchor: str = "median"):
        self.seq = {}
        for r in runs:
            a = np.load(r["npy"]).astype(np.float32)   # already 25fps
            if lowpass:
                a = lowpass_temporal(a, DST_FPS, lowpass)
            if depth:
                fore = _fore_ratio(Path(r["npy"]).parent, r["fps"], len(a),
                                   anchor=depth_anchor)
                a = np.concatenate([a, fore[..., None]], axis=2)
            self.seq[r["run"]] = a.astype(np.float32)

    def __getitem__(self, run: str) -> np.ndarray:
        return self.seq[run]


def _neutral_channel_values(joints: str, use: tuple) -> np.ndarray:
    """Per-channel 'no observation' neutral values (C,). Mostly 0, but foreshortening r is 1.

    r = on-screen bone length / usual length, so **0 is the extreme value 'the length
    collapsed to zero'** and the neutral is 1 (unchanged from usual). Invalid-frame
    handling (_fore_ratio) also uses 1.
    Filling padding entirely with 0 creates an artificial pattern where r plunges
    1->0 at the real/padding boundary, and mask pooling cannot prevent it
    (convolutions propagate it).
    """
    if joints == "pair":
        return np.zeros(11, dtype=np.float32)
    vals: list[float] = []
    for b in CH.ALL_BRANCHES:
        if b not in use:
            continue
        if b == "bone":                      # vecx, vecy, r, delta r
            vals += [0.0, 0.0, 1.0, 0.0]
        else:
            vals += [0.0] * CH.BRANCH_WIDTH[b]
    return np.asarray(vals, dtype=np.float32)


def _channels_padded(w: np.ndarray, joints: str, use: tuple, drop_conf: bool,
                     T: int, pad_value: str = "zero") -> tuple[np.ndarray, np.ndarray]:
    """Window frames w (L<=T) -> (channels (C,T,V), mask (T,)).

    **Pad after computing channels**: the velocity/acceleration differences are
    finished within the real frames only, then the missing tail is zero-filled.
    Reversing the order (zero-fill first, then differentiate) bakes 'fake spike
    accelerations' several times larger than real motion into the channels at the
    real/padding boundary.
    Mask is 1=observed / 0=padded, used to correct the pooling denominator.
    """
    if joints == "pair":
        x = CH.build_pair_channels(w)
    else:
        x = CH.build_channels(w, joints, use, drop_conf)
    L = x.shape[1]
    m = np.zeros(T, dtype=np.float32)
    m[:L] = 1.0
    if L < T:
        x = np.pad(x, ((0, 0), (0, T - L), (0, 0)))
        if pad_value == "neutral":
            nv = _neutral_channel_values(joints, use)
            x[:, L:, :] = nv[:, None, None]
    return np.ascontiguousarray(x), m


class TrainWindows(Dataset):
    """n_per_run random windows from each run per epoch."""

    def __init__(self, runs, cache, labels, joints, use, T, n_per_run, seed=0,
                 noise=0.02, frame_drop=0.1, drop_conf=False,
                 local_hands=False, pad="none", pad_value="zero"):
        self.runs, self.cache, self.labels = runs, cache, labels
        self.joints, self.use, self.T = joints, use, T
        self.drop_conf = drop_conf
        self.local_hands = local_hands
        self.pad = pad
        self.pad_value = pad_value
        self.n = n_per_run
        self.rng = np.random.default_rng(seed)
        self.noise, self.frame_drop = noise, frame_drop

    def __len__(self):
        return len(self.runs) * self.n

    def __getitem__(self, i):
        r = self.runs[i // self.n]
        seq = self.cache[r["run"]]
        if self.pad == "none":
            s = W.sample_start(r["segments"], self.T, self.rng)
            L = self.T
        else:
            s, L = W.sample_span(r["segments"], self.T, self.rng)
        w = seq[s:s + L].copy() if self.pad != "none" \
            else W.crop(seq, s, self.T).copy()
        if self.local_hands:
            w = CH.localize_hands(w)
        if self.noise:
            w[:, :, :2] += self.rng.normal(0, self.noise, w[:, :, :2].shape)
        if self.frame_drop and len(w) > 1 and self.rng.random() < 0.5:
            k = self.rng.integers(0, max(1, int(len(w) * self.frame_drop)))
            for _ in range(k):   # frame drop = replace with the previous frame
                j = int(self.rng.integers(1, len(w)))
                w[j] = w[j - 1]
        x, m = _channels_padded(w, self.joints, self.use, self.drop_conf,
                                self.T, self.pad_value)
        return torch.from_numpy(x), self.labels[r["conductor"]], torch.from_numpy(m)


class EvalWindows(Dataset):
    """Equally spaced windows — also returns run/recording identifiers for the vote aggregation."""

    def __init__(self, runs, cache, labels, joints, use, T, n_eval,
                 shuffle_frames=False, seed=0, drop_conf=False,
                 local_hands=False, min_stride_frac=0.5, max_per_rec=40,
                 eval_margin=0, pad="none", pad_value="zero"):
        # Cap on windows per recording — keeps long recordings from dominating
        # window-level metrics.
        by_rec: dict = {}
        for r in runs:
            by_rec.setdefault((r["conductor"], r["video_id"]), []).append(r)
        self.items = []
        for _k, rs in by_rec.items():
            cand = []
            for r in rs:
                if pad == "none":
                    spans = [(s, T) for s in
                             W.window_starts(r["segments"], T, n_eval,
                                             min_stride_frac=min_stride_frac,
                                             margin=eval_margin)]
                else:
                    spans = W.window_spans(r["segments"], T, n_eval,
                                           min_stride_frac=min_stride_frac,
                                           margin=eval_margin)
                for s, L in spans:
                    cand.append((r, s, L))
            if max_per_rec and len(cand) > max_per_rec:
                idx = np.linspace(0, len(cand) - 1, max_per_rec)
                cand = [cand[int(round(i))] for i in idx]
            self.items += cand
        self.cache, self.labels = cache, labels
        self.joints, self.use, self.T = joints, use, T
        self.shuffle_frames = shuffle_frames
        self.drop_conf = drop_conf
        self.local_hands = local_hands
        self.pad = pad
        self.pad_value = pad_value
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        r, s, L = self.items[i]
        seq = self.cache[r["run"]]
        w = seq[s:s + L] if self.pad != "none" else W.crop(seq, s, self.T)
        if self.local_hands:
            w = CH.localize_hands(w)
        if self.shuffle_frames:      # S1 control: if it still succeeds with order destroyed, it reads static cues
            w = w[self.rng.permutation(len(w))]
        x, m = _channels_padded(w, self.joints, self.use, self.drop_conf,
                                self.T, self.pad_value)
        # Untrained classes (the third conductor in the bridging test) have no label -> -1
        y = self.labels.get(r["conductor"], -1)
        return (torch.from_numpy(x), y, r["video_id"], r["conductor"], r["run"],
                torch.from_numpy(m), int(L < self.T))


class STGCNPPScratch(torch.nn.Module):
    """For architecture comparison: the ST-GCN++ backbone trained from scratch, without
    pretraining, on **our input (normalized 51 joints, 13 channels)**. The graph uses the
    bone connectivity of our joint set (graph.edges), input (B,C,T,V) -> (N,M=1,T,V,C).
    Same (logit, emb) interface as ResGCN."""

    def __init__(self, in_channels: int, num_classes: int, joints: str, coord: bool):
        super().__init__()
        from conductor_classifier.stgcnpp import STGCNPP
        V = len(CH.JOINT_SETS[joints])
        # edges are (min,max) undirected — build inward as (child, parent) from parents
        par = CH.parents(joints)
        inward = [(c, int(p)) for c, p in enumerate(par) if c != p]
        for a, b in G.edges(joints, coord):           # add off-tree edges (symmetry correction / coordination)
            if (a, b) not in inward and (b, a) not in inward:
                inward.append((a, b))
        self.net = STGCNPP(num_classes=num_classes, in_channels=in_channels,
                           graph_cfg=dict(layout="custom", mode="spatial",
                                          num_node=V, inward=inward))

    def forward(self, x, return_embedding: bool = False):   # x: (B,C,T,V)
        x = x.permute(0, 2, 3, 1).unsqueeze(1)             # (B,1,T,V,C)
        emb = self.net.forward_features(x)                  # (B,256) after pooling
        logit = self.net.cls_head.fc_cls(emb)
        return (logit, emb) if return_embedding else logit


def supcon_loss(emb, y, tau=0.1):
    """Contrastive loss pulling same-class samples together and pushing other classes apart within a batch."""
    z = Fn.normalize(emb, dim=1)
    sim = z @ z.t() / tau
    n = len(y)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = (y[:, None] == y[None, :]) & ~eye
    if not pos.any():
        return torch.zeros((), device=z.device)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    cnt = pos.sum(1)
    ok = cnt > 0
    return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()


def run_fold(tr_runs, te_runs, cache, labels, args, device):
    joints, use = args.joints, tuple(args.use.split(","))
    if joints == "pair":
        A, C = G.pair_adjacency(), 11
    else:
        A = G.adjacency(joints, coord=args.coord)
        C = sum(CH.BRANCH_WIDTH[b] for b in use)
    if args.arch == "stgcnpp":
        model = STGCNPPScratch(C, len(labels), joints, args.coord).to(device)
    else:
        model = ResGCN(C, len(labels), A).to(device)
    # The mask is passed to the model only with pad=zeromask — "zero" is the
    # "let the model figure it out" condition (the professor's original proposal),
    # feeding raw zeros with no denominator correction.
    use_mask = args.pad == "zeromask"
    ds = TrainWindows(tr_runs, cache, labels, joints, use, args.T,
                      args.windows_per_run, seed=args.seed,
                      drop_conf=args.drop_conf, local_hands=args.local_hands,
                      pad=args.pad, pad_value=args.pad_value)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, persistent_workers=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl)))
    model.train()
    for _ in range(args.epochs):
        for x, y, m in dl:
            x, y = x.to(device, non_blocking=True), y.to(device)
            kw = {"mask": m.to(device, non_blocking=True)} if use_mask else {}
            logit, emb = model(x, return_embedding=True, **kw)
            loss = Fn.cross_entropy(logit, y) + args.supcon * supcon_loss(emb, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

    model.eval()
    ev = EvalWindows(te_runs, cache, labels, joints, use, args.T, args.n_eval,
                     shuffle_frames=args.shuffle_test, seed=args.seed,
                     drop_conf=args.drop_conf, local_hands=args.local_hands,
                     eval_margin=args.eval_margin, pad=args.pad,
                     pad_value=args.pad_value)
    dl_e = DataLoader(ev, batch_size=args.batch, num_workers=4)
    per_rec = defaultdict(lambda: [0.0, None])   # video_id -> [logit sum, ground truth]
    win_by_rec = defaultdict(list)               # per-recording window hits (for macro averaging)
    win_ok = win_n = 0
    win_cm = np.zeros((len(labels), len(labels)), dtype=int)  # window-level confusion
    per_run = defaultdict(lambda: [np.zeros(len(labels)), -1])  # run -> [probability sum, ground truth]
    # Separate tallies for padded / unpadded windows — checks "is the salvaged data
    # actually classified correctly" without mixing it into the overall accuracy
    # (the pre-registered reading rule).
    pad_ok = pad_n = un_ok = un_n = 0
    pad_cls_ok: dict = defaultdict(int)
    pad_cls_n: dict = defaultdict(int)
    with torch.no_grad():
        for x, y, vid, cond, runid, m, is_pad in dl_e:
            kw = {"mask": m.to(device)} if use_mask else {}
            p = Fn.softmax(model(x.to(device), **kw), dim=1).cpu().numpy()
            y = y.numpy()
            ip = is_pad.numpy()
            win_ok += int((p.argmax(1) == y).sum()); win_n += len(y)
            hit = p.argmax(1) == y
            pad_ok += int(hit[ip == 1].sum()); pad_n += int((ip == 1).sum())
            un_ok += int(hit[ip == 0].sum()); un_n += int((ip == 0).sum())
            for k in range(len(y)):
                if ip[k] == 1 and y[k] >= 0:
                    pad_cls_n[int(y[k])] += 1
                    pad_cls_ok[int(y[k])] += int(hit[k])
            for yt, yp in zip(y, p.argmax(1)):
                if yt >= 0:                      # exclude untrained classes (-1)
                    win_cm[yt, yp] += 1
            for k in range(len(y)):
                win_by_rec[(cond[k], vid[k])].append(bool(p[k].argmax() == y[k]))
                cur = per_rec[(cond[k], vid[k])]
                cur[0] = cur[0] + p[k] if cur[1] is not None else p[k]
                cur[1] = y[k]
                pr = per_run[runid[k]]           # shot (run) level — probability summation
                pr[0] = pr[0] + p[k]
                pr[1] = y[k]
    rec = [(np.argmax(v[0]) == v[1]) for v in per_rec.values()]
    run_ok = [(int(np.argmax(v[0])) == v[1]) for v in per_run.values() if v[1] >= 0]
    preds = {f"{c}|{v}": {"p": (np.asarray(val[0]) / np.sum(val[0])).tolist(),
                          "y": int(val[1])}
             for (c, v), val in per_rec.items()}
    macro = float(np.mean([np.mean(v) for v in win_by_rec.values()])) \
        if win_by_rec else 0.0
    return {"rec_correct": int(np.sum(rec)), "rec_n": len(rec),
            "run_correct": int(np.sum(run_ok)), "run_n": len(run_ok),
            "win_acc": win_ok / max(1, win_n), "win_macro": macro,
            "win_n": win_n, "win_cm": win_cm, "preds": preds,
            "pad_ok": pad_ok, "pad_n": pad_n, "unpad_ok": un_ok, "unpad_n": un_n,
            "pad_cls_ok": dict(pad_cls_ok), "pad_cls_n": dict(pad_cls_n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/review/run_verdicts.csv")
    ap.add_argument("--root", default="skeletons_A")
    ap.add_argument("--joints", default="both",
                    choices=list(CH.JOINT_SETS) + ["pair"])
    ap.add_argument("--coord", action="store_true",
                    help="add left-right coordination edges (both wrists, elbows, hands)")
    ap.add_argument("--only", nargs="*", default=[],
                    help="restrict to these conductors (e.g. --only Bernstein Dudamel)")
    ap.add_argument("--use", default="pos,vel,acc,bone")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--windows-per-run", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--eval-margin", type=int, default=0,
                    help="pull evaluation windows this far (frames, 25fps) back from "
                         "both segment ends — boundary-artifact sensitivity experiment")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--supcon", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-test", action="store_true")
    ap.add_argument("--local-hands", action="store_true",
                    help="subtract the wrist from hand coordinates to remove the arm trajectory — finger shape only")
    ap.add_argument("--fps-only", type=float, default=0.0,
                    help="use only runs whose native fps equals this value — blocks resampling confounds")
    ap.add_argument("--depth-anchor", default="median", choices=["median", "p95"])
    ap.add_argument("--no-depth", action="store_true",
                    help="turn off the foreshortening (depth) channel — depth-contribution ablation")
    ap.add_argument("--lowpass", type=float, default=0.0,
                    help="additional low-pass filtering (Hz) — control removing fine temporal texture")
    ap.add_argument("--drop-conf", action="store_true",
                    help="remove the confidence channel — control blocking the video-quality proxy")
    ap.add_argument("--arch", default="resgcn", choices=["resgcn", "stgcnpp"],
                    help="classifier architecture — stgcnpp is trained from scratch on our input, no pretraining")
    ap.add_argument("--pad", default="none", choices=["none", "zero", "zeromask"],
                    help="salvage segments shorter than the window: none=discard (previous behavior) · "
                         "zero=zero-fill the tail · zeromask=zero-fill + exclude from the pooling denominator")
    ap.add_argument("--pad-value", default="zero", choices=["zero", "neutral"],
                    help="fill value for padded frames: zero=all channels 0 (previous behavior) · "
                         "neutral=per-channel neutral values (foreshortening r is 1)")
    ap.add_argument("--min-real-frac", type=float, default=0.5,
                    help="segment length floor when --pad is used (fraction of real frames per window). "
                         "0.5 = at least half the window must be observed frames")
    ap.add_argument("--tag", default="B1")
    ap.add_argument("--out", default="data/experiments")
    ap.add_argument("--seg-verdicts", default="",
                    help="segment identity verdict CSV — excludes contaminated segments "
                         "(snapshot copy recommended, e.g. data/experiments/segver_YYYYMMDD.csv)")
    ap.add_argument("--ood-exclude", default="",
                    help="training-exclusion video CSV (conductor,video_id) — a second "
                         "safeguard protecting the OOD held-out set "
                         "(e.g. data/experiments/rehearsal_ood_YYYYMMDD.csv)")
    ap.add_argument("--session-merges", default="",
                    help="CSV of manual session merges confirmed by audio comparison "
                         "(e.g. data/experiments/session_merges_YYYYMMDD.csv)")
    ap.add_argument("--no-session-grouping", action="store_true",
                    help="disable same-performance grouping and treat each video as its own group — "
                         "control measuring the effect of the leakage correction (paper §3.2)")
    ap.add_argument("--dup-drops", default="",
                    help="audio cross-correlation verdict=same pair CSV — keeps only one video "
                         "(highest quality) per cluster of true duplicates (re-uploads / "
                         "excerpts) and drops the rest "
                         "(e.g. data/experiments/audio_sweep_20260814.csv)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.pad == "zeromask" and args.arch != "resgcn":
        ap.error("--pad zeromask is implemented only for resgcn")
    seg_bl = load_blacklist(args.seg_verdicts) if args.seg_verdicts else frozenset()
    # With padding, the segment length floor drops from T to T*min_real_frac,
    # so short spans that used to be discarded become window candidates.
    min_seg = args.T if args.pad == "none" \
        else int(np.ceil(args.T * args.min_real_frac))
    runs = load_runs(Path(args.csv), Path(args.root), min_seg, tuple(args.only),
                     seg_blacklist=seg_bl)
    # OOD (rehearsal etc.) videos are held-out — even if they remain in the snapshot,
    # they are blocked here.
    ood_excl = load_video_exclude(args.ood_exclude) if args.ood_exclude else frozenset()
    if ood_excl:
        before = len(runs)
        runs = [r for r in runs
                if (r["conductor"], r["video_id"]) not in ood_excl]
        print(f"OOD exclusion ({args.ood_exclude}): {len(ood_excl)} videos listed -> "
              f"runs {before}->{len(runs)}")
    if args.fps_only:
        runs = [r for r in runs if abs(r["fps"] - args.fps_only) < 0.05]

    dup_dropped: list = []
    if args.dup_drops:
        dup_clusters = load_duplicate_clusters(args.dup_drops)
        cluster_vids = {(c, v) for c, gs in dup_clusters.items()
                        for g in gs for v in g}
        quality = _video_quality(runs, cluster_vids)
        # Representative selection: total clean footage first, then quality (conf) as
        # tie-breaker — for excerpt vs. full-piece duplicates, using conf alone lets
        # the short excerpt win and loses unique footage.
        footage: dict = defaultdict(int)
        for r in runs:
            k = (r["conductor"], r["video_id"])
            if k in cluster_vids:
                footage[k] += sum(b - a for a, b in r["segments"])
        drop = set()
        for c, groups in dup_clusters.items():
            for g in groups:
                present = [(c, v) for v in g if (c, v) in quality]
                if len(present) < 2:
                    continue
                best = max(present, key=lambda k: (footage[k], quality[k]))
                drop |= {k for k in present if k != best}
        if drop:
            before = len(runs)
            runs = [r for r in runs if (r["conductor"], r["video_id"]) not in drop]
            dup_dropped = sorted(f"{c}:{v}" for c, v in drop)
            print(f"True-duplicate drop ({args.dup_drops}): {len(drop)} videos excluded "
                  f"(runs {before}->{len(runs)})")

    labels = {c: i for i, c in enumerate(sorted({r["conductor"] for r in runs}))}
    recs = sorted({(r["conductor"], r["video_id"]) for r in runs})
    print(f"[{args.tag}] runs {len(runs)} · recordings {len(recs)} · classes {labels} · {device}")

    cache = RunCache(runs, lowpass=args.lowpass, depth=not args.no_depth,
                     depth_anchor=args.depth_anchor)
    # Folds are **session-level** — if per-movement uploads of the same performance
    # split across train/test, the stage, attire, and camera leak the answer
    # (the hole in video_id-level splitting).
    titles = {}
    for m in csv.DictReader(Path("data/recordings_meta.csv").open()):
        titles.setdefault(m["conductor"], {})[m["video_id"].strip()] = m["title"]
    merge_ov = load_merge_overrides(args.session_merges) if args.session_merges else {}
    sess_of = {}
    for c in sorted({r["conductor"] for r in runs}):
        vids = {r["video_id"] for r in runs if r["conductor"] == c}
        tmap = {v: titles.get(c, {}).get(v, "") for v in vids}
        if args.no_session_grouping:
            g = {v: v for v in vids}         # one video = one group (no correction)
        else:
            g = group_sessions(tmap)
            if c in merge_ov:
                g = apply_merge_overrides(g, merge_ov[c])
        for v, sid in g.items():
            sess_of[(c, v)] = (c, sid)
    sessions = sorted({sess_of[k] for k in recs})
    print(f"Session grouping: {len(recs)} recordings -> {len(sessions)} sessions")
    # Keep the group (session) level split but equalize conductor ratios across folds —
    # with pure randomness, classes with few groups (e.g. Bernstein, 16 groups / 5 folds)
    # were empirically observed to land 0 groups in some fold (2026-08-25, no stratification).
    y = np.array([labels[c] for c, _ in recs])
    groups = np.array([str(sess_of[k]) for k in recs])
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                random_state=args.seed)
    fold_of = {}
    for f, (_, te_idx) in enumerate(sgkf.split(np.arange(len(recs)), y, groups)):
        for i in te_idx:
            fold_of[recs[i]] = f

    t0 = time.time()
    all_preds: dict = {}     # per-recording predictions — for cross-condition error correlation analysis
    tot_c = tot_n = 0
    win_accs, win_macros = [], []
    win_tot = 0
    run_c = run_n = 0                            # shot (run) level tally
    win_cm_tot = 0                               # window-level confusion tally
    pad_c = pad_t = unpad_c = unpad_t = 0        # separate padded/unpadded window tallies
    pad_cls_c: dict = defaultdict(int)
    pad_cls_t: dict = defaultdict(int)
    for f in range(args.folds):
        tr = [r for r in runs if fold_of[(r["conductor"], r["video_id"])] != f]
        te = [r for r in runs if fold_of[(r["conductor"], r["video_id"])] == f]
        if not te:
            continue
        res = run_fold(tr, te, cache, labels, args, device)
        all_preds.update(res.pop("preds"))
        tot_c += res["rec_correct"]; tot_n += res["rec_n"]
        run_c += res["run_correct"]; run_n += res["run_n"]
        win_accs.append(res["win_acc"]); win_macros.append(res["win_macro"])
        win_tot += res["win_n"]
        win_cm_tot = win_cm_tot + res["win_cm"]
        pad_c += res["pad_ok"]; pad_t += res["pad_n"]
        unpad_c += res["unpad_ok"]; unpad_t += res["unpad_n"]
        for k, v in res["pad_cls_n"].items():
            pad_cls_t[k] += v; pad_cls_c[k] += res["pad_cls_ok"].get(k, 0)
        print(f"  fold {f}: recordings {res['rec_correct']}/{res['rec_n']} "
              f"· window {res['win_acc']:.3f} / macro {res['win_macro']:.3f} "
              f"({res['win_n']})")

    # chance = majority-class fraction at the recording level
    cnt = defaultdict(int)
    for c, _ in recs:
        cnt[c] += 1
    chance = max(cnt.values()) / len(recs)
    inv_lab = {v: k for k, v in labels.items()}
    acc = tot_c / max(1, tot_n)
    z = (acc - chance) / np.sqrt(max(1e-9, chance * (1 - chance) / max(1, tot_n)))
    out = {"tag": args.tag, "joints": args.joints, "use": args.use,
           "rec_acc": acc, "rec_n": tot_n, "chance": chance, "z": float(z),
           "win_acc": float(np.mean(win_accs)) if win_accs else None,
           "win_macro": float(np.mean(win_macros)) if win_macros else None,
           "win_n": win_tot,
           "run_acc": run_c / max(1, run_n), "run_n": run_n,
           "win_cm": win_cm_tot.tolist() if not isinstance(win_cm_tot, int) else None,
           "shuffle_test": args.shuffle_test, "drop_conf": args.drop_conf,
           "local_hands": args.local_hands, "fps_only": args.fps_only,
           "lowpass": args.lowpass, "depth": not args.no_depth, "depth_anchor": args.depth_anchor, "epochs": args.epochs,
           "T": args.T, "folds": args.folds, "only": args.only, "csv": args.csv, "coord": args.coord,
           "seg_verdicts": args.seg_verdicts, "session_merges": args.session_merges,
           "no_session_grouping": args.no_session_grouping,
           "ood_exclude": args.ood_exclude, "n_ood_excluded": len(ood_excl),
           "dup_drops": args.dup_drops, "dup_dropped": dup_dropped,
           "arch": args.arch, "eval_margin": args.eval_margin,
           "pad": args.pad, "pad_value": args.pad_value, "min_real_frac": args.min_real_frac,
           "min_seg_frames": min_seg,
           "pad_acc": pad_c / pad_t if pad_t else None, "pad_n": pad_t,
           "unpad_acc": unpad_c / unpad_t if unpad_t else None, "unpad_n": unpad_t,
           "pad_cls_recall": {inv_lab[k]: [pad_cls_c[k], pad_cls_t[k]]
                              for k in sorted(pad_cls_t)},
           # window-level chance = recomputed from the actual window counts, since the
           # window composition differs per condition
           "win_chance": (float(np.max(np.sum(win_cm_tot, axis=1))
                                / max(1, np.sum(win_cm_tot)))
                          if not isinstance(win_cm_tot, int) else None),
           "n_sessions": len(sessions), "minutes": (time.time() - t0) / 60,
           "preds": all_preds}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n[{args.tag}] recordings {acc:.3f} (chance {chance:.3f}, z={z:.1f}) · "
          f"window {out['win_acc']:.3f} / macro {out['win_macro']:.3f} (n={win_tot}) · "
          f"{out['minutes']:.1f} min -> {args.out}/{args.tag}.json")
    if args.pad != "none":
        print(f"  [{args.pad} T={args.T}] padded windows {pad_c}/{pad_t} = "
              f"{out['pad_acc']:.3f} · unpadded windows {unpad_c}/{unpad_t} = "
              f"{out['unpad_acc']:.3f} · window chance {out['win_chance']:.3f}")
        for k in sorted(pad_cls_t):
            print(f"    padded windows {inv_lab[k]:<10s} {pad_cls_c[k]}/{pad_cls_t[k]} = "
                  f"{pad_cls_c[k]/max(1,pad_cls_t[k]):.3f}")


if __name__ == "__main__":
    main()

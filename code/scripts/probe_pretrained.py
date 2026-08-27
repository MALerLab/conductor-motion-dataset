"""Pretrained-backbone probe — MotionBERT-Lite / ST-GCN++ (NTU60) vs. our own ResGCN.

An empirical answer to the advisor's question ("can an existing model be
used for this analysis"). Both backbones use **exactly the same
session-LOSO split and window-aggregation evaluation as ours** — the only
thing that differs is the representation (backbone), so the numbers are
directly a "pretrained representation vs. training from scratch" comparison.

  - P1  MotionBERT-Lite (arXiv:2210.06551), frozen + linear probe
        Input: COCO-17 -> H36M-17 conversion, crop_scale to [-1,1] (original repo's approach)
  - P2  ST-GCN++ (pyskl, NTU60-XSub HRNet 2D), frozen probe / full fine-tune
        Input: COCO-17 as-is (no remapping), frame-centered [-1,1] normalization

The skeleton used is the **first 17 points of the raw 2D data (skeleton.npy)**,
not the normalized version — points 0..16 of RTMPose WholeBody's 133 are
exactly COCO-17, and each backbone must receive the same preprocessing as its
own pretraining for a fair comparison. Only the time axis matches ours, at 25fps.

Usage:
    PYTHONPATH=src python scripts/probe_pretrained.py --model mb --tag P1_mb_probe
    PYTHONPATH=src python scripts/probe_pretrained.py --model stgcnpp --tag P2_stgcn_probe
    PYTHONPATH=src python scripts/probe_pretrained.py --model stgcnpp --finetune --tag P2_stgcn_ft
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import train_pilot as TP                      # reuse load_runs / window utilities
from conductor_classifier import windows as W
from conductor_classifier.sessions import (apply_merge_overrides, group_sessions,
                                           load_merge_overrides)

DST_FPS = 25.0


# ---------------- Raw COCO-17 loader ----------------

def load_raw17(run: dict) -> np.ndarray:
    """Resamples the first 17 points (x,y,conf) of the run's raw skeleton to 25fps and returns them.

    The length uses the same formula as skeleton_norm (round(T*25/fps)) so the
    frame axis lines up with the qc segments (which load_runs has already
    rescaled to 25fps).
    """
    raw = np.load(Path(run["npy"]).parent / "skeleton.npy").astype(np.float32)
    a = raw[:, :17, :3]
    T = len(a)
    fps = run["fps"]
    if abs(fps - DST_FPS) < 0.05:
        return a
    L = int(round(T * DST_FPS / fps))
    xo = np.arange(T) / fps
    xn = np.minimum(np.arange(L) / DST_FPS, xo[-1])
    out = np.empty((L, 17, 3), dtype=np.float32)
    for j in range(17):
        for c in range(3):
            out[:, j, c] = np.interp(xn, xo, a[:, j, c])
    return out


# ---------------- Per-model preprocessing ----------------

COCO2H36M = None  # see the function below


def coco2h36m(x: np.ndarray) -> np.ndarray:
    """(T,17,3) COCO -> H36M 17 joints (MotionBERT convention). Synthesized joints take conf = min of their parents."""
    y = np.zeros_like(x)
    mn = np.minimum
    y[:, 0] = (x[:, 11] + x[:, 12]) * 0.5           # pelvis = mid hip
    y[:, 0, 2] = mn(x[:, 11, 2], x[:, 12, 2])
    y[:, 1] = x[:, 12]                              # R hip
    y[:, 2] = x[:, 14]                              # R knee
    y[:, 3] = x[:, 16]                              # R ankle
    y[:, 4] = x[:, 11]                              # L hip
    y[:, 5] = x[:, 13]                              # L knee
    y[:, 6] = x[:, 15]                              # L ankle
    thorax = (x[:, 5] + x[:, 6]) * 0.5              # mid shoulder
    y[:, 8] = thorax
    y[:, 8, 2] = mn(x[:, 5, 2], x[:, 6, 2])
    y[:, 7] = (y[:, 0] + y[:, 8]) * 0.5             # spine
    y[:, 7, 2] = mn(y[:, 0, 2], y[:, 8, 2])
    y[:, 9] = x[:, 0]                               # nose
    y[:, 10] = (x[:, 3] + x[:, 4]) * 0.5            # head = mid ear
    y[:, 10, 2] = mn(x[:, 3, 2], x[:, 4, 2])
    y[:, 11] = x[:, 5]                              # L shoulder
    y[:, 12] = x[:, 7]
    y[:, 13] = x[:, 9]
    y[:, 14] = x[:, 6]                              # R shoulder
    y[:, 15] = x[:, 8]
    y[:, 16] = x[:, 10]
    return y


def crop_scale(motion: np.ndarray) -> np.ndarray:
    """Identical to MotionBERT's lib/utils/utils_data.crop_scale (scale_range=[1,1])."""
    result = motion.copy()
    valid = motion[motion[..., 2] != 0][:, :2]
    if len(valid) < 4:
        return np.zeros_like(motion)
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros_like(motion)
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    return np.clip(result, -1, 1)


_STD_W = [426, 640, 854, 1280, 1440, 1920, 2560, 3840]
_STD_H = [240, 360, 480, 720, 1080, 1440, 2160]


def frame_norm_params(seq: np.ndarray) -> tuple[float, float]:
    """Estimates frame (w,h) across the whole run — rounds up to a standard resolution."""
    v = seq[seq[..., 2] > 0.3]
    if len(v) == 0:
        return 1920.0, 1080.0
    mx, my = float(v[:, 0].max()), float(v[:, 1].max())
    w = next((s for s in _STD_W if s >= mx), mx)
    h = next((s for s in _STD_H if s >= my), my)
    return float(w), float(h)


def prep_window_mb(w: np.ndarray) -> np.ndarray:
    return crop_scale(coco2h36m(w)).astype(np.float32)


def prep_window_stgcn(w: np.ndarray, wh: tuple[float, float]) -> np.ndarray:
    out = w.copy()
    out[..., 0] = (out[..., 0] - wh[0] / 2) / (wh[0] / 2)   # PreNormalize2D
    out[..., 1] = (out[..., 1] - wh[1] / 2) / (wh[1] / 2)
    return out.astype(np.float32)


# ---------------- Backbone loading ----------------

def load_backbone(model_name: str, device: str):
    if model_name == "mb":
        sys.path.insert(0, str(REPO / "pretrained" / "motionbert"))
        from lib.model.DSTformer import DSTformer
        net = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                        depth=5, num_heads=8, mlp_ratio=4,
                        maxlen=243, num_joints=17)
        ck = torch.load(REPO / "pretrained" / "mb_lite.bin",
                        map_location="cpu", weights_only=False)
        sd = ck.get("model_pos", ck.get("model", ck))
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        # All backbone weights must be present — only auxiliary parts like the pos head are allowed to be missing
        core_missing = [k for k in missing if not k.startswith("head")]
        assert not core_missing, f"MotionBERT core weights missing: {core_missing[:5]}"
        dim = 512

        def embed(x):                     # x: (N,T,17,3) [-1,1]
            rep = net.get_representation(x)      # (N,T,17,512)
            return rep.mean(dim=(1, 2))
    elif model_name == "stgcnpp":
        from conductor_classifier.stgcnpp import STGCNPP
        net = STGCNPP(num_classes=60)
        ck = torch.load(REPO / "pretrained" / "stgcnpp_ntu60_xsub_hrnet_j.pth",
                        map_location="cpu", weights_only=False)
        net.load_state_dict(ck.get("state_dict", ck), strict=True)
        dim = 256

        def embed(x):                     # x: (N,T,17,3) → (N,1,T,V,C)
            return net.forward_features(x.unsqueeze(1))
    else:
        raise SystemExit(f"unknown model: {model_name}")
    net.to(device).eval()
    return net, embed, dim


# ---------------- Window list (same rules as train_pilot) ----------------

def train_windows(runs, T, per_run, seed):
    rng = np.random.default_rng(seed)
    out = []
    for r in runs:
        for _ in range(per_run):
            out.append((r, W.sample_start(r["segments"], T, rng)))
    return out


def eval_windows(runs, T, n_eval, min_stride_frac=0.5, max_per_rec=40):
    by_rec: dict = {}
    for r in runs:
        by_rec.setdefault((r["conductor"], r["video_id"]), []).append(r)
    items = []
    for _k, rs in by_rec.items():
        cand = []
        for r in rs:
            for s in W.window_starts(r["segments"], T, n_eval,
                                     min_stride_frac=min_stride_frac):
                cand.append((r, s))
        if max_per_rec and len(cand) > max_per_rec:
            idx = np.linspace(0, len(cand) - 1, max_per_rec)
            cand = [cand[int(round(i))] for i in idx]
        items += cand
    return items


def crop(seq: np.ndarray, s: int, T: int) -> np.ndarray:
    e = min(len(seq), s + T)
    w = seq[s:e]
    if len(w) < T:                      # protects the tail end against resampling round-off
        w = np.concatenate([w, np.repeat(w[-1:], T - len(w), axis=0)])
    return w


# ---------------- Batch embedding extraction ----------------

@torch.no_grad()
def extract(embed, items, cache, wh, model_name, T, device, batch=256,
            shuffle_rng=None):
    out = np.empty((len(items), 0), dtype=np.float32)
    feats, metas = [], []
    buf = []
    for r, s in items:
        w = crop(cache[r["run"]], s, T)
        if shuffle_rng is not None:      # S1 control: destroy order -> static cue only
            w = w[shuffle_rng.permutation(len(w))]
        if model_name == "mb":
            buf.append(prep_window_mb(w))
        else:
            buf.append(prep_window_stgcn(w, wh[r["run"]]))
        metas.append(r)
        if len(buf) == batch:
            x = torch.from_numpy(np.stack(buf)).to(device)
            feats.append(embed(x).float().cpu().numpy())
            buf = []
    if buf:
        x = torch.from_numpy(np.stack(buf)).to(device)
        feats.append(embed(x).float().cpu().numpy())
    return np.concatenate(feats), metas


# ---------------- Head training / evaluation ----------------

def fit_head(ftr, ytr, dim, ncls, device, epochs=200, lr=1e-2, wd=1e-4):
    head = torch.nn.Linear(dim, ncls).to(device)
    x = torch.from_numpy(ftr).to(device)
    y = torch.from_numpy(ytr).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        logit = head(x)
        loss = Fn.cross_entropy(logit, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head


@torch.no_grad()
def eval_fold(head, fte, metas, labels, device):
    p = Fn.softmax(head(torch.from_numpy(fte).to(device)), dim=1).cpu().numpy()
    per_rec = defaultdict(lambda: [None, None])
    win_by_rec = defaultdict(list)
    win_ok = 0
    for k, r in enumerate(metas):
        y = labels[r["conductor"]]
        key = (r["conductor"], r["video_id"])
        win_ok += int(p[k].argmax() == y)
        win_by_rec[key].append(bool(p[k].argmax() == y))
        cur = per_rec[key]
        cur[0] = p[k] if cur[1] is None else cur[0] + p[k]
        cur[1] = y
    rec = [(np.argmax(v[0]) == v[1]) for v in per_rec.values()]
    preds = {f"{c}|{v}": {"p": (np.asarray(val[0]) / np.sum(val[0])).tolist(),
                          "y": int(val[1])}
             for (c, v), val in per_rec.items()}
    macro = float(np.mean([np.mean(v) for v in win_by_rec.values()]))
    return {"rec_correct": int(np.sum(rec)), "rec_n": len(rec),
            "win_acc": win_ok / max(1, len(metas)), "win_macro": macro,
            "win_n": len(metas), "preds": preds}


# ---------------- Fine-tuning ----------------

class MBClassifier(torch.nn.Module):
    """Classification layer on top of the DSTformer representation (512-dim, averaged over T*17) — for full fine-tuning."""

    def __init__(self, backbone, ncls):
        super().__init__()
        self.backbone = backbone
        self.fc = torch.nn.Linear(512, ncls)

    def forward(self, x):
        rep = self.backbone.get_representation(x)
        return self.fc(rep.mean(dim=(1, 2)))


def build_ft_net(model_name: str, ncls: int, device: str):
    if model_name == "stgcnpp":
        from conductor_classifier.stgcnpp import STGCNPP
        net = STGCNPP(num_classes=60)
        ck = torch.load(REPO / "pretrained" / "stgcnpp_ntu60_xsub_hrnet_j.pth",
                        map_location="cpu", weights_only=False)
        net.load_state_dict(ck.get("state_dict", ck), strict=True)
        net.cls_head.fc_cls = torch.nn.Linear(256, ncls)
    else:
        sys.path.insert(0, str(REPO / "pretrained" / "motionbert"))
        from lib.model.DSTformer import DSTformer
        bb = DSTformer(dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                       depth=5, num_heads=8, mlp_ratio=4,
                       maxlen=243, num_joints=17)
        ck = torch.load(REPO / "pretrained" / "mb_lite.bin",
                        map_location="cpu", weights_only=False)
        sd = ck.get("model_pos", ck.get("model", ck))
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        missing, _ = bb.load_state_dict(sd, strict=False)
        core_missing = [k for k in missing if not k.startswith("head")]
        assert not core_missing, f"MotionBERT core weights missing: {core_missing[:5]}"
        net = MBClassifier(bb, ncls)
    return net.to(device)


def finetune_fold(model_name, tr_items, te_items, cache, wh, labels, T,
                  device, epochs, lr, shuffle_rng=None):
    net = build_ft_net(model_name, len(labels), device)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    def _prep(w, run):
        return (prep_window_stgcn(w, wh[run]) if model_name == "stgcnpp"
                else prep_window_mb(w))

    def _shape(x):                       # stgcnpp uses (N,M=1,T,V,C)
        return x.unsqueeze(1) if model_name == "stgcnpp" else x

    xs = np.stack([_prep(crop(cache[r["run"]], s, T), r["run"])
                   for r, s in tr_items])
    ys = np.array([labels[r["conductor"]] for r, _ in tr_items])
    n = len(xs)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, 64):
            idx = order[i:i + 64]
            x = _shape(torch.from_numpy(xs[idx])).to(device)
            y = torch.from_numpy(ys[idx]).to(device)
            loss = Fn.cross_entropy(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    net.eval()

    def _win(r, s):
        w = crop(cache[r["run"]], s, T)
        if shuffle_rng is not None:
            w = w[shuffle_rng.permutation(len(w))]
        return _prep(w, r["run"])

    p_all, metas = [], []
    with torch.no_grad():
        for i in range(0, len(te_items), 256):
            chunk = te_items[i:i + 256]
            x = np.stack([_win(r, s) for r, s in chunk])
            x = _shape(torch.from_numpy(x)).to(device)
            p_all.append(Fn.softmax(net(x), dim=1).cpu().numpy())
            metas += [r for r, _ in chunk]
    p = np.concatenate(p_all)
    per_rec = defaultdict(lambda: [None, None])
    win_by_rec = defaultdict(list)
    win_ok = 0
    for k, r in enumerate(metas):
        y = labels[r["conductor"]]
        key = (r["conductor"], r["video_id"])
        win_ok += int(p[k].argmax() == y)
        win_by_rec[key].append(bool(p[k].argmax() == y))
        cur = per_rec[key]
        cur[0] = p[k] if cur[1] is None else cur[0] + p[k]
        cur[1] = y
    rec = [(np.argmax(v[0]) == v[1]) for v in per_rec.values()]
    preds = {f"{c}|{v}": {"p": (np.asarray(val[0]) / np.sum(val[0])).tolist(),
                          "y": int(val[1])}
             for (c, v), val in per_rec.items()}
    macro = float(np.mean([np.mean(v) for v in win_by_rec.values()]))
    return {"rec_correct": int(np.sum(rec)), "rec_n": len(rec),
            "win_acc": win_ok / max(1, len(metas)), "win_macro": macro,
            "win_n": len(metas), "preds": preds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/experiments/snapshot5_20260812.csv")
    ap.add_argument("--model", required=True, choices=["mb", "stgcnpp"])
    ap.add_argument("--finetune", action="store_true",
                    help="Full fine-tuning — the default is a frozen probe "
                         "(for mb, --ft-lr 1e-4 is recommended for transformer stability on small data)")
    ap.add_argument("--shuffle-test", action="store_true",
                    help="Randomize the frame order of evaluation windows — the remaining "
                         "score is the portion attributable to static cues (body build, posture)")
    ap.add_argument("--mask-legs", action="store_true",
                    help="Zero out knees/ankles (13..16) — checks whether off-screen "
                         "extrapolated garbage coordinates are dragging down the pretrained "
                         "model (pyskl's missing-value convention)")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--windows-per-run", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="data/experiments")
    ap.add_argument("--seg-verdicts", default="",
                    help="Segment identity-verdict CSV — excludes contaminated segments (same as train_pilot)")
    ap.add_argument("--session-merges", default="",
                    help="Manual session-merge CSV confirmed via audio comparison (same as train_pilot)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from conductor_classifier.segver import load_blacklist
    seg_bl = load_blacklist(args.seg_verdicts) if args.seg_verdicts else frozenset()
    runs = TP.load_runs(Path(args.csv), Path("."), args.T, tuple(args.only),
                        seg_blacklist=seg_bl)
    labels = {c: i for i, c in enumerate(sorted({r["conductor"] for r in runs}))}
    recs = sorted({(r["conductor"], r["video_id"]) for r in runs})
    print(f"[{args.tag}] run {len(runs)} - recordings {len(recs)} - {labels} - {device}")

    cache = {r["run"]: load_raw17(r) for r in runs}
    if args.mask_legs:
        for a in cache.values():
            a[:, 13:17, :] = 0.0
    wh = {r["run"]: frame_norm_params(cache[r["run"]]) for r in runs}

    # Session folds — same as train_pilot's main (same seed -> same split)
    titles = {}
    for m in csv.DictReader(Path("data/recordings_meta.csv").open()):
        titles.setdefault(m["conductor"], {})[m["video_id"].strip()] = m["title"]
    merge_ov = load_merge_overrides(args.session_merges) if args.session_merges else {}
    sess_of = {}
    for c in sorted({r["conductor"] for r in runs}):
        vids = {r["video_id"] for r in runs if r["conductor"] == c}
        tmap = {v: titles.get(c, {}).get(v, "") for v in vids}
        g = group_sessions(tmap)
        if c in merge_ov:
            g = apply_merge_overrides(g, merge_ov[c])
        for v, sid in g.items():
            sess_of[(c, v)] = (c, sid)
    sessions = sorted({sess_of[k] for k in recs})
    print(f"Session grouping: {len(recs)} recording(s) -> {len(sessions)} session(s)")
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(sessions))
    fold_of_sess = {sessions[j]: i % args.folds for i, j in enumerate(order)}
    fold_of = {k: fold_of_sess[sess_of[k]] for k in recs}

    t0 = time.time()
    if not args.finetune:
        net, embed, dim = load_backbone(args.model, device)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"Backbone loaded - parameters {n_params/1e6:.1f}M - embedding dim {dim}")

    all_preds: dict = {}
    tot_c = tot_n = 0
    win_accs, win_macros = [], []
    win_tot = 0
    for f in range(args.folds):
        tr = [r for r in runs if fold_of[(r["conductor"], r["video_id"])] != f]
        te = [r for r in runs if fold_of[(r["conductor"], r["video_id"])] == f]
        if not te:
            continue
        tr_items = train_windows(tr, args.T, args.windows_per_run, args.seed + f)
        te_items = eval_windows(te, args.T, args.n_eval)
        sh_rng = np.random.default_rng(args.seed) if args.shuffle_test else None
        if args.finetune:
            res = finetune_fold(args.model, tr_items, te_items, cache, wh,
                                labels, args.T, device, args.ft_epochs,
                                args.ft_lr, shuffle_rng=sh_rng)
        else:
            ftr, mtr = extract(embed, tr_items, cache, wh, args.model, args.T, device)
            fte, mte = extract(embed, te_items, cache, wh, args.model, args.T, device,
                               shuffle_rng=sh_rng)
            ytr = np.array([labels[r["conductor"]] for r in mtr])
            head = fit_head(ftr, ytr, dim, len(labels), device)
            res = eval_fold(head, fte, mte, labels, device)
        all_preds.update(res.pop("preds"))
        tot_c += res["rec_correct"]; tot_n += res["rec_n"]
        win_accs.append(res["win_acc"]); win_macros.append(res["win_macro"])
        win_tot += res["win_n"]
        print(f"  fold {f}: recording {res['rec_correct']}/{res['rec_n']} "
              f"- window {res['win_acc']:.3f} / macro {res['win_macro']:.3f} "
              f"({res['win_n']})")

    cnt = defaultdict(int)
    for c, _ in recs:
        cnt[c] += 1
    chance = max(cnt.values()) / len(recs)
    acc = tot_c / max(1, tot_n)
    z = (acc - chance) / np.sqrt(max(1e-9, chance * (1 - chance) / max(1, tot_n)))
    out = {"tag": args.tag, "model": args.model, "finetune": args.finetune,
           "rec_acc": acc, "rec_n": tot_n, "chance": chance, "z": float(z),
           "win_acc": float(np.mean(win_accs)), "win_macro": float(np.mean(win_macros)),
           "win_n": win_tot, "T": args.T, "folds": args.folds,
           "only": args.only, "csv": args.csv, "seg_verdicts": args.seg_verdicts,
           "session_merges": args.session_merges,
           "mask_legs": args.mask_legs, "shuffle_test": args.shuffle_test,
           "ft_lr": args.ft_lr, "ft_epochs": args.ft_epochs,
           "n_sessions": len(sessions), "minutes": (time.time() - t0) / 60,
           "preds": all_preds}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{args.tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n[{args.tag}] recording {acc:.3f} (chance {chance:.3f}, z={z:.1f}) - "
          f"window {out['win_acc']:.3f} / macro {out['win_macro']:.3f} (n={win_tot}) - "
          f"{out['minutes']:.1f} min -> {args.out}/{args.tag}.json")


if __name__ == "__main__":
    main()

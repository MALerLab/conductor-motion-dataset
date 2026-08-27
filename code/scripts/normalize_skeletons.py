"""Raw skeletons -> temporal (frame) normalization -> spatial (skeleton) normalization.

Takes the output of extract_skeletons_from_clips.py
(<root>/<conductor>/<clip>__r<NN>/skeleton.npy, meta.native_fps) and, in order:
  1) resample_to_fps(native -> 25)  : common fps (time axis, downsampling direction)
  2) lowpass_temporal(25, cutoff)   : common bandwidth (unify smoothness across eras)
  3) spatial normalization:
       --mode shoulder : origin at shoulder midpoint + shoulder-width scale  (body shape (a) -- size only)
       --mode bone     : bone lengths pinned to a common canonical           (body shape (b) -- size + proportions)
Saves the result as skeleton_norm.npy in the same folder and records the
normalization parameters in meta.

Doing the temporal steps (1, 2) on raw coordinates before the spatial step (3)
matters: it keeps per-frame scale noise from leaking into the temporal filter.

mode=bone needs a global canonical bone length, so it runs in 2 passes
(full scan -> apply). The canonical value is a per-shoulder-width-unit median,
so it is invariant to resolution/body shape.

Usage:
    PYTHONPATH=src python scripts/normalize_skeletons.py \
        --root skeletons_A --mode bone --target-fps 25 --cutoff-hz 6.0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from conductor_classifier.pose import (
    bone_lengths,
    normalize_bone_lengths,
    normalize_skeleton,
)
from conductor_classifier.temporal import lowpass_temporal, resample_to_fps

_EPS = 1e-6


def _temporal(raw, native, fps, cutoff):
    return lowpass_temporal(resample_to_fps(raw, native, fps), fps, cutoff)


def _shoulder_width(seq):
    from conductor_classifier import keypoints as kp
    w = np.linalg.norm(seq[:, kp.LEFT_SHOULDER, :2] - seq[:, kp.RIGHT_SHOULDER, :2], axis=1)
    return w


def _fore_factors(seq, min_conf=0.3):
    """(T,len(BONES)) foreshortening ratio -- computed on post-lowpass coordinates (noise already attenuated).

    r = (per-frame 2D bone length / shoulder width) / run p95.  p95 = "the moment
    the limb looked most extended" ~= a proxy for true length (Taylor 2000: at the
    instant a limb is screen-parallel, l -> s*L) -> r ~= |cos theta|.
    Invalid frames default to 1 (full length = unchanged motion). Clipped to [0.15, 1.15].
    """
    from conductor_classifier.pose import BONES, ROOT
    T = len(seq)
    ls, rs = seq[:, 5, :2], seq[:, 6, :2]
    sw = np.linalg.norm(ls - rs, axis=1)
    sc = (ls + rs) / 2.0
    sw_ok = (sw > 1e-6) & (seq[:, 5, 2] >= min_conf) & (seq[:, 6, 2] >= min_conf)
    out = np.ones((T, len(BONES)), dtype=np.float64)
    for b, (pj, cj) in enumerate(BONES):
        pxy = sc if pj == ROOT else seq[:, pj, :2]
        l = np.linalg.norm(seq[:, cj, :2] - pxy, axis=1)
        ok = sw_ok & (seq[:, cj, 2] >= min_conf)
        if pj != ROOT:
            ok = ok & (seq[:, pj, 2] >= min_conf)
        rel = np.where(ok, l / np.maximum(sw, 1e-6), np.nan)
        if not np.isfinite(rel).any():
            continue
        anchor = np.nanpercentile(rel, 95)
        if not np.isfinite(anchor) or anchor < 1e-6:
            continue
        r = rel / anchor
        out[:, b] = np.clip(np.where(np.isfinite(r), r, 1.0), 0.15, 1.15)
    return out


def compute_canonical(metas, fps, cutoff):
    """Median shoulder-width-unit bone length across all sequences -> common canonical (resolution-independent)."""
    ratios = []
    for mp in metas:
        meta = json.loads(mp.read_text())
        seq = _temporal(np.load(mp.parent / "skeleton.npy"),
                        float(meta.get("native_fps", fps)), fps, cutoff)
        bl = bone_lengths(seq)                       # (T, n_bones), NaN for low confidence
        sw = _shoulder_width(seq)                     # (T,)
        ok = sw > _EPS
        ratios.append(bl[ok] / sw[ok, None])
    allr = np.concatenate(ratios, axis=0)
    return np.nanmedian(allr, axis=0)                 # (n_bones,)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="skeletons_A")
    ap.add_argument("--mode", choices=["bone", "shoulder"], default="bone")
    ap.add_argument("--target-fps", type=float, default=25.0)
    ap.add_argument("--cutoff-hz", type=float, default=6.0)
    ap.add_argument("--canonical", default="",
                    help="Path to an existing canonical_bones.npy -- share a coordinate frame with another dataset")
    ap.add_argument("--fore", action="store_true",
                    help="Preserve foreshortening -- draw bones as canonical x r(t) to keep depth geometry")
    args = ap.parse_args()

    root = Path(args.root)
    metas = sorted(root.glob("*/*/meta.json"))
    fps, cut = args.target_fps, args.cutoff_hz

    canonical = None
    if args.mode == "bone":
        if args.canonical:
            # Must match **the same coordinate frame** as another dataset for the
            # model to be applied across them (e.g. the bridge test feeding
            # skeletons_H into a model trained on skeletons_A).
            canonical = np.load(args.canonical)
            print(f"reusing canonical: {args.canonical} ({canonical.shape[0]} bones)")
        else:
            canonical = compute_canonical(metas, fps, cut)
            print(f"computed canonical bone lengths (shoulder-width units): {canonical.shape[0]} bones")
        np.save(root / "canonical_bones.npy", canonical)

    n = 0
    for mp in metas:
        meta = json.loads(mp.read_text())
        seq = _temporal(np.load(mp.parent / "skeleton.npy"),
                        float(meta.get("native_fps", fps)), fps, cut)
        fore = _fore_factors(seq) if (args.mode == "bone" and args.fore) else None
        seq = (normalize_bone_lengths(seq, canonical, fore=fore)
               if args.mode == "bone" else normalize_skeleton(seq))
        np.save(mp.parent / "skeleton_norm.npy", seq)
        meta.update({"norm_fps": fps, "lowpass_cutoff_hz": cut,
                     "norm_mode": args.mode + ("_fore" if args.fore else ""),
                     "norm_n_frames": int(seq.shape[0])})
        mp.write_text(json.dumps(meta, indent=2))
        n += 1
    print(f"normalization done: {n} sequences -> skeleton_norm.npy "
          f"(fps={fps}, cutoff={cut}Hz, mode={args.mode})")


if __name__ == "__main__":
    main()

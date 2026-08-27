"""How well does the image-quality proxy (mean keypoint confidence) predict recording era -- reproduction script.

Source of the paper's cited figure (AUC 0.79). This is the positive control for
the bridge test (bridge_test.py): instead of model scores, it checks "can the
year label and the era_auc statistic detect an era signal at all" using the
image-quality proxy. Definitions:
  proxy for recording r = the arithmetic mean confidence (over all frames x 133
                           joints) across the raw skeleton.npy of every run in that recording
  era_auc = the AUC of the proxy separating recordings before/after year 2000 (bridge.era_auc)
Cohort = same as bridge_test: runs in skeletons_H/Haitink with clean segments >= T,
year from recordings_meta.csv (year_manual > year_desc > year_title).

Usage: PYTHONPATH=src:scripts python scripts/verify_conf_era_proxy.py
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from conductor_classifier import bridge as BR
from bridge_test import load_third, load_years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="skeletons_H")
    ap.add_argument("--third", default="Haitink")
    ap.add_argument("--meta", default="data/recordings_meta.csv")
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--cutoff", type=int, default=2000)
    args = ap.parse_args()

    runs = load_third(Path(args.root), args.third, args.T)
    years = load_years(Path(args.meta), args.third)
    conf_sum, conf_n, low_sum = defaultdict(float), defaultdict(int), defaultdict(float)
    for r in runs:
        raw = np.load(Path(r["npy"]).parent / "skeleton.npy")   # raw coordinates + confidence
        c = raw[:, :, 2]
        conf_sum[r["video_id"]] += float(c.sum()); conf_n[r["video_id"]] += c.size
        low_sum[r["video_id"]] += float((c.mean(axis=1) < 0.3).sum())
    mean_conf = {v: conf_sum[v] / conf_n[v] for v in conf_sum}
    low_frac = {v: low_sum[v] / (conf_n[v] / 133) for v in conf_sum}
    dated = {v: s for v, s in mean_conf.items() if v in years}
    old = [v for v in dated if years[v] < args.cutoff]
    new = [v for v in dated if years[v] >= args.cutoff]
    auc_conf = BR.era_auc(dated, years, args.cutoff)
    rho_conf = BR.spearman([years[v] for v in dated], [dated[v] for v in dated])
    lf = {v: low_frac[v] for v in dated}
    auc_low = BR.era_auc(lf, years, args.cutoff)
    print(f"[{args.third}] {len(runs)} runs · {len(mean_conf)} recordings · {len(dated)} with a year "
          f"(<{args.cutoff}: {len(old)} / >={args.cutoff}: {len(new)})")
    print(f"mean keypoint confidence -> era AUC = {auc_conf:.3f}  (Spearman rho = {rho_conf:+.3f})")
    print(f"low-confidence frame fraction -> era AUC = {auc_low:.3f}")
    print(f"mean-confidence median: old {np.median([dated[v] for v in old]):.3f} / recent {np.median([dated[v] for v in new]):.3f}")


if __name__ == "__main__":
    main()

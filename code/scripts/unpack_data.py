"""Unpack skeleton packs into the per-run directory layout the code expects.

Creates  skeletons/<Conductor>/<run>/{skeleton.npy, qc.json, meta.json}
from     data/packs/<Conductor>_XX.npz  +  data/meta_<Conductor>.json

Run from the release root:  python code/scripts/unpack_data.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]     # release root


def main():
    packs = sorted((ROOT / "data" / "packs").glob("*.npz"))
    if not packs:
        raise SystemExit("no packs found under data/packs/")
    metas = {}
    for mj in (ROOT / "data").glob("meta_*.json"):
        c = mj.stem.removeprefix("meta_")
        metas[c] = json.loads(mj.read_text())
    n = 0
    for p in packs:
        conductor = p.stem.rsplit("_", 1)[0]
        with np.load(p) as z:
            for run in z.files:
                d = ROOT / "skeletons" / conductor / run
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / "skeleton.npy", z[run])
                m = metas[conductor][run]
                (d / "qc.json").write_text(json.dumps(m["qc"]))
                (d / "meta.json").write_text(json.dumps(m["meta"]))
                n += 1
        print(f"{p.name}: done")
    print(f"unpacked {n} runs → skeletons/")


if __name__ == "__main__":
    main()

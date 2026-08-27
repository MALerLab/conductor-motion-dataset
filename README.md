# Conductor Motion Dataset — skeletons, code, and weights

Companion release for *"Conductor Identity in Motion: A Skeleton Corpus and
Identifiability Study From In-the-Wild Concert Video"* (ISMIR 2026 LBD submission).

2D skeleton sequences of five orchestra conductors — **Bernstein, Haitink,
Abbado, Mehta, Dudamel** — extracted from in-the-wild concert videos spanning
1959–2021, with quality-control metadata, the identification code, and trained
classifier weights.

| | |
|---|---|
| Accepted runs (human-verified identity) | 1,318 |
| Recordings / sessions | 267 / 195 (sessions merged by title match + audio cross-correlation, incl. duplicate-upload dedup) |
| Total frames (native fps) | ~317k (~3.2 h) |
| Keypoints | 133 (COCO-WholeBody, RTMPose) — facial 68 zeroed |
| Identification accuracy (session-disjoint, stratified group k-fold, 5 seeds) | 0.715±0.010 per 2-s window (chance 0.302) · 0.730±0.011 per shot |

## What is in the box

```
data/
  packs/<Conductor>_XX.npz   compressed skeletons, one array per run (T,133,3)=(x,y,conf)
  meta_<Conductor>.json      per-run QC + provenance metadata
  run_index.csv              run catalog (also serves as the training CSV)
  recordings_meta.csv        per-video titles and recording-year fields
  canonical_bones.npy        canonical bone lengths (needed for normalization)
  experiments/
    segver_20260813.csv           segment-level identity audit (contaminated segments excluded)
    session_merges_v10_20260826.csv  session merges — declared title/composer/cycle rule
                                   + audio cross-correlation confirmation (see paper §3)
    audio_same_20260826.csv       true-duplicate video clusters (re-uploads/excerpts) to
                                   drop down to one representative per cluster
    rehearsal_ood_20260826.csv    held-out out-of-distribution videos (rehearsal footage) —
                                   pass --ood-exclude at training time to keep these out
code/
  src/conductor_classifier/  model, graph, channels, sessions, windows, QC ...
  scripts/                   unpack_data.py, normalize_skeletons.py,
                             train_pilot.py (session-disjoint CV eval), train_final_model.py,
                             probe_pretrained.py (MotionBERT / ST-GCN++ baselines),
                             verify_repertoire.py, verify_conf_era_proxy.py, bridge_test.py
weights/
  resgcn_5class.pt           ResGCN (~320k params) trained on all accepted runs
```

Raw video and audio are **not** included (copyright). `recordings_meta.csv`
and each run's `meta.json` carry the YouTube video id, source clip, and frame
range, so provenance is fully traceable.

## Keypoint layout

COCO-WholeBody order: 0–16 body (COCO-17), 17–22 feet, 23–90 face
(**zeroed in this release** — facial geometry of real persons is withheld;
our models never use it), 91–132 left/right hands (21 each). Third channel
is per-joint detector confidence, *not* depth.

## Quickstart

```bash
# 1. unpack npz packs into per-run directories  → skeletons/<Conductor>/<run>/
python code/scripts/unpack_data.py

# 2. normalize (25 fps, 6 Hz low-pass, canonical bone lengths, fore/depth-preserving)
PYTHONPATH=code/src python code/scripts/normalize_skeletons.py \
    --root skeletons --mode bone --fore --canonical data/canonical_bones.npy

# 3. reproduce the session-disjoint experiment (~10 min on one GPU)
PYTHONPATH=code/src python code/scripts/train_pilot.py \
    --csv data/run_index.csv --joints both --no-depth --tag reproduce \
    --seg-verdicts data/experiments/segver_20260813.csv \
    --session-merges data/experiments/session_merges_v10_20260826.csv \
    --dup-drops data/experiments/audio_same_20260826.csv \
    --ood-exclude data/experiments/rehearsal_ood_20260826.csv

# 4. repertoire controls (matched composers, shared pieces) from the saved predictions
PYTHONPATH=code/src python code/scripts/verify_repertoire.py \
    data/experiments/reproduce.json \
    --session-merges data/experiments/session_merges_v10_20260826.csv
```

Windows are sampled on the fly (2 s, random phase); evaluation folds are
**session-disjoint** (stratified group k-fold) — duplicate/split uploads of one
performance are merged into one session by a declared title/composer/cycle
rule plus audio cross-correlation, true-duplicate re-uploads are deduplicated
via `audio_same_20260826.csv`, and held-out rehearsal footage is blocked at
training time via `--ood-exclude`. Contaminated segments (skeleton switching
to a nearby musician, ~1.9% of the corpus by duration) are excluded via
`segver_20260813.csv`. See the paper for why.

`train_pilot.py` also accepts `--pad {none,zero,zeromask}` to salvage segments
shorter than the window length instead of discarding them (see `--help`) —
an evaluation in progress on top of the settings above; not yet folded into
the headline number.

### Using the trained classifier

```python
import torch
from conductor_classifier.resgcn import ResGCN
from conductor_classifier import graph as G, channels as CH

ck = torch.load("weights/resgcn_5class.pt", map_location="cpu")
model = ResGCN(ck["input_channels"], len(ck["labels"]), G.adjacency(ck["joints"]))
model.load_state_dict(ck["state_dict"]); model.eval()
# input: a (50,133,3) window of *normalized* skeleton  →  CH.build_channels(...)
```

Note: `weights/resgcn_5class.pt` is trained on **all** accepted runs
(contaminated segments excluded; the usual convention for released
checkpoints), and predates the session-merge/dedup/OOD-guard update described
above — a refreshed checkpoint trained under the current protocol is planned
for a future release. The 0.715 / 0.730 figures come from the session-disjoint
evaluation protocol, not from this specific checkpoint.

## QC philosophy (why the data looks the way it does)

Pose spikes and identity switches are detected and only clean segments ≥2 s
are used, but **image quality is never a reason to discard**: in a corpus
spanning six decades, quality is a proxy for recording era, and filtering on
it would curate the dataset by era. QC decisions live in metadata only —
thresholds can be revised without re-running pose estimation.

## License

- **Code** (`code/`, `weights/`): MIT — see `LICENSE`.
- **Data** (`data/`): CC BY-NC 4.0 — see `LICENSE-DATA`. Skeleton coordinates
  are derived, non-photographic data; source videos remain the property of
  their rights holders and must be obtained from the original sources.

## Citation

```bibtex
@inproceedings{kim2026conductor,
  title  = {Conductor Identity in Motion: A Skeleton Corpus and
            Identifiability Study From In-the-Wild Concert Video},
  author = {Kim, Jiyun and Jeong, Dasaem},
  booktitle = {ISMIR Late-Breaking Demo},
  year   = {2026}
}
```

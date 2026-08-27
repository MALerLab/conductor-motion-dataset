from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Frontal-conductor detection/extraction parameters (beat-labeling fields have been removed).

    Raising max_frontal_angle_deg (e.g. 30 -> 55) lets 3/4-profile shots pass too.
    Lowering min_shoulder_width_fraction also captures conductors in wide shots.
    """

    data_root: Path = Path("data")

    # Frontal filter
    min_clip_seconds: float = 10.0
    frontal_max_gap_seconds: float = 0.3  # bridge short non-frontal gaps (motion blur, momentary occlusion)
    max_frontal_angle_deg: float = 30.0
    min_keypoint_conf: float = 0.3
    center_band_fraction: float = 0.55  # allowed center-band width fraction (0.55 = +/-27.5% of frame width)
    min_shoulder_width_fraction: float = 0.08

    # Arm/hand crop check (automates the manual A/B curation). filter.wrists_in_frame/hands_in_frame.
    edge_margin_fraction: float = 0.02   # edge band = 2% of min(W,H)
    crop_ok_min_fraction: float = 0.70   # fraction of crop-OK frames >= this -> ACCEPT
    crop_hold_min_fraction: float = 0.50  # in between -> HOLD, below -> REJECT
    hand_cut_point_fraction: float = 0.30  # if this fraction or more of one hand's keypoints fall in the edge band, call it "cropped"

    # Face-recognition identity verification (identity.py). Based on buffalo_l cosine similarity.
    # Calibrated empirically on Bernstein/Dudamel (calibrate_identity --raw-check):
    #   same-conductor median 0.35 / cross-conductor p90 0.11 -> accept 0.30, reject 0.12.
    #   (Clean reference-to-reference pairs are ~1.0, but shaky/low-quality conducting
    #   frames score much lower.)
    #   The shot score is aggregated as the median over multiple frames, so it's more
    #   stable than a single frame.
    #   Recalibrate if conductors/data are added.
    id_accept: float = 0.30   # >= -> identity ACCEPT
    id_reject: float = 0.12   # <  -> identity REJECT; in between/None -> HOLD
    id_min_faces: int = 3     # minimum face detections needed to trust the match score

    # Shot-level frontal-fraction bands (qualify.decide). Fraction of sampled frames passing is_frontal.
    frontal_accept_fraction: float = 0.25  # >= -> frontal OK
    frontal_hold_fraction: float = 0.15    # in between -> HOLD, below -> REJECT
    frontal_min_count: int = 3             # minimum count of frontal frames (guards against chance false positives)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import keypoints as kp
from .config import PipelineConfig


@dataclass
class FrontalResult:
    ok: bool
    reasons: list[str]


def _conf_ok(kpts: np.ndarray, idx: int, cfg: PipelineConfig) -> bool:
    return kpts[idx, 2] >= cfg.min_keypoint_conf


def is_frontal(kpts: np.ndarray, img_width: int, cfg: PipelineConfig) -> FrontalResult:
    """Determine whether a COCO-17 keypoint array (17,3) shows a near-frontal conductor."""
    reasons: list[str] = []

    if not (_conf_ok(kpts, kp.LEFT_SHOULDER, cfg) and _conf_ok(kpts, kp.RIGHT_SHOULDER, cfg)):
        reasons.append("shoulder not visible")
        return FrontalResult(False, reasons)

    if not (_conf_ok(kpts, kp.LEFT_WRIST, cfg) and _conf_ok(kpts, kp.RIGHT_WRIST, cfg)):
        reasons.append("wrist not visible")

    lsx, rsx = kpts[kp.LEFT_SHOULDER, 0], kpts[kp.RIGHT_SHOULDER, 0]
    shoulder_w = abs(lsx - rsx)
    if shoulder_w < cfg.min_shoulder_width_fraction * img_width:
        reasons.append("person too small")

    mid_x = (lsx + rsx) / 2.0
    half = cfg.center_band_fraction / 2.0
    if not (img_width * (0.5 - half) <= mid_x <= img_width * (0.5 + half)):
        reasons.append("off center")

    if _conf_ok(kpts, kp.NOSE, cfg):
        nose_off = abs(kpts[kp.NOSE, 0] - mid_x) / max(shoulder_w, 1e-6)
        # Normalize the nose offset by shoulder width. Factor 2 maps the shoulder
        # edge (offset 0.5) to 45deg, so the 30deg threshold treats roughly the
        # inner +/-29% of shoulder width as frontal.
        angle = np.degrees(np.arctan(2.0 * nose_off))
        if angle > cfg.max_frontal_angle_deg:
            reasons.append("face not frontal")
    else:
        reasons.append("nose not visible")

    return FrontalResult(len(reasons) == 0, reasons)


def _edge_margin(img_width: int, img_height: int, cfg: PipelineConfig) -> float:
    """Edge band width (px) = min(W,H) * edge_margin_fraction."""
    return cfg.edge_margin_fraction * min(img_width, img_height)


def _inside_safe_box(x: float, y: float, w: int, h: int, m: float) -> bool:
    """True if (x,y) is within [m, w-m] x [m, h-m] (i.e. at least m px from the frame edge)."""
    return m <= x <= (w - m) and m <= y <= (h - m)


def wrists_in_frame(kpts17: np.ndarray, img_width: int, img_height: int,
                    cfg: PipelineConfig) -> bool:
    """True if both COCO-17 wrists (9,10) are clearly inside the frame's safe area (cheap stage).

    A wrist counts as 'cropped out' if it has low confidence or falls inside/outside
    the edge band. This extends is_frontal's 'wrist not visible' signal with a
    geometric check. Used at the YOLO sampling stage.
    """
    m = _edge_margin(img_width, img_height, cfg)
    for idx in (kp.LEFT_WRIST, kp.RIGHT_WRIST):
        if kpts17[idx, 2] < cfg.min_keypoint_conf:
            return False
        if not _inside_safe_box(kpts17[idx, 0], kpts17[idx, 1], img_width, img_height, m):
            return False
    return True


# RTMPose WholeBody(133): left hand 91-111, right hand 112-132 (21 points each).
_LEFT_HAND = list(range(91, 112))
_RIGHT_HAND = list(range(112, 133))


def hands_in_frame(kpts133: np.ndarray, img_width: int, img_height: int,
                   cfg: PipelineConfig) -> bool:
    """True if the WholeBody-133 wrist + both-hand keypoints are not cropped out of frame (authoritative stage).

    A hand is considered 'cropped' if the fraction of its 'present' (conf>=min)
    keypoints that fall inside the edge band is >= hand_cut_point_fraction. If
    either wrist is low-confidence or outside the safe area, it's an immediate
    crop. Only run on shots that already passed the frontal+identity checks, to
    bound the cost.
    """
    m = _edge_margin(img_width, img_height, cfg)

    # Wrists are a strong signal -- both must be inside the safe area.
    for idx in (kp.LEFT_WRIST, kp.RIGHT_WRIST):
        if kpts133[idx, 2] < cfg.min_keypoint_conf:
            return False
        if not _inside_safe_box(kpts133[idx, 0], kpts133[idx, 1], img_width, img_height, m):
            return False

    for hand in (_LEFT_HAND, _RIGHT_HAND):
        pts = kpts133[hand]
        present = pts[pts[:, 2] >= cfg.min_keypoint_conf]
        if len(present) == 0:
            continue  # No fingers detected (occlusion, etc.) -- deferred to the wrist check, so pass here
        cut = sum(
            not _inside_safe_box(p[0], p[1], img_width, img_height, m) for p in present
        )
        if cut / len(present) >= cfg.hand_cut_point_fraction:
            return False
    return True


def crop_ok_fraction(flags: list[bool]) -> float:
    """Fraction of per-frame 'not cropped' flags. An empty list yields 0.0 (treated as cropped)."""
    return (sum(flags) / len(flags)) if flags else 0.0


def frontal_spans(
    flags: list[bool], fps: float, min_seconds: float, max_gap_seconds: float = 0.0
) -> list[tuple[int, int]]:
    """List of frontal (start, end) index spans with length >= min_seconds.

    Short non-frontal gaps of max_gap_seconds or less (brief drops in keypoint
    confidence from motion blur, momentary occlusion, etc.) are bridged to join
    the spans together (morphological closing).
    """
    min_frames = int(round(min_seconds * fps))
    max_gap = int(round(max_gap_seconds * fps))

    # 1) Collect raw True runs
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))

    # 2) Merge adjacent runs whose gap is <= max_gap
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= max_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # 3) Length filter (post-merge length must be >= min_frames)
    return [(s, e) for s, e in merged if e - s + 1 >= min_frames]


def frontal_spans_from_detector(
    frame_indices,
    fps: float,
    detect: Callable[[int, object], tuple[np.ndarray, int]],
    cfg: PipelineConfig,
) -> list[tuple[int, int]]:
    """detect(frame_idx, frame) -> (keypoints(17,3), img_width).

    frame_indices: an iterable of absolute frame numbers to check (e.g. the
    range of a single shot). Runs the frontal check per frame, merges
    consecutive spans, and returns a list of absolute-frame (start, end) spans.
    NOTE: detect owns video I/O directly -- implement it with sequential
    decoding (random seeking per frame is extremely slow).
    """
    frames = list(frame_indices)
    flags: list[bool] = []
    for i in frames:
        kpts, img_w = detect(i, None)
        flags.append(is_frontal(kpts, img_w, cfg).ok)
    rel = frontal_spans(flags, fps, cfg.min_clip_seconds, cfg.frontal_max_gap_seconds)
    return [(frames[s], frames[e]) for s, e in rel]

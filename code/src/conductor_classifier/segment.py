from dataclasses import dataclass
from pathlib import Path


@dataclass
class Shot:
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


def shots_from_scene_list(scene_list, fps: float) -> list[Shot]:
    """Convert a list of (start_frame, end_frame) pairs into a list of Shot."""
    shots = []
    for start_f, end_f in scene_list:
        sf, ef = int(start_f), int(end_f)
        shots.append(
            Shot(
                start_frame=sf,
                end_frame=ef,
                start_sec=sf / fps,
                end_sec=ef / fps,
            )
        )
    return shots


def shot_is_conductor(frontal_flags: list[bool], min_frac: float, min_count: int) -> bool:
    """Decide whether a shot is a 'conductor cut' from its list of per-sample-frame frontal flags.

    Returns True only when frontal frames make up a sufficient fraction
    (min_frac) AND meet a minimum count (min_count). This keeps a shot from
    surviving on one or two frames of a chance false positive.
    """
    if not frontal_flags:
        return False
    n_true = sum(frontal_flags)
    return n_true >= min_count and n_true / len(frontal_flags) >= min_frac


def detect_shots(video_path: Path, fps: float) -> list[Shot]:
    """Detect shot boundaries with PySceneDetect. scenedetect is imported lazily.

    If there are no cuts, scenedetect returns an empty list; in that case treat
    the whole video as a single shot (the most common case is a single
    continuous conducting clip).
    """
    from scenedetect import ContentDetector, detect, open_video  # noqa: PLC0415

    scenes = detect(str(video_path), ContentDetector())
    scene_list = [(s.get_frames(), e.get_frames()) for s, e in scenes]
    if not scene_list:
        total = open_video(str(video_path)).duration.get_frames()
        scene_list = [(0, total)]
    return shots_from_scene_list(scene_list, fps)

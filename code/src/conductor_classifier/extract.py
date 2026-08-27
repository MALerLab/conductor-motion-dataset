"""Core functions for extracting a front-facing conductor's skeleton from local video.

Picks the conductor out of multi-person frames (orchestra wide shots) via
pick_conductor, crops to the conductor's bbox, runs RTMPose WholeBody (133
keypoints), and maps the coordinates back to the original frame. Beat/audio
are not handled here (this is a classification task).
"""
from pathlib import Path

import numpy as np

from . import keypoints as kp
from .config import PipelineConfig
from .filter import is_frontal
from .pose import add_crop_offset


def frontal_score(k: np.ndarray) -> float:
    """Frontalness score: the product of confidence over both shoulders, both wrists, and the nose."""
    idx = [kp.LEFT_SHOULDER, kp.RIGHT_SHOULDER, kp.LEFT_WRIST, kp.RIGHT_WRIST, kp.NOSE]
    return float(np.prod([k[i, 2] for i in idx]))


def pick_conductor(persons: np.ndarray, width: int, cfg: PipelineConfig) -> np.ndarray:
    """Among (N,K,3), prefer whoever passes is_frontal; otherwise pick whoever has the highest frontal_score."""
    passing = [k for k in persons if is_frontal(k, width, cfg).ok]
    pool = passing if passing else list(persons)
    return max(pool, key=frontal_score)


def build_yolo_detector(video_path: Path, cfg: PipelineConfig):
    """Run YOLO-pose once over every frame and cache the COCO-17 conductor keypoints.

    Returns: (detect, n_frames, width).  detect(i, _) -> (kpts(17,3), width).
    The YOLO weights file 'yolov8n-pose.pt' must live in the working directory
    (repo root).
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    cache: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            cache.append(np.zeros((17, 3)))
            continue
        data = res.keypoints.data.cpu().numpy()  # (N,17,3)
        cache.append(pick_conductor(data, width, cfg))
    cap.release()

    def detect(frame_idx: int, _frame) -> tuple[np.ndarray, int]:
        if frame_idx >= len(cache):  # guard against scenedetect reporting more frames than actually decoded
            return np.zeros((17, 3)), width
        return cache[frame_idx], width

    return detect, len(cache), width


def build_rtmpose_extractor_cropped(video_path: Path, width: int, cfg: PipelineConfig,
                                    device: str = "cpu"):
    """Extractor that crops to the conductor's bbox and runs RTMPose WholeBody.

    extract(start, end, bboxes) -> (T,133,3). Cropping upsamples the small subject
    to the model's input size, which sharpens the hand/baton keypoints. Within the
    crop, pick_conductor selects the conductor, and coordinates are mapped back to
    the original frame (the returned coordinate system is the original one).
    Pass device='cuda' for GPU acceleration.
    """
    import cv2
    from rtmlib import Wholebody

    model = Wholebody(mode="lightweight", backend="onnxruntime", device=device)

    def extract(start_frame: int, end_frame: int, bboxes: np.ndarray) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = []
        for j in range(end_frame - start_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            x0, y0, x1, y1 = (int(v) for v in bboxes[j])
            crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
            if crop.size == 0:
                out.append(np.zeros((133, 3)))
                continue
            kpts, scores = model(crop)  # (N,133,2), (N,133) in crop coords
            kpts, scores = np.asarray(kpts), np.asarray(scores)
            if kpts.shape[0] == 0:
                out.append(np.zeros((133, 3)))
                continue
            persons = np.concatenate([kpts, scores[:, :, None]], axis=2)  # (N,133,3)
            crop_w = x1 - x0
            chosen = pick_conductor(persons, crop_w, cfg)  # crop coords
            out.append(add_crop_offset(chosen, x0, y0))    # -> original coords
        cap.release()
        return np.stack(out)

    return extract

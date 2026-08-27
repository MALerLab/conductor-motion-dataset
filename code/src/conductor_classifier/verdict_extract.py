"""Feature extraction wrapper for the review-verdict model — GPU/IO (SigLIP, YOLO-pose, scenedetect, cache).

Heavy dependencies (torch/open_clip/ultralytics/scenedetect) are imported lazily
inside the functions that need them, so the GPU stack is not loaded on the
test path or the CSV-only path.
Embeddings are computed once per clip and cached as npz (data/verdict/features/<conductor>/<stem>.npz);
retraining on top of the cache then finishes in a few seconds.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from conductor_classifier.verdict_model import motion_stats, pool_embeddings, sample_times

EMBED_MODEL = ("ViT-B-16-SigLIP", "webli")


def load_or_extract(cache_dir: Path, stem: str, extract_fn) -> dict:
    """npz cache — load it if present, otherwise run extract_fn() and save atomically."""
    f = cache_dir / f"{stem}.npz"
    if f.exists():
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    feat = extract_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{stem}.npz.tmp"
    with tmp.open("wb") as fh:          # np.savez appends .npz to the filename, so we pass a file handle instead
        np.savez(fh, **feat)
    os.replace(tmp, f)
    return feat


def clip_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    return dur


def read_frames(path: Path, times: list[float]) -> list[np.ndarray]:
    """BGR frames at the given timestamps — timestamps that fail to read are silently skipped."""
    cap = cv2.VideoCapture(str(path))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok:
            out.append(frame)
    cap.release()
    return out


def build_embedder(device: str = "cuda"):
    """SigLIP image embedder — frames(BGR list) -> (n,d) L2-normalized embeddings."""
    import open_clip
    import torch
    from PIL import Image
    name, pretrained = EMBED_MODEL
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.to(device).eval()

    def embed(frames_bgr: list[np.ndarray]) -> np.ndarray:
        ims = torch.stack([
            preprocess(Image.fromarray(np.ascontiguousarray(f[:, :, ::-1])))
            for f in frames_bgr]).to(device)
        with torch.no_grad():
            e = model.encode_image(ims)
            e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()

    return embed


def build_pose_frac(weights: str = "yolov8n-pose.pt", conf: float = 0.4,
                    device: str = "cuda"):
    """Fraction of frames in which a person is detected — a signal for non-conducting shots (stage-front, captions, audience)."""
    from ultralytics import YOLO
    m = YOLO(weights)

    def pose_frac(frames_bgr: list[np.ndarray]) -> float:
        if not frames_bgr:
            return 0.0
        res = m.predict(frames_bgr, conf=conf, device=device, verbose=False)
        return sum(1 for r in res if len(r.boxes) > 0) / len(frames_bgr)

    return pose_frac


def scene_cuts_per_min(path: Path, span: tuple[float, float] | None = None) -> float:
    """Scene-cut density (cuts/min) — a signal for heavily edited clips (a leading cause of disqualification)."""
    from scenedetect import ContentDetector, FrameTimecode, SceneManager, open_video
    video = open_video(str(path))
    sm = SceneManager()
    sm.add_detector(ContentDetector())
    if span is not None:
        start, end = span
        video.seek(FrameTimecode(float(start), video.frame_rate))
        sm.detect_scenes(video, end_time=FrameTimecode(float(end), video.frame_rate))
        dur = end - start
    else:
        sm.detect_scenes(video)
        dur = video.duration.get_seconds()
    cuts = max(0, len(sm.get_scene_list()) - 1)
    return cuts / max(dur / 60.0, 1e-6)


def extract_clip_features(path: Path, span: tuple[float, float] | None,
                          embed_fn, pose_frac_fn=None,
                          n_frames: int = 12, motion_fps: float = 2.0) -> dict:
    """Features for a single clip — embedding pooling only.

    The four scalar features are commented out because the 2026-07-25 ablation
    found they contributed 0 (feature_vector likewise uses embeddings only).
    Scalar keys in existing cached npz files are ignored. pose_frac_fn is kept
    for caller compatibility — it is unused.
    """
    if span is None:
        start, end = 0.0, clip_duration(path)
    else:
        start, end = span
    dur = max(end - start, 0.1)

    times = [start + t for t in sample_times(dur, n_frames)]
    frames = read_frames(path, times)
    if not frames:
        raise RuntimeError(f"0 frames read: {path} span={span}")
    embs = embed_fn(frames)

    # mtimes = list(np.arange(start, end, 1.0 / motion_fps))[:120]
    # mframes = read_frames(path, mtimes)
    # gray = np.stack([cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY)
    #                  for f in mframes]) if len(mframes) >= 2 else np.zeros((1, 4, 4), np.uint8)
    # m = motion_stats(gray)

    return {
        "emb_pooled": pool_embeddings(embs).astype(np.float32),
        # "pose_frac": float(pose_frac_fn(frames)),
        # "cuts_per_min": float(scene_cuts_per_min(path, span)),
        # "motion_mean": m["motion_mean"],
        # "motion_p90": m["motion_p90"],
    }

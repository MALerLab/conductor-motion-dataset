"""Face-recognition identity verification — "is this frontal-facing person actually that conductor?"

Solves the core unresolved limitation from the devlog (is_frontal cannot distinguish the
conductor from a frontal-facing orchestra member) using InsightFace (ArcFace buffalo_l)
embeddings. Crops only the selected conductor's "head region", embeds it via
RetinaFace+ArcFace, and compares by cosine similarity against a per-conductor reference DB.

The pure numeric functions (cosine, aggregate_match, head_bbox_from_coco17, match_embedding)
are testable without insightface. Only build_face_app/embed_faces load the model.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import keypoints as kp

_EPS = 1e-9
_HEAD_IDX = [kp.NOSE, kp.LEFT_EYE, kp.RIGHT_EYE, kp.LEFT_EAR, kp.RIGHT_EAR]


# --- numeric utilities (no dependencies) --------------------------------------


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity after L2 normalization."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def aggregate_match(scores: list[float], min_faces: int) -> float | None:
    """Per-frame best-match cosines -> a cutoff score (median). None (-> HOLD) if too few detections."""
    if len(scores) < min_faces:
        return None
    return float(np.median(scores))


def select_by_identity(scores: list[float | None]) -> int | None:
    """Index of the highest per-person face-match score — selects the conductor candidate.

    If the frontal-facing decision is what picks the person, it can mistakenly pick a
    frontal-facing orchestra member instead (devlog 2026-07-14). So identity score is used
    to pick the person first, and only that person is then judged for frontal-facing/cropping.
    If all scores are None (face detection failed entirely), returns None -> caller falls back
    to center-frontal (pick_conductor).
    """
    best_i, best_s = None, -1.0
    for i, s in enumerate(scores):
        if s is not None and s > best_s:
            best_i, best_s = i, s
    return best_i


def can_auto_reject(decision: str, id_score: float | None) -> bool:
    """Whether an identity-first re-judgment can be used for review prefill — an asymmetric
    policy (devlog 7/14).

    Validated on 548 rows: REJECT automation hit 97% precision (122/126, passing), while ACCEPT
    automation had a 26% failure rate -> only REJECT is taken automatically. However, if
    id_score is None (no face detection at all — old black-and-white footage), it's left to a
    human, since all 4/77 wrongful discards fell into this category (a no-loss policy).
    """
    return decision == "reject" and id_score is not None


def head_bbox_from_coco17(kpts17: np.ndarray, width: int, height: int,
                          pad: float = 0.6, min_conf: float = 0.3,
                          min_size_frac: float = 0.04) -> tuple[int, int, int, int] | None:
    """Estimate the head ROI bbox for the selected conductor from COCO-17 keypoints.

    Uses face points (nose/eyes/ears) for the center, shoulder width (or eye spacing as a
    fallback) for scale, and extends upward (the head sits above the shoulders). Returns None
    if no face points are found or the ROI is too small (a tiny head in a wide shot) —
    insufficient identity evidence -> HOLD. Returned coordinates are in the original frame.
    """
    face = [(kpts17[i, 0], kpts17[i, 1]) for i in _HEAD_IDX if kpts17[i, 2] >= min_conf]
    if not face:
        return None
    fxs = np.array([p[0] for p in face]); fys = np.array([p[1] for p in face])
    cx, cy = float(fxs.mean()), float(fys.mean())

    ls, rs = kpts17[kp.LEFT_SHOULDER], kpts17[kp.RIGHT_SHOULDER]
    if ls[2] >= min_conf and rs[2] >= min_conf:
        scale = abs(ls[0] - rs[0])
    elif kpts17[kp.LEFT_EYE, 2] >= min_conf and kpts17[kp.RIGHT_EYE, 2] >= min_conf:
        scale = 2.5 * abs(kpts17[kp.LEFT_EYE, 0] - kpts17[kp.RIGHT_EYE, 0])
    else:
        scale = max(fxs.max() - fxs.min(), fys.max() - fys.min(), 1.0) * 2.0

    if scale < min_size_frac * min(width, height):
        return None  # head is too small — embedding would be unreliable

    half = scale * (0.5 + pad)
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half * 1.2, cy + half * 0.8  # extend further upward (include the top of the head)
    x0 = int(np.clip(x0, 0, width - 1)); x1 = int(np.clip(x1, 1, width))
    y0 = int(np.clip(y0, 0, height - 1)); y1 = int(np.clip(y1, 1, height))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


# --- reference DB ---------------------------------------------------------------


@dataclass
class FaceDB:
    """Collection of normalized embeddings per conductor, plus their centroid."""
    conductor: str
    embeddings: np.ndarray   # (N, 512), L2-normalized
    centroid: np.ndarray     # (512,) normalized mean

    def match(self, emb: np.ndarray) -> float:
        """Max cosine similarity between a candidate embedding and (centroid + each individual reference).

        The centroid gives stability; the per-reference max rescues a distinctive single shot.
        """
        best = cosine(emb, self.centroid)
        for ref in self.embeddings:
            c = cosine(emb, ref)
            if c > best:
                best = c
        return best


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, _EPS, None)


def make_face_db(conductor: str, embeddings: np.ndarray) -> FaceDB:
    """(N,512) embeddings -> FaceDB via normalization + centroid."""
    emb = normalize_rows(embeddings)
    centroid = emb.mean(axis=0)
    centroid = centroid / max(np.linalg.norm(centroid), _EPS)
    return FaceDB(conductor, emb, centroid)


def score_candidates(cand_embs: np.ndarray, db: FaceDB) -> np.ndarray:
    """db.match score for each candidate embedding (M,D) -> (M,)."""
    cands = normalize_rows(cand_embs)
    return np.array([db.match(c) for c in cands])


def select_diverse(embs: np.ndarray, scores: np.ndarray, k: int, min_score: float,
                   dup_thresh: float = 0.95) -> list[int]:
    """Select up to k items — start from the highest score, then prefer whichever is "least
    similar to what's already picked."

    Excludes anything below min_score, and anything whose cosine similarity to an already-picked
    item is at or above dup_thresh (effectively the same photo). Uses greedy farthest-point
    selection so that photos spanning different eras/angles get spread out rather than piling up
    on a single-session burst: the next candidate is the one whose maximum similarity to the
    picked set is lowest (ties broken by higher score)."""
    scores = np.asarray(scores)
    eligible = [i for i in range(len(scores)) if scores[i] >= min_score]
    if not eligible:
        return []
    embs = normalize_rows(embs)
    picked = [max(eligible, key=lambda i: scores[i])]
    while len(picked) < k:
        best, best_key = None, None
        for i in eligible:
            if i in picked:
                continue
            max_sim = max(cosine(embs[i], embs[j]) for j in picked)
            if max_sim >= dup_thresh:
                continue
            key = (-max_sim, scores[i])
            if best_key is None or key > best_key:
                best, best_key = i, key
        if best is None:
            break
        picked.append(best)
    return [int(i) for i in picked]


def loo_scores(embs: np.ndarray, members: list[int]) -> np.ndarray:
    """Leave-one-out: each embedding's average cosine similarity to the other cluster members
    (always excluding itself).

    Prevents self-match inflation to 1.0 in the anchor-less fallback, where a candidate also
    doubles as part of the reference set."""
    e = normalize_rows(embs)
    out = np.zeros(len(e))
    for i in range(len(e)):
        others = [j for j in members if j != i]
        out[i] = float(np.mean([cosine(e[i], e[j]) for j in others])) if others else 0.0
    return out


def dominant_cluster(embs: np.ndarray, thresh: float) -> list[int]:
    """Self-consistency fallback for when there's no anchor — indices of the dominant group
    (the largest mutually-similar cluster).

    Picks the medoid (the representative with the highest average similarity to the rest), then
    returns the members whose cosine similarity to the medoid is at or above thresh. Relies on
    the assumption that most candidates are photos of the actual person."""
    if len(embs) == 0:
        return []
    e = normalize_rows(embs)
    sim = e @ e.T
    medoid = int(np.argmax(sim.mean(axis=1)))
    return [int(i) for i in np.flatnonzero(sim[medoid] >= thresh)]


def cross_video_cluster(embs: np.ndarray, video_ids: list[str], thresh: float) -> list[int]:
    """Cluster of the person who appears across multiple videos in common — finds the
    conductor's face across the video set.

    Exploits the signal that guest soloists/concertmasters tend to be concentrated in one or two
    videos while the conductor appears in most of them: gathers, around each face, the neighbors
    whose cosine similarity is at or above thresh, then returns whichever cluster maximizes
    (number of distinct videos appeared in, number of members)."""
    if len(embs) == 0:
        return []
    e = normalize_rows(embs)
    sim = e @ e.T
    best: list[int] = []
    best_key = (-1, -1)
    for i in range(len(e)):
        members = np.flatnonzero(sim[i] >= thresh)
        key = (len({video_ids[j] for j in members}), len(members))
        if key > best_key:
            best_key, best = key, [int(j) for j in members]
    return best


def save_face_db(db: FaceDB, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=db.embeddings, centroid=db.centroid, conductor=db.conductor)


def load_face_db(path: str | Path) -> FaceDB:
    d = np.load(path, allow_pickle=True)
    return FaceDB(str(d["conductor"]), d["embeddings"], d["centroid"])


# --- model loading / inference (lazy insightface import) --------------------------------


def build_face_app(device: str = "cuda", det_size: tuple[int, int] = (640, 640),
                   root: str | None = None):
    """Prepare an InsightFace FaceAnalysis (buffalo_l) instance. device='cuda' -> ctx_id=0.

    If root is not specified, the model is downloaded inside userdata
    (<repo>/.cache/insightface).
    """
    from insightface.app import FaceAnalysis

    from . import insightface_root
    ctx = 0 if device == "cuda" else -1
    app = FaceAnalysis(name="buffalo_l", root=root or insightface_root())
    app.prepare(ctx_id=ctx, det_size=det_size)
    return app


def embed_faces(app, bgr_image: np.ndarray):
    """BGR image -> list of detected Face objects (each with .normed_embedding, .bbox, .det_score)."""
    return app.get(bgr_image)


def _pad(img: np.ndarray, frac: float) -> np.ndarray:
    import cv2
    m = int(max(img.shape[:2]) * frac)
    return cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=(127, 127, 127))


def detect_padded(app, bgr_image: np.ndarray, pads=(0.0, 0.3, 0.6)):
    """Since RetinaFace misses faces that fill the whole frame, retry with increasing padding
    until something is detected.

    If no face is found, retries with progressively larger padding. Because the embedding is
    extracted from the aligned face, padding doesn't distort the embedding (it only helps
    detection). Returns the faces from the first padding level that succeeds. The ladder starts
    from minimal padding — 0.3 rescues the entire reference set (det_score 0.79+), with 0.6 kept
    as a backup. Over-padding (where the face shrinks to under ~20px after the 640 resize) only
    happens above 1000%+, so this is comfortably safe.
    """
    if bgr_image is None or bgr_image.size == 0:
        return []
    for p in pads:
        faces = app.get(_pad(bgr_image, p) if p > 0 else bgr_image)
        if faces:
            return faces
    return []


def embed_best(app, bgr_image: np.ndarray):
    """(embedding, det_score, face_px) for the single sharpest face in the image. None if no face."""
    faces = detect_padded(app, bgr_image)
    if not faces:
        return None
    f = max(faces, key=lambda x: x.det_score)
    sz = int(min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]))
    return np.asarray(f.normed_embedding, float), float(f.det_score), sz


def best_match_in_crop(app, crop_bgr: np.ndarray, db: FaceDB) -> float | None:
    """Detect faces in a crop and match them against the DB, returning the highest score. None if no face."""
    faces = detect_padded(app, crop_bgr)
    if not faces:
        return None
    return max(db.match(f.normed_embedding) for f in faces)

"""Review-time visualization utility — builds a filmstrip (N frames per clip).

Used to turn one contact-sheet tile (= one clip) into a horizontal strip of N
frames instead of a single frame. Handles only frame selection/compositing;
video decoding is the caller's responsibility.
"""
from PIL import Image


def strip_frame_indices(n: int, total_frames: int) -> list[int]:
    """Split the span into n equal parts and pick the middle frame index of each.

    Avoids the very first/last frames (fade/transition artifacts), and
    guarantees a strictly increasing sequence with no duplicates even when
    total is smaller than n.
    """
    if total_frames <= 0:
        return []
    raw = [int((i + 0.5) / n * total_frames) for i in range(n)]
    out: list[int] = []
    for idx in raw:
        idx = min(idx, total_frames - 1)
        if not out or idx > out[-1]:
            out.append(idx)
    return out


def compose_strip(images: list[Image.Image], height: int) -> Image.Image:
    """Resize all frames to a common `height` and concatenate them horizontally into one image."""
    resized = [im.resize((max(1, round(im.width * height / im.height)), height))
               for im in images]
    strip = Image.new("RGB", (sum(im.width for im in resized), height))
    x = 0
    for im in resized:
        strip.paste(im, (x, 0))
        x += im.width
    return strip

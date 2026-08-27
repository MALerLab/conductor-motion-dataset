"""YouTube crawler core — search / metadata filter / download (yt-dlp wrapper).

Runs on a single IP with polite throttling, no VPN. Uses yt-dlp's ytsearch
instead of the YouTube Data API, so there's no API quota. Downloads are
pose-ready video (<=720p mp4). Failures are classified as
block/permanent/transient so retry and abort behavior are handled correctly.

The pure functions (passes_metadata, classify_failure) are testable without
dependencies; only search_video_ids/download_video spawn a yt-dlp subprocess.
"""
import json
import subprocess
from dataclasses import dataclass, field

from .registry import Conductor

# --- Download settings -------------------------------------------------------


@dataclass
class CrawlConfig:
    max_height: int = 720         # 720p is plenty for pose, and it cuts bandwidth/ban exposure
    sleep_requests: float = 1.5
    min_sleep_interval: int = 3
    max_sleep_interval: int = 8
    limit_rate: str = "3M"
    retries: int = 3
    min_view_count: int = 200     # cuts title noise (metadata gate)

    def video_format(self) -> str:
        # Prefer h264 (avc1) — opencv (cv2) can't read AV1, so AV1 is avoided.
        # avc1 mp4 -> mp4 container -> only then any codec. (If AV1 still slips
        # through, scripts/normalize_raw.py re-encodes to h264 to guarantee cv2 compatibility.)
        h = self.max_height
        return (f"bv*[height<={h}][vcodec^=avc1]+ba[ext=m4a]/"
                f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}][ext=mp4]/"
                f"bv*[height<={h}]+ba/b[height<={h}]")


@dataclass
class DownloadResult:
    video_id: str
    status: str            # "downloaded" | "permanent" | "transient" | "blocked"
    reason: str = ""


# --- Search --------------------------------------------------------------------


def search_video_ids(query: str, n: int, *, _runner=None) -> list[dict]:
    """Get a list of candidate metadata via yt-dlp ytsearch (no download).

    Uses --flat-playlist --dump-json to avoid heavy per-video extraction.
    Parses the JSONL stdout into a list of
    {id,title,duration,view_count,channel,upload_date} dicts.
    _runner is for test injection (defaults to a real yt-dlp subprocess).
    """
    cmd = [
        "yt-dlp", f"ytsearch{n}:{query}",
        "--flat-playlist", "--dump-json", "--no-warnings", "--ignore-errors",
    ]
    out = _runner(cmd) if _runner else _run_stdout(cmd)
    results: list[dict] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        results.append({
            "id": d.get("id"),
            "title": d.get("title") or "",
            "duration": d.get("duration"),
            "view_count": d.get("view_count"),
            "channel": d.get("channel") or d.get("uploader") or "",
            "upload_date": d.get("upload_date"),
        })
    return [r for r in results if r["id"]]


def _run_stdout(cmd: list[str]) -> str:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return res.stdout or ""


# --- Metadata filter (cheap gate before download) ------------------------------


def passes_metadata(meta: dict, c: Conductor, cfg: CrawlConfig) -> tuple[bool, str]:
    """First-pass filtering by duration/title/view count before download. Final identity is decided by face recognition (Phase 2).

    Alias matching against the title is deliberately weak — titles lie (e.g.
    an existing Dudamel folder containing a file titled 'Leonard Bernstein -
    Mambo'). This is only meant to be a cheap funnel.
    """
    title = meta.get("title", "").lower()
    channel = (meta.get("channel") or "").lower()

    dur = meta.get("duration")
    if dur is not None:
        if dur < c.min_duration_s:
            return False, f"too_short({dur}s)"
        if dur > c.max_duration_s:
            return False, f"too_long({dur}s)"

    if any(x.lower() in title for x in c.exclude_terms):
        return False, "exclude_term"

    alias_hit = any(a.lower() in title for a in c.aliases) or \
        any(a.lower() in channel for a in c.aliases)
    if not alias_hit:
        return False, "no_alias_in_title"

    vc = meta.get("view_count")
    if vc is not None and vc < cfg.min_view_count:
        return False, f"low_views({vc})"

    return True, "ok"


# --- Failure classification -----------------------------------------------------

_BLOCK_MARKERS = (
    "http error 429", "sign in to confirm", "not a bot", "confirm your age",
    "this video may be inappropriate", "too many requests",
)
_PERMANENT_MARKERS = (
    "private video", "video unavailable", "has been removed", "members-only",
    "this video is not available", "account associated with this video has been terminated",
    "video is no longer available", "copyright",
)


def classify_failure(stderr: str) -> str:
    """yt-dlp stderr -> 'block' | 'permanent' | 'transient'.

    block: IP/bot block — abort everything, back off, and resume later (handles
    the lack of IP "rotation" on a single IP).
    permanent: the video itself is unavailable — mark done, do not retry.
    transient: network/temporary issue — retry.
    """
    s = stderr.lower()
    if any(m in s for m in _BLOCK_MARKERS):
        return "block"
    if any(m in s for m in _PERMANENT_MARKERS):
        return "permanent"
    return "transient"


# --- Download --------------------------------------------------------------------


def download_video(video_id: str, out_dir: str, archive_path: str,
                   cfg: CrawlConfig, *, _runner=None) -> DownloadResult:
    """Download a single video (video only, throttled, idempotent via --download-archive).

    The output naming '%(title).80s [%(id)s].%(ext)s' is compatible with the
    existing video_id_of regex (\\[([\\w-]{11})\\]). Testable via _runner injection.
    """
    cmd = [
        "yt-dlp", f"https://youtu.be/{video_id}",
        "-f", cfg.video_format(),
        "--merge-output-format", "mp4",
        "-o", f"{out_dir}/%(title).80s [%(id)s].%(ext)s",
        "--download-archive", archive_path,
        "--sleep-requests", str(cfg.sleep_requests),
        "--min-sleep-interval", str(cfg.min_sleep_interval),
        "--max-sleep-interval", str(cfg.max_sleep_interval),
        "--limit-rate", cfg.limit_rate,
        "--retries", str(cfg.retries),
        "--no-overwrites", "--no-progress", "--no-warnings",
    ]
    rc, stderr = _runner(cmd) if _runner else _run_rc_stderr(cmd)
    if rc == 0:
        return DownloadResult(video_id, "downloaded")
    cls = classify_failure(stderr)
    return DownloadResult(video_id, "blocked" if cls == "block" else cls,
                          reason=stderr.strip().splitlines()[-1] if stderr.strip() else "")


def _run_rc_stderr(cmd: list[str]) -> tuple[int, str]:
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stderr or ""

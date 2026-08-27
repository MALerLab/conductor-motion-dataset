"""conductor_classifier — classifies conductors (style) from skeleton data alone.

On this box, the home directory (~/.config etc.) is either unwritable or
volatile (/tmp), so all model/config caches are pinned inside the repo
(<repo>/.cache, i.e. under userdata/jiyun). This keeps not just the final
data but also the model weight caches from leaking outside userdata.
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = _REPO / ".cache"

# Point at the repo cache instead of home (once set, downstream libraries follow it).
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("CB_CACHE", str(CACHE_DIR))
# insightface takes a root argument, and rtmlib follows XDG_CACHE_HOME, so set that too.
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))

for _sub in ("ultralytics", "matplotlib", "insightface", "xdg", "bin"):
    (CACHE_DIR / _sub).mkdir(parents=True, exist_ok=True)


def _ensure_ffmpeg() -> None:
    """Get ffmpeg without root: symlink the imageio-ffmpeg bundled binary to
    .cache/bin/ffmpeg and prepend it to PATH. If a system ffmpeg already exists,
    leave it alone."""
    bindir = CACHE_DIR / "bin"
    link = bindir / "ffmpeg"
    if not link.exists():
        try:
            import imageio_ffmpeg
            src = imageio_ffmpeg.get_ffmpeg_exe()
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
        except Exception:  # noqa: BLE001
            pass  # fall back to relying on system ffmpeg
    cur = os.environ.get("PATH", "")
    if str(bindir) not in cur.split(os.pathsep):
        os.environ["PATH"] = str(bindir) + os.pathsep + cur


_ensure_ffmpeg()


def _ensure_nvidia_libs() -> None:
    """Augment LD_LIBRARY_PATH so onnxruntime-gpu can find torch's bundled cuDNN/CUDA (.so) libs.

    pip's nvidia-* wheels put their libraries under site-packages/nvidia/<pkg>/lib.
    The dynamic linker reads LD_LIBRARY_PATH only at process startup, so if the
    path is missing we populate the environment and re-exec the interpreter once.
    Only applies when running on GPU (CB_DEVICE=cuda).
    """
    import glob
    import sys

    if os.environ.get("CB_DEVICE") != "cuda" or os.environ.get("CB_LDPATH_SET"):
        return
    try:
        import nvidia
        base = os.path.dirname(nvidia.__file__)
    except Exception:  # noqa: BLE001
        return
    libs = sorted(glob.glob(os.path.join(base, "*", "lib")))
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if not libs or all(p in cur.split(":") for p in libs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([cur] if cur else []))
    os.environ["CB_LDPATH_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_nvidia_libs()


def insightface_root() -> str:
    """insightface model download root (inside userdata)."""
    return os.environ.get("CB_CACHE", str(CACHE_DIR)) + "/insightface"

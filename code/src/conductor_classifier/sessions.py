"""Group split uploads of the same performance (movements/parts/duplicates) into one "session".

Why: if fold splitting is done at the video_id level, movement 1 of a
performance can land in train while movement 2 lands in test, defeating
per-recording LORO (same stage, costume, camera == effectively leaked labels).

Merge rule (conservative — over-merging shrinks the sample, under-merging leaks):
  Within the same conductor, after stripping movement/part markup from the
  title, two videos are the same session (union-find) if ① their sets of
  opus numbers match and ② text similarity is >= 0.8.
Different performances of the same work (e.g. two recordings of Mahler 4) can
also get merged this way, but that's an error in the safe direction.
"""
import csv as _csv
import difflib
import re
from pathlib import Path as _Path

# Movement/part markup — opus numbers ("No. 7", "Op. 92") are left untouched.
_MOVEMENT_PATTERNS = [
    r"\(\s*\d\s*/\s*\d\s*\)",                      # (1/2)
    r"\b(?:mvmt|movement|mov)\.?\s*\d+(?:\s*&\s*\d+)*\b",
    r"\b(?:part|pt)\.?\s*\d+\b",
    r"\d+\s*악장",                                  # "<N> movement" (Korean)
    r",\s*[ivx]+\b",                               # ", I", ", II" (Roman-numeral movement)
    r"\s-\s[ivx]+\b",
]
_ROMAN_TAIL = re.compile(r"\b[ivx]{1,4}\b")

# Composer guard dictionary — blocks a blind spot in title merging where "a
# different composer, same number" (e.g. Beethoven No. 2 vs Mendelssohn No. 2)
# passes both the number-set and similarity checks
# (measured 2026-08-25: 3 mis-merged groups — Abbado 2, Dudamel 1).
_COMPOSERS = [
    "beethoven", "mahler", "brahms", "tchaikovsky", "bruckner", "mozart",
    "haydn", "dvorak", "dvořák", "shostakovich", "berlioz", "wagner", "verdi",
    "ravel", "debussy", "stravinsky", "mendelssohn", "schubert", "schumann",
    "sibelius", "rossini", "bizet", "elgar", "prokofiev", "rachmaninoff",
    "bartok", "bartók", "copland", "gershwin", "orff", "puccini", "smetana",
    "liszt", "grieg", "weber", "mascagni", "reinecke",
]


def _composer_set(norm_title: str) -> frozenset[str]:
    """Composers that appear in the normalized title (with diacritics folded)."""
    fold = norm_title.replace("ř", "r").replace("ó", "o")
    return frozenset(c.replace("ř", "r").replace("ó", "o")
                     for c in _COMPOSERS if c in fold)


def normalize_title(title: str) -> str:
    """Lowercase, strip movement markup, strip punctuation, collapse whitespace."""
    t = title.lower()
    for pat in _MOVEMENT_PATTERNS:
        t = re.sub(pat, " ", t)
    t = _ROMAN_TAIL.sub(" ", t)
    t = re.sub(r"[^a-z0-9가-힣 ]", " ", t)
    return " ".join(t.split())


def work_numbers(title: str) -> set[int]:
    """The set of numbers left after normalization — opus/key numbers etc. that identify the work."""
    return {int(m) for m in re.findall(r"\d+", normalize_title(title))}


def group_sessions(titles: dict[str, str], sim_th: float = 0.8) -> dict[str, str]:
    """video_id -> session id. A video with an empty title is its own session."""
    ids = sorted(titles)
    norm = {v: normalize_title(titles[v] or "") for v in ids}
    nums = {v: work_numbers(titles[v] or "") for v in ids}
    parent = {v: v for v in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        if not norm[a]:
            continue
        for b in ids[i + 1:]:
            if not norm[b] or nums[a] != nums[b]:
                continue
            ca, cb = _composer_set(norm[a]), _composer_set(norm[b])
            if ca and cb and ca.isdisjoint(cb):   # explicitly different composers -> don't merge
                continue
            if difflib.SequenceMatcher(None, norm[a], norm[b]).ratio() >= sim_th:
                parent[find(a)] = find(b)
    return {v: find(v) for v in ids}


def load_merge_overrides(path: str) -> dict[str, list[set[str]]]:
    """Manual merge list CSV (conductor,video_id,group,...) -> groups per conductor.

    Same-performance duplicates that title-based merging misses (differing
    year/opus-number notation, mistyped titles) get confirmed via audio
    comparison and then force-merged here
    (2026-08-13 audio fingerprint audit). Returns an empty dict if the file is absent."""
    p = _Path(path)
    if not p.exists():
        return {}
    groups: dict[tuple[str, str], set[str]] = {}
    for r in _csv.DictReader(p.open()):
        groups.setdefault((r["conductor"], r["group"]), set()).add(r["video_id"])
    out: dict[str, list[set[str]]] = {}
    for (c, _g), vids in groups.items():
        out.setdefault(c, []).append(vids)
    return out


def apply_merge_overrides(sess: dict[str, str],
                          groups: list[set[str]]) -> dict[str, str]:
    """Apply forced merges on top of group_sessions results — unifies the sessions
    of video_ids within a group into their union (transitive: any existing
    session that a group member belongs to gets merged wholesale)."""
    out = dict(sess)
    for g in groups:
        target_sessions = {out[v] for v in g if v in out}
        if len(target_sessions) <= 1:
            continue
        rep = min(target_sessions)
        for v, s in out.items():
            if s in target_sessions:
                out[v] = rep
    return out


def load_duplicate_clusters(path: str) -> dict[str, list[set[str]]]:
    """Audio cross-correlation verdict=same pairs (2026-08-14 audit) -> per-conductor
    clusters of true-duplicate video_ids (re-uploads, excerpts; transitive union-find).

    The session merging above is grouping to prevent fold leakage (both videos
    stay in the dataset). This is separate — its purpose is to pick which
    video_id to keep, among duplicates of the same actual footage uploaded
    under multiple video_ids, so windows don't get double-counted; the rest
    are dropped. Returns an empty dict if the file is absent."""
    p = _Path(path)
    if not p.exists():
        return {}
    pairs: dict[str, list[tuple[str, str]]] = {}
    for r in _csv.DictReader(p.open()):
        if r.get("verdict") != "same":
            continue
        pairs.setdefault(r["conductor"], []).append((r["vid_a"], r["vid_b"]))
    out: dict[str, list[set[str]]] = {}
    for c, ps in pairs.items():
        parent: dict[str, str] = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in ps:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        clusters: dict[str, set[str]] = {}
        for v in parent:
            clusters.setdefault(find(v), set()).add(v)
        out[c] = [g for g in clusters.values() if len(g) > 1]
    return out

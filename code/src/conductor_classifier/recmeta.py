"""Recording metadata -- title-based recording-year extraction and metadata CSV merging.

Year is a covariate for separating out age/generation/video-quality confounds.
It's auto-filled only when the title makes it unambiguous; otherwise it's left
blank and deferred to manual entry (year_manual).
"""
from __future__ import annotations

import re

# Range of years a live-performance video could plausibly exist in -- anything
# outside this (e.g. a composer's birth/death year) is ignored.
_YEAR_MIN, _YEAR_MAX = 1930, 2029
_YEAR_RE = re.compile(r"(?<!\d)(19[3-9]\d|20[0-2]\d)(?!\d)")
_VIDEO_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def recording_year(title: str) -> int | None:
    """Return the recording year from the title only when exactly one is unambiguous.

    - Years outside the valid range (composer birth/death years, etc.) never
      match in the first place.
    - A year immediately adjacent to the word "remaster" (separated only by
      whitespace) is excluded, since that's usually the remaster release
      year, not the recording year. If a delimiter separates them, e.g.
      "1989 - REMASTERED", the year is accepted as the recording year.
    - If multiple distinct years remain, it's ambiguous -> None.
    """
    years: set[int] = set()
    for m in _YEAR_RE.finditer(title):
        before, after = title[:m.start()], title[m.end():]
        if re.search(r"remaster\w*\s*$", before, re.IGNORECASE):
            continue
        if re.match(r"\s*remaster", after, re.IGNORECASE):
            continue
        years.add(int(m.group(1)))
    if len(years) != 1:
        return None
    return years.pop()


def parse_video_id(filename: str) -> str | None:
    """Extract the 11-char YouTube ID from a 'Title [videoID].mp4' filename. If multiple bracketed groups exist, use the last.

    Also accepts a filename whose entire stem is the ID with no brackets, but
    excludes stems that look like ordinary words (all lowercase, or only the
    first letter capitalized), treating those as titles instead.
    """
    ids = _VIDEO_ID_RE.findall(filename)
    if ids:
        return ids[-1]
    stem = filename.rsplit(".", 1)[0]
    if (re.fullmatch(r"[A-Za-z0-9_-]{11}", stem)
            and not re.fullmatch(r"[A-Z]?[a-z]+", stem)):
        return stem
    return None


_BUCKET_RE = re.compile(r"^(\d{4})s(?:~(\d{2,4})s)?$")


def _bucket_range(name: str) -> tuple[int, int] | None:
    """Era folder name -> (start, end) year. '1960s~70s'=1960-1979, '1990s'=1990-1999,
    '2000s~20s'=2000-2029 (the two-digit end follows the start century, unless
    it's smaller, in which case it rolls to the next century)."""
    m = _BUCKET_RE.match(name)
    if not m:
        return None
    start = int(m.group(1))
    if m.group(2) is None:
        return start, start + 9
    e = int(m.group(2))
    if e < 100:
        e += (start // 100) * 100
        if e < start:
            e += 100
    return start, e + 9


def era_bucket(year: int, folder_names: list[str]) -> str | None:
    """Match a year to an era folder name. Returns None if no match."""
    for name in folder_names:
        rng = _bucket_range(name)
        if rng and rng[0] <= year <= rng[1]:
            return name
    return None


_DECADE_RE = re.compile(r"^(\d{2}|\d{4})s$", re.IGNORECASE)


def normalize_year_input(s: str) -> str | None:
    """Normalize a manual year input -- '1985'/'1980s' ('80s' -> '1980s'). An empty value returns '' (cancel).

    The allowed range matches the valid recording-year range (1930-2029).
    Anything outside that, or malformed, returns None (rejected)."""
    s = s.strip()
    if not s:
        return ""
    if s.isdigit() and len(s) == 4:
        y = int(s)
        return s if _YEAR_MIN <= y <= _YEAR_MAX else None
    m = _DECADE_RE.match(s)
    if m:
        d = int(m.group(1))
        if len(m.group(1)) == 2:  # '80s' -- 30 or above means 1900s, below means 2000s
            d += 1900 if d >= _YEAR_MIN % 100 else 2000
        if d % 10 == 0 and _YEAR_MIN <= d and d + 9 <= _YEAR_MAX:
            return f"{d}s"
    return None


def year_span(s: str) -> tuple[int, int] | None:
    """Manual input -> (min, max) year. '1985'->(1985,1985), '1980s'->(1980,1989).

    An analysis that needs an exact year can simply exclude entries with a
    span width > 0 (decades)."""
    norm = normalize_year_input(s)
    if not norm:
        return None
    if norm.endswith("s"):
        d = int(norm[:-1])
        return d, d + 9
    return int(norm), int(norm)


def era_bucket_str(s: str, folder_names: list[str]) -> str | None:
    """Match a manual input (year/decade) to an era folder -- only when the entire span fits within one bucket."""
    span = year_span(s)
    if span is None:
        return None
    for name in folder_names:
        rng = _bucket_range(name)
        if rng and rng[0] <= span[0] and span[1] <= rng[1]:
            return name
    return None


def merge_meta(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    """When regenerating, preserve the existing CSV's manual entries (year_manual) keyed by video_id."""
    manual = {r["video_id"]: r.get("year_manual", "") for r in old_rows}
    out = []
    for r in new_rows:
        r = dict(r)
        if manual.get(r["video_id"]):
            r["year_manual"] = manual[r["video_id"]]
        out.append(r)
    return out

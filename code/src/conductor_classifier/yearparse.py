"""Extracts the **recording year** from a YouTube title/description (distinct from upload/remaster year).

The era-confound analysis needs "when was this performed," but the values we
can extract automatically (upload year, numbers in the title) are often the
remaster/re-upload year instead. So we:
  1) prioritize a year adjacent to **recording context** words like
     recorded/live/aufnahme (confidence=context)
  2) also treat date formats (12 May 1976, 1976-05-12) as context
  3) fall back to a year in parentheses etc. as a weak candidate
     (confidence=weak) when there's no context
  4) discard years after the upload year, before 1940, or adjacent to an
     opus/catalog number (Op./No.)
"""
import datetime
import re
from dataclasses import dataclass

MIN_YEAR = 1940

_CONTEXT = (r"recorded|recording|live|concert|filmed|taped|aufnahme|aufgenommen|"
            r"enregistr|registrazione|grabado|실황|녹음|녹화|공연|연주")
_YEAR = r"(19[4-9]\d|20[0-2]\d)"
_MONTHS = (r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec")

_RE_CONTEXT = re.compile(rf"(?:{_CONTEXT})[^.\n]{{0,60}}?{_YEAR}", re.I)
_RE_DATE_DMY = re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})[a-z]*\.?,?\s+{_YEAR}\b", re.I)
_RE_DATE_MDY = re.compile(rf"\b(?:{_MONTHS})[a-z]*\.?\s+\d{{1,2}},?\s+{_YEAR}\b", re.I)
_RE_ISO = re.compile(rf"\b{_YEAR}-\d{{2}}-\d{{2}}\b")
_RE_PAREN = re.compile(rf"[(\[]\s*{_YEAR}\s*[)\]]")
_RE_OPUS = re.compile(rf"(?:op\.?|no\.?|k\.?|bwv|d\.)\s*{_YEAR}", re.I)


@dataclass
class YearGuess:
    year: int | None
    confidence: str          # "context" | "weak" | ""
    evidence: str            # a snippet of the matching text (so a human can verify it)


def _valid(y: int, upload_year: int | None) -> bool:
    cap = upload_year or datetime.date.today().year
    return MIN_YEAR <= y <= cap


def _snippet(text: str, m: re.Match) -> str:
    a, b = max(0, m.start() - 25), min(len(text), m.end() + 25)
    return " ".join(text[a:b].split())[:80]


def recording_year(text: str, upload_year: int | None = None) -> YearGuess:
    """Estimate the recording year from title+description text. year=None if not found."""
    if not text:
        return YearGuess(None, "", "")
    opus_spans = [m.span() for m in _RE_OPUS.finditer(text)]

    def blocked(m):   # a number adjacent to an opus/catalog number is not a year
        return any(a <= m.start() < b for a, b in opus_spans)

    for label, regexes in (("context", (_RE_CONTEXT, _RE_DATE_DMY, _RE_DATE_MDY, _RE_ISO)),
                           ("weak", (_RE_PAREN,))):
        hits = []
        for rx in regexes:
            for m in rx.finditer(text):
                if blocked(m):
                    continue
                y = int(m.group(1))
                if _valid(y, upload_year):
                    hits.append((y, _snippet(text, m)))
        if hits:
            y, ev = min(hits, key=lambda h: h[0])   # if multiple, take the earliest year
            return YearGuess(y, label, ev)
    return YearGuess(None, "", "")

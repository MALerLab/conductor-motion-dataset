"""Repertoire mapping (conf/repertoire.yaml) — selects the matched subset of shared pieces.

Piece-conductor correlation is the single biggest confound (experimental
design step 5), so this provides a filter that keeps only videos of pieces
both conductors have conducted. doc (documentary/compilation) videos have a
meaningless piece label and are excluded from the count — as a result, any
piece that effectively belongs to only one side gets dropped entirely.
"""
import yaml

_EXCLUDE_TYPES = {"doc"}
_EXCLUDE_PIECES = {"misc"}


def load_repertoire(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def matched_video_ids(rep: dict) -> set[str]:
    """Set of video_id for pieces shared by two or more conductors (doc/misc excluded)."""
    piece_owners: dict[str, set[str]] = {}
    for conductor, vids in rep.items():
        for _vid, m in vids.items():
            if m["type"] in _EXCLUDE_TYPES or m["piece"] in _EXCLUDE_PIECES:
                continue
            piece_owners.setdefault(m["piece"], set()).add(conductor)
    shared = {p for p, owners in piece_owners.items() if len(owners) >= 2}
    return {vid for vids in rep.values() for vid, m in vids.items()
            if m["piece"] in shared and m["type"] not in _EXCLUDE_TYPES}

"""Conductor registry — the single source of truth for the crawler, face DB, and orchestrator.

Search queries, aliases, filters, and reference-face paths are all managed
from one place: conf/conductors.yaml.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Conductor:
    name: str                       # folder name + classification label (data/raw/<name>)
    aliases: list[str] = field(default_factory=list)   # used for title matching
    queries: list[str] = field(default_factory=list)   # explicit search queries (precise)
    exclude_terms: list[str] = field(default_factory=list)  # keywords that exclude a title
    min_duration_s: int = 120
    max_duration_s: int = 4800
    reference_faces: str = ""       # data/reference_faces/<name> (for building the face DB)

    def __post_init__(self) -> None:
        if not self.aliases:
            self.aliases = [self.name]
        if not self.reference_faces:
            self.reference_faces = f"data/reference_faces/{self.name}"


def load_conductors(path: str | Path, only: str | None = None) -> list[Conductor]:
    """conf/conductors.yaml -> list of Conductor. If `only` is given, keep just that name."""
    import yaml  # lazy import (collect extra)

    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top level must be a list of conductors")
    conductors = [Conductor(**entry) for entry in raw]
    if only:
        conductors = [c for c in conductors if c.name == only]
        if not conductors:
            raise ValueError(f"conductor '{only}' not found in {path}")
    return conductors

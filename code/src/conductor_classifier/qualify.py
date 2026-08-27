"""Overall shot-level verdict: frontal + identity + arm/hand crop -> ACCEPT / HOLD / REJECT.

Each of the three criteria is scored into an accept/hold/reject band, then combined:
  - If frontal or identity lands in the reject band  -> REJECT (strongest)
  - Else if any criterion lands in the hold band      -> HOLD (send to human review)
  - If all three are accept                           -> ACCEPT
Crop is an auxiliary signal -- even in its reject band it only escalates to
HOLD (it has no authority to REJECT on its own). Rationale: in a regression
against manual A/B curation, the wrist check alone only reproduced half of the
human-excluded set (AUC~0.65).

The identity score (id_score) can be None when face detection is insufficient,
and in that case it's treated as "insufficient evidence" and routed to HOLD
rather than REJECT (non-destructive -- protects older SD-quality footage).
"""
from dataclasses import dataclass, field

from .config import PipelineConfig

ACCEPT = "accept"
HOLD = "hold"
REJECT = "reject"


@dataclass
class ShotVerdict:
    frontal_frac: float
    id_score: float | None
    crop_frac: float
    n_frontal: int = 0
    decision: str = HOLD
    reasons: list[str] = field(default_factory=list)


def _band(value: float, accept: float, hold: float) -> str:
    """value >= accept -> accept; >= hold -> hold; below -> reject."""
    if value >= accept:
        return ACCEPT
    if value >= hold:
        return HOLD
    return REJECT


def decide(frontal_frac: float, id_score: float | None, crop_frac: float,
           cfg: PipelineConfig, n_frontal: int | None = None,
           identity_enabled: bool = True) -> ShotVerdict:
    """Combine the three criteria into a final verdict. Records each criterion's status in reasons.

    If identity_enabled=False (the fast path with no face DB), the identity
    criterion is skipped.
    """
    reasons: list[str] = []
    bands: list[str] = []

    # 1) Frontal -- fraction band + minimum count (too few frontal frames is treated as a chance false positive, so reject).
    fb = _band(frontal_frac, cfg.frontal_accept_fraction, cfg.frontal_hold_fraction)
    if n_frontal is not None and n_frontal < cfg.frontal_min_count:
        fb = REJECT
    reasons.append(f"frontal={frontal_frac:.2f}({fb})")
    bands.append(fb)

    # 2) Identity -- None (insufficient evidence) means HOLD. Skipped if disabled.
    if not identity_enabled:
        reasons.append("id=off")
    elif id_score is None:
        bands.append(HOLD)
        reasons.append("id=none(hold)")
    else:
        ib = _band(id_score, cfg.id_accept, cfg.id_reject)
        reasons.append(f"id={id_score:.2f}({ib})")
        bands.append(ib)

    # 3) Arm/hand crop -- demoted to an auxiliary signal (no authority to REJECT alone).
    #    Empirically, in the A/B curation regression, even the worst-case wrist
    #    check only caught 52% of the human-excluded set (B) (AUC~0.65) --
    #    human exclusion reasons aren't limited to wrist framing.
    #    Automatic REJECT risks losing data -> even worst case escalates only
    #    to HOLD (human review).
    cb = _band(crop_frac, cfg.crop_ok_min_fraction, cfg.crop_hold_min_fraction)
    reasons.append(f"crop={crop_frac:.2f}({cb})")
    bands.append(HOLD if cb == REJECT else cb)

    if REJECT in bands:
        decision = REJECT
    elif HOLD in bands:
        decision = HOLD
    else:
        decision = ACCEPT

    return ShotVerdict(
        frontal_frac=frontal_frac, id_score=id_score, crop_frac=crop_frac,
        n_frontal=n_frontal or 0, decision=decision, reasons=reasons,
    )

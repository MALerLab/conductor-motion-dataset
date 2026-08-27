"""Gentle crawl-pacing policy — pure functions.

Conservative defaults reflecting an observed bot block on 2026-07-20 (48
videos in a row at 3-8s intervals triggered "Sign in to confirm you're not
a bot"). Designed to look more like a human viewing pattern: a random wait
between videos (tens of seconds) + a long rest after each batch (tens of
minutes) + a rolling 24h total cap. If blocked again, back off in stepped
stages. Everything here is a pure function, so tests inject the clock/RNG.
"""
from dataclasses import dataclass


@dataclass
class PacingPolicy:
    sleep_min: float = 20.0            # minimum wait between videos (s)
    sleep_max: float = 45.0            # maximum wait between videos (s)
    batch_size: int = 20               # take a long rest after this many downloads
    rest_min_s: float = 15 * 60.0      # minimum batch rest
    rest_max_s: float = 30 * 60.0      # maximum batch rest
    daily_cap: int = 250               # download cap within a rolling 24h window
    block_backoffs_s: tuple = (3 * 3600.0, 6 * 3600.0, 12 * 3600.0)
    min_free_gb: float = 60.0          # stop if free disk drops below this


def inter_video_sleep(pol: PacingPolicy, rng) -> float:
    return rng.uniform(pol.sleep_min, pol.sleep_max)


def is_batch_end(pol: PacingPolicy, n_done: int) -> bool:
    return n_done > 0 and n_done % pol.batch_size == 0


def rest_duration(pol: PacingPolicy, rng) -> float:
    return rng.uniform(pol.rest_min_s, pol.rest_max_s)


def block_backoff_s(pol: PacingPolicy, n_consecutive_blocks: int) -> float:
    idx = min(n_consecutive_blocks, len(pol.block_backoffs_s)) - 1
    return pol.block_backoffs_s[idx]


def daily_cap_wait_s(pol: PacingPolicy, timestamps: list, now: float) -> float:
    window = sorted(t for t in timestamps if t > now - 86400)
    if len(window) < pol.daily_cap:
        return 0.0
    # cap reached: a slot opens only once the cap-th most recent timestamp in the window ages out of it.
    return window[len(window) - pol.daily_cap] + 86400 - now


def disk_ok(pol: PacingPolicy, free_bytes: int) -> bool:
    return free_bytes >= pol.min_free_gb * 2**30

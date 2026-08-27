"""Per-joint-set adjacency matrix -- prior knowledge that tells the GCN the body's connectivity structure in advance.

Built in three layers:
  1) **Anatomical tree** -- the parent-child links from channels.parents() (same as the bone-normalization tree).
  2) **Symmetry correction** -- the tree implementation roots at the left
     shoulder, so the nose ends up attached only to the left shoulder,
     making it left-right asymmetric. The nose is connected to **both**
     shoulders so it can act like a shoulder-midpoint.
  3) **Coordination edges (optional)** -- direct connections between the
     left/right wrist, elbow, and hand. It's reasonable to treat bimanual
     coordination in conducting as a single hop (currently left-wrist ->
     right-wrist is 4 hops), but since this operates at a different level of
     interpretation, it's off by default and enabled only for experiments.

After adding self-loops, apply **symmetric normalization** (D^-1/2 A D^-1/2, the standard ST-GCN convention).
"""
import numpy as np

from . import keypoints as kp
from .channels import JOINT_SETS, parents

# Left-right coordination -- treats patterns formed jointly by both hands as a single hop
_COORD_PAIRS = [
    (kp.LEFT_WRIST, kp.RIGHT_WRIST),
    (kp.LEFT_ELBOW, kp.RIGHT_ELBOW),
    (91, 112),                        # both-hand roots
]


def edges(joint_set: str, coord: bool = False) -> list[tuple[int, int]]:
    """Undirected edge list in within-set indices (deduplicated)."""
    js = JOINT_SETS[joint_set]
    pos = {j: i for i, j in enumerate(js)}
    out: set[tuple[int, int]] = set()

    def add(a: int, b: int) -> None:
        if a in pos and b in pos and a != b:
            i, j = pos[a], pos[b]
            out.add((min(i, j), max(i, j)))

    for c, p in enumerate(parents(joint_set)):     # 1) anatomical tree
        if c != p:
            out.add((min(c, p), max(c, p)))
    add(kp.NOSE, kp.RIGHT_SHOULDER)                # 2) symmetry correction
    if coord:                                      # 3) coordination
        for a, b in _COORD_PAIRS:
            add(a, b)
    return sorted(out)


def adjacency(joint_set: str, coord: bool = False) -> np.ndarray:
    """(V,V) symmetrically-normalized adjacency matrix. Raises KeyError for an unknown set."""
    v = len(JOINT_SETS[joint_set])
    A = np.eye(v, dtype=np.float32)
    for i, j in edges(joint_set, coord):
        A[i, j] = A[j, i] = 1.0
    dinv = 1.0 / np.sqrt(A.sum(axis=1))
    return (A * dinv[:, None] * dinv[None, :]).astype(np.float32)


# Coordination-pair graph -- nodes are left-right corresponding pairs (shoulder, elbow, wrist, hip + 21 hand points), following left-side anatomy.
_PAIR_EDGES = [(0, 1), (1, 2), (0, 3), (2, 4)] + \
              [(4 + p, 4 + c) for p, c in
               [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
                (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]]


def pair_adjacency() -> np.ndarray:
    """(25,25) symmetrically-normalized adjacency matrix -- matches build_pair_channels' node ordering."""
    v = 25
    A = np.eye(v, dtype=np.float32)
    for i, j in _PAIR_EDGES:
        A[i, j] = A[j, i] = 1.0
    dinv = 1.0 / np.sqrt(A.sum(axis=1))
    return (A * dinv[:, None] * dinv[None, :]).astype(np.float32)

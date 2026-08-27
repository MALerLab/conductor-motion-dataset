"""Repertoire cross-check recomputation -- fixed rules, reproducible (source script for the ISMIR LBD numbers).

Using the preds (per-recording probabilities) from a train_pilot result JSON and
the titles in recordings_meta.csv, computes:
1) accuracy on the composer-match subset (common to all 5: Beethoven, Mahler, Tchaikovsky)
2) recording/session-level correctness on the same piece (Beethoven Symphonies 6 and 7)
The matching rules are fixed in this file -- paper numbers must always be
refreshed from this script's output (the ad-hoc calculation practice was
retired in the 2026-08-13 reproducibility audit).

Usage:
    PYTHONPATH=src python scripts/verify_repertoire.py data/experiments/V5s_both.json
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")
from conductor_classifier.sessions import (apply_merge_overrides, group_sessions,
                                           load_merge_overrides)

META = Path("data/recordings_meta.csv")

# ---- matching rules (fixed) ----
COMPOSERS = {
    "beethoven": r"beethoven|pastoral",
    "mahler": r"mahler",
    "tchaikovsky": r"tchaikovsky|tchaikowsky|chaikovsky|čajkovskij",
}
PIECES = {
    "Beethoven Sym 6": r"(?=.*(beethoven|pastoral))(?=.*(no\.?\s*6|nr\.?\s*6|symphony\s*6|pastoral))",
    "Beethoven Sym 7": r"(?=.*beethoven)(?=.*(no\.?\s*7|nr\.?\s*7|symphony\s*7))",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", help="train_pilot result JSON (must include preds)")
    ap.add_argument("--session-merges", default="",
                    help="Manual session-merge CSV -- must match the one used in the experiment")
    args = ap.parse_args()
    merge_ov = load_merge_overrides(args.session_merges) if args.session_merges else {}

    d = json.loads(Path(args.result_json).read_text())
    preds = d["preds"]
    classes = sorted({k.split("|")[0] for k in preds})
    titles = {(r["conductor"], r["video_id"]): (r["title"] or "")
              for r in csv.DictReader(META.open())}

    rows = []   # (conductor, vid, title, pred, correct)
    for k, v in preds.items():
        c, vid = k.split("|")
        t = titles.get((c, vid), "")
        pred = classes[max(range(len(classes)), key=lambda i: v["p"][i])]
        rows.append((c, vid, t, pred, pred == classes[v["y"]]))

    print(f"[{d['tag']}] {len(rows)} recordings · overall accuracy {sum(r[4] for r in rows)/len(rows):.3f}")

    # 1) composer match
    sub = [r for r in rows
           if any(re.search(p, r[2].lower()) for p in COMPOSERS.values())]
    cc = Counter(r[0] for r in sub)
    acc = sum(r[4] for r in sub) / len(sub)
    print(f"\ncomposer match (3 common composers): {len(sub)} recordings · accuracy {acc:.3f} "
          f"· chance {max(cc.values())/len(sub):.3f} · distribution {dict(cc)}")

    # 2) same-piece -- recording/session level
    for name, pat in PIECES.items():
        piece_rows = [r for r in rows if re.search(pat, r[2].lower())]
        n_ok = sum(r[4] for r in piece_rows)
        by_sess = defaultdict(list)
        for c in {r[0] for r in piece_rows}:
            tset = {r[1]: r[2] for r in piece_rows if r[0] == c}
            g = group_sessions(tset)
            if c in merge_ov:
                g = apply_merge_overrides(g, merge_ov[c])
            for vid, s in g.items():
                by_sess[(c, s)] += [r for r in piece_rows if r[1] == vid]
        sess_ok = sum(all(r[4] for r in items) for items in by_sess.values())
        print(f"{name}: recordings {n_ok}/{len(piece_rows)} · sessions {sess_ok}/{len(by_sess)}")
        for r in piece_rows:
            if not r[4]:
                print(f"  wrong: {r[0]} {r[1]} -> {r[3]} | {r[2][:60]}")


if __name__ == "__main__":
    main()

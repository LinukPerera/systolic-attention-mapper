"""
One-off merge script for the larger-fresh-ceiling-budget follow-up
(docs/NEXT_INSTRUCTIONS.md's own §9-flagged next step, run 2026-09-03 after the
user asked to "try to close the gap"): combines the original exp6.csv's
baselines + zero_shot_transfer rows (fully deterministic, unchanged, not
re-simulated -- see run_exp6.py's --roles flag) with exp6_n15.csv's
fresh_ceiling rows (the new, n_iterations=15 run) into one CSV analyze_exp6.py
can read directly. Deliberately does NOT include the original exp6.csv's
n_iterations=6 fresh_ceiling rows -- mixing two different budgets under one
"fresh_ceiling" label would silently let analyze_exp6.py take the best of both
pools per seed, which is not what this follow-up tests (it isolates budget
alone, mirroring Experiment 4's own budget-isolation reasoning).
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

THIS_DIR = Path(__file__).parent

original = pd.read_csv(THIS_DIR / "results" / "raw" / "exp6.csv")
n15 = pd.read_csv(THIS_DIR / "results" / "raw" / "exp6_n15.csv")

assert set(n15["optimizer"].unique()) == {"fresh_ceiling"}, (
    f"expected exp6_n15.csv to contain only fresh_ceiling rows, got {sorted(n15['optimizer'].unique())}"
)

kept = original[original["optimizer"] != "fresh_ceiling"]
merged = pd.concat([kept, n15], ignore_index=True)
out_path = THIS_DIR / "results" / "raw" / "exp6_n15_merged.csv"
merged.to_csv(out_path, index=False)
print(f"kept {len(kept)} baseline/transfer rows from exp6.csv + {len(n15)} new fresh_ceiling (n=15) rows "
      f"-> {out_path} ({len(merged)} total)")

#!/usr/bin/env python3
"""Plot comparison charts from saved Pointax MVP runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MVP_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, MVP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pointax_mvp.plotting import plot_learning_curve, plot_main_comparison
from pointax_mvp.utils import ensure_dir


def resolve_run_dir(results_root: Path, run_name: str) -> Path:
    latest = results_root / run_name / "latest"
    if latest.exists():
        return latest.resolve()
    candidates = sorted((results_root / run_name).glob("*"))
    if not candidates:
        raise FileNotFoundError(f"No runs found for {run_name}.")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    args = parser.parse_args()
    results_root = Path(args.results_root)
    no_curriculum = resolve_run_dir(results_root, "no_curriculum")
    manual_curriculum = resolve_run_dir(results_root, "manual_curriculum")
    frames = []
    learning_frames = []
    for run_dir, run_name in ((no_curriculum, "no_curriculum"), (manual_curriculum, "manual_curriculum")):
        frame = pd.read_csv(run_dir / "eval_summary.csv")
        frame["run_name"] = run_name
        frames.append(frame.groupby("group", as_index=False).tail(1))
        learning = pd.read_csv(run_dir / "learning_curve.csv")
        learning["run_name"] = run_name
        learning_frames.append(learning)
    output_dir = ensure_dir(results_root / "plots")
    plot_main_comparison(pd.concat(frames, ignore_index=True).to_dict("records"), output_dir / "main_comparison.png")
    plot_learning_curve(pd.concat(learning_frames, ignore_index=True), output_dir / "learning_curve.png")
    print(output_dir)


if __name__ == "__main__":
    main()

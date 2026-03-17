#!/usr/bin/env python3
"""Run both curriculum conditions and generate comparison plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MVP_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, MVP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pointax_mvp.training import train
from pointax_mvp.utils import TrainConfig, apply_overrides, load_yaml_config
from plot_results import build_comparison_plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", default="mvp_pointax_huvfa/results")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    payload = load_yaml_config(args.config)
    payload["results_root"] = args.results_root
    payload = apply_overrides(payload, args.override)

    manual_payload = dict(payload)
    manual_payload["run_name"] = "manual_curriculum"
    manual_payload["training_mode"] = "manual_curriculum"

    no_payload = dict(payload)
    no_payload["run_name"] = "no_curriculum"
    no_payload["training_mode"] = "no_curriculum"

    manual_run = train(TrainConfig.from_mapping(manual_payload))
    no_run = train(TrainConfig.from_mapping(no_payload))
    plots_dir = build_comparison_plots(Path(args.results_root))

    print(f"manual_run={manual_run}")
    print(f"no_curriculum_run={no_run}")
    print(f"plots_dir={plots_dir}")


if __name__ == "__main__":
    main()

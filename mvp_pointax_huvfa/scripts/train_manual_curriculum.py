#!/usr/bin/env python3
"""Train the Pointax MVP under the manual-curriculum condition."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    payload = load_yaml_config(args.config)
    payload["training_mode"] = "manual_curriculum"
    if args.results_root:
        payload["results_root"] = args.results_root
    payload = apply_overrides(payload, args.override)
    config = TrainConfig.from_mapping(payload)
    run_dir = train(config)
    print(run_dir)


if __name__ == "__main__":
    main()

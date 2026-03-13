#!/usr/bin/env python3
"""Generate value heatmaps for a saved Pointax MVP run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flax.serialization import from_bytes
import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
MVP_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, MVP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pointax_mvp.plotting import make_value_heatmaps
from pointax_mvp.task_library import build_eval_scenarios
from pointax_mvp.training import build_model
from pointax_mvp.utils import TrainConfig, load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = TrainConfig.from_mapping(load_yaml_config(run_dir / "config.json"))
    model = build_model(config)
    params_template = model.init(
        jax.random.PRNGKey(config.seed),
        jnp.zeros((1, 8), dtype=jnp.float32),
        jnp.zeros((1, 2), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    params = from_bytes(params_template, (run_dir / "checkpoints/latest.msgpack").read_bytes())
    scenario = build_eval_scenarios()["composed_ood"][0]
    output_dir = run_dir / "plots"
    make_value_heatmaps(
        model=model,
        params=params,
        scenario=scenario,
        action_scale=config.action_scale,
        resolution=config.heatmap_resolution,
        output_dir=output_dir,
    )
    print(output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a saved Pointax MVP run on the OOD suites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import flax
from flax.serialization import from_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
MVP_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, MVP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pointax_mvp.evaluation import evaluate_scenarios
from pointax_mvp.models import SharedQNetwork
from pointax_mvp.task_library import build_eval_scenarios
from pointax_mvp.training import build_model
from pointax_mvp.utils import LEARNED_OPTIONS, TrainConfig, load_yaml_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = TrainConfig.from_mapping(load_yaml_config(run_dir / "config.json"))
    model = build_model(config)
    sample_params = model.init(
        jax.random.PRNGKey(config.seed),
        jnp.zeros((1, 8), dtype=jnp.float32),
        jnp.zeros((1, 2), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
    )
    params = from_bytes(sample_params, (run_dir / "checkpoints/latest.msgpack").read_bytes())
    summary = evaluate_scenarios(
        model=model,
        params=params,
        eval_sets=build_eval_scenarios(),
        seed=config.seed + 12345,
        action_scale=config.action_scale,
        episodes_per_scenario=config.num_eval_episodes,
    )
    save_json(run_dir / "ood_eval.json", summary)
    print(run_dir / "ood_eval.json")


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    main()

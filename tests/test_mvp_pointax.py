"""Tests for the Pointax H-UVFA MVP subproject."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MVP_ROOT = REPO_ROOT / "mvp_pointax_huvfa"
for path in (REPO_ROOT, MVP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("yaml")

from pointax_mvp.discrete_actions import discrete_to_continuous
from pointax_mvp.envs import build_base_env
from pointax_mvp.models import SharedQNetwork, q_values_for_all_options
from pointax_mvp.replay import ReplayBuffer, TransitionBatch
from pointax_mvp.task_library import build_eval_scenarios, build_training_scenarios, make_t_junction
from pointax_mvp.teacher import branching_cells, infer_option_for_path_index, shortest_path
import pointax_mvp.training as training_module
from pointax_mvp.plotting import make_value_heatmaps
from pointax_mvp.training import init_train_bundle
from pointax_mvp.utils import NoiseSpec, OPTION_TO_ID, ScenarioSpec, TrainConfig


def _count_marker(layout, marker):
    return sum(cell == marker for row in layout for cell in row)


def test_training_scenarios_have_single_reset_and_goal():
    for scenario in build_training_scenarios()["A1"]:
        assert _count_marker(scenario.maze_layout, "R") == 1
        assert _count_marker(scenario.maze_layout, "G") == 1


def test_eval_and_train_scenario_names_are_disjoint():
    train_names = {scenario.name for stage in build_training_scenarios().values() for scenario in stage}
    eval_names = {scenario.name for group in build_eval_scenarios().values() for scenario in group}
    assert train_names.isdisjoint(eval_names)


def test_diagonal_action_is_normalized():
    action = discrete_to_continuous(5, scale=0.6)
    assert pytest.approx(float(action[0])) == pytest.approx(float(action[1]))
    assert pytest.approx(float((action ** 2).sum() ** 0.5), rel=1e-5) == 0.6


def test_junction_option_label_is_used_near_branch():
    scenario = make_t_junction("test_junction", "A3", "train", approach=5, left_branch=4, right_branch=4)
    maze = jnp.asarray([[1 if cell == 1 else 0 for cell in row] for row in scenario.maze_layout]) != 1
    path = shortest_path(maze, scenario.start_cell, scenario.goal_cell)
    branch_cells = branching_cells(maze)
    labels = [infer_option_for_path_index(path, idx, branch_cells) for idx in range(len(path) - 1)]
    assert OPTION_TO_ID["JUNCTION_RESOLVE"] in labels
    assert OPTION_TO_ID["SEGMENT_FOLLOW"] in labels


def test_model_forward_shapes():
    model = SharedQNetwork(hidden_dims=(32, 32), embedding_dim=16, num_options=2, num_actions=9)
    params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((3, 8), dtype=jnp.float32),
        jnp.zeros((3, 2), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    outputs = model.apply(
        params,
        jnp.zeros((3, 8), dtype=jnp.float32),
        jnp.zeros((3, 2), dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.int32),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    assert outputs["q"].shape == (3,)
    full = q_values_for_all_options(model, params, jnp.zeros((3, 8), dtype=jnp.float32), jnp.zeros((3, 2), dtype=jnp.float32))
    assert full.shape == (3, 2, 9)


def test_replay_future_relabeling():
    buffer = ReplayBuffer()
    scenario = ScenarioSpec(
        name="tiny",
        family="A1",
        stage="A1",
        split="train",
        maze_layout=((1, 1, 1), (1, "R", "G"), (1, 1, 1)),
        start_cell=(1, 1),
        goal_cell=(1, 2),
        noise=NoiseSpec(),
    )

    class Transition:
        def __init__(self, obs, next_obs):
            self.obs = obs
            self.next_obs = next_obs
            self.goal_xy = next_obs[:2]
            self.option_id = 0
            self.action_id = 3
            self.reward = 1.0
            self.done = False
            self.success = 0.0
            self.wall_contact = 0.0

    buffer.add_trajectory(
        scenario,
        [
            Transition(jnp.zeros(8), jnp.ones(8)),
            Transition(jnp.ones(8), 2 * jnp.ones(8)),
        ],
    )
    buffer.relabel_future_observations(np.random.default_rng(0))
    batch = buffer.batch([0, 1])
    assert batch.future_obs.shape == (2, 8)


def test_mini_pipeline_smoke(tmp_path):
    scenario = build_training_scenarios()["A1"][0]
    bundle_env = build_base_env(scenario, max_steps=12, action_scale=0.6)
    obs, _ = bundle_env.wrapper.reset(jax.random.PRNGKey(0))
    next_obs, reward, done, info = bundle_env.wrapper.step(3, jax.random.PRNGKey(1))
    assert obs.shape == (8,)
    assert next_obs.shape == (8,)
    assert isinstance(reward, float)

    config = TrainConfig(
        seed=3,
        hidden_dims=(32, 32),
        embedding_dim=16,
        batch_size=8,
        updates_per_stage=1,
        trajectories_per_scenario=1,
        num_eval_episodes=1,
        heatmap_resolution=12,
        checkpoint_every=1,
        eval_every=1,
        action_scale=0.6,
        max_steps_per_episode=24,
        run_name="pytest_manual",
        training_mode="manual_curriculum",
    )
    bundle = init_train_bundle(config)
    batch = TransitionBatch(
        obs=np.stack([obs, next_obs], axis=0).astype(np.float32),
        next_obs=np.stack([next_obs, next_obs], axis=0).astype(np.float32),
        goal_xy=np.stack([bundle_env.wrapper.current_goal(), bundle_env.wrapper.current_goal()], axis=0).astype(np.float32),
        option_id=np.array([0, 1], dtype=np.int32),
        action_id=np.array([3, 0], dtype=np.int32),
        reward=np.array([reward, 0.0], dtype=np.float32),
        done=np.array([float(done), 0.0], dtype=np.float32),
        stage_id=np.array([0, 0], dtype=np.int32),
        family_id=np.array([0, 0], dtype=np.int32),
        success=np.array([float(info["success"]), 0.0], dtype=np.float32),
        wall_contact=np.array([float(info["wall_contact"]), 0.0], dtype=np.float32),
        future_obs=np.stack([next_obs, obs], axis=0).astype(np.float32),
    )
    updated_bundle, metrics = training_module._train_step(bundle, batch, config)
    assert "loss" in metrics

    heatmaps = make_value_heatmaps(
        model=updated_bundle.model,
        params=updated_bundle.train_state.params,
        scenario=build_eval_scenarios()["composed_ood"][0],
        action_scale=config.action_scale,
        resolution=4,
        output_dir=tmp_path,
    )
    assert "SEGMENT_FOLLOW" in heatmaps
    assert (tmp_path / "segment_follow_heatmap.png").exists()

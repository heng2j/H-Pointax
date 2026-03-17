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
from pointax_mvp.replay import ReplayBuffer
from pointax_mvp.task_library import build_eval_scenarios, build_training_scenarios, make_t_junction
from pointax_mvp.teacher import TeacherTransition, branching_cells, infer_option_for_path_index, shortest_path
import pointax_mvp.training as training_module
from pointax_mvp.plotting import make_value_heatmaps
from pointax_mvp.training import init_train_bundle
from pointax_mvp.utils import NoiseSpec, OPTION_TO_ID, ScenarioSpec, TrainConfig, apply_overrides
from scripts.plot_results import build_comparison_plots


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


def test_wall_contact_proxy_uses_geometry():
    scenario = build_training_scenarios()["A1"][0]
    bundle = build_base_env(scenario, max_steps=12, action_scale=0.6)
    wall_center = bundle.wrapper.cell_to_world((scenario.start_cell[0], scenario.start_cell[1] - 1))
    free_center = bundle.wrapper.cell_to_world(scenario.start_cell)
    assert bundle.wrapper._touching_wall(wall_center)
    assert not bundle.wrapper._touching_wall(free_center)


def test_collect_stage_data_retries_failed_teacher_rollouts(monkeypatch):
    scenario = build_training_scenarios()["A1"][0]
    attempts = {"count": 0}

    def fake_collect_teacher_trajectory(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return []
        obs = np.zeros(8, dtype=np.float32)
        return [
            TeacherTransition(
                obs=obs,
                next_obs=obs,
                goal_xy=np.zeros(2, dtype=np.float32),
                option_id=0,
                action_id=3,
                reward=1.0,
                done=True,
                stage_id=0,
                family_id=0,
                success=1.0,
                wall_contact=0.0,
                controller_id="A_STAR_NAVIGATION",
                scenario_name=scenario.name,
            )
        ]

    monkeypatch.setattr(training_module, "collect_teacher_trajectory", fake_collect_teacher_trajectory)
    buffer = training_module.collect_stage_data(
        scenarios=[scenario],
        trajectories_per_scenario=1,
        config=TrainConfig(teacher_retry_limit=4),
        seed_offset=0,
    )
    assert len(buffer) == 1
    assert attempts["count"] == 3


def test_apply_overrides_parses_scalar_and_list_values():
    updated = apply_overrides(
        {"updates_per_stage": 10, "use_hcrl_aux": False},
        ["updates_per_stage=3", "use_hcrl_aux=true", "hidden_dims=32,64"],
    )
    assert updated["updates_per_stage"] == 3
    assert updated["use_hcrl_aux"] is True
    assert updated["hidden_dims"] == [32, 64]


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

    heatmaps = make_value_heatmaps(
        model=bundle.model,
        params=bundle.train_state.params,
        scenario=build_eval_scenarios()["composed_ood"][0],
        action_scale=config.action_scale,
        resolution=4,
        output_dir=tmp_path,
    )
    assert "SEGMENT_FOLLOW" in heatmaps
    assert (tmp_path / "segment_follow_heatmap.png").exists()


def _make_mock_buffer(scenario):
    buffer = ReplayBuffer()
    obs = np.zeros(8, dtype=np.float32)
    next_obs = np.ones(8, dtype=np.float32)
    buffer.add_trajectory(
        scenario,
        [
            TeacherTransition(
                obs=obs,
                next_obs=next_obs,
                goal_xy=np.zeros(2, dtype=np.float32),
                option_id=0,
                action_id=3,
                reward=1.0,
                done=True,
                stage_id=0,
                family_id=0,
                success=1.0,
                wall_contact=0.0,
                controller_id="A_STAR_NAVIGATION",
                scenario_name=scenario.name,
            )
        ],
    )
    buffer.relabel_future_observations(np.random.default_rng(0))
    return buffer


def test_train_smoke_both_modes(monkeypatch, tmp_path):
    train_scenarios = build_training_scenarios()
    eval_scenarios = build_eval_scenarios()
    scenario = train_scenarios["A1"][0]

    monkeypatch.setattr(training_module, "stage_order", lambda: ("A1",))
    monkeypatch.setattr(training_module, "build_training_scenarios", lambda: {"A1": train_scenarios["A1"][:1]})
    monkeypatch.setattr(
        training_module,
        "build_eval_scenarios",
        lambda: {
            "isolated_ood": eval_scenarios["isolated_ood"][:1],
            "composed_ood": eval_scenarios["composed_ood"][:1],
        },
    )
    monkeypatch.setattr(training_module, "collect_stage_data", lambda *args, **kwargs: _make_mock_buffer(scenario))
    monkeypatch.setattr(
        training_module,
        "_train_step",
        lambda bundle, batch, config: (bundle, {"loss": 0.0, "td_loss": 0.0, "teacher_loss": 0.0, "aux_loss": 0.0}),
    )
    monkeypatch.setattr(
        training_module,
        "evaluate_scenarios",
        lambda *args, **kwargs: {
            "isolated_ood": {
                "success_rate": 0.5,
                "steps_to_goal": 10.0,
                "timeout_rate": 0.5,
                "wall_contacts": 0.0,
                "normalized_efficiency": 0.7,
                "option_usage_segment_follow": 6.0,
                "option_usage_junction_resolve": 4.0,
            },
            "composed_ood": {
                "success_rate": 0.25,
                "steps_to_goal": 12.0,
                "timeout_rate": 0.75,
                "wall_contacts": 1.0,
                "normalized_efficiency": 0.4,
                "option_usage_segment_follow": 7.0,
                "option_usage_junction_resolve": 5.0,
            },
        },
    )

    def fake_heatmaps(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "segment_follow_heatmap.png").write_bytes(b"png")
        return {"SEGMENT_FOLLOW": np.zeros((2, 2), dtype=np.float32)}

    monkeypatch.setattr(training_module, "make_value_heatmaps", fake_heatmaps)

    base_kwargs = dict(
        seed=5,
        hidden_dims=(32, 32),
        embedding_dim=16,
        batch_size=8,
        updates_per_stage=1,
        trajectories_per_scenario=1,
        teacher_retry_limit=2,
        num_eval_episodes=1,
        heatmap_resolution=4,
        checkpoint_every=1,
        eval_every=1,
        action_scale=0.6,
        max_steps_per_episode=24,
        results_root=str(tmp_path),
    )

    manual_dir = training_module.train(
        TrainConfig(
            **base_kwargs,
            run_name="manual_curriculum",
            training_mode="manual_curriculum",
        )
    )
    no_dir = training_module.train(
        TrainConfig(
            **base_kwargs,
            run_name="no_curriculum",
            training_mode="no_curriculum",
        )
    )

    for run_dir in (manual_dir, no_dir):
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "eval_summary.csv").exists()
        assert (run_dir / "checkpoints" / "latest.msgpack").exists()


def test_build_comparison_plots_from_synthetic_runs(tmp_path):
    for run_name in ("manual_curriculum", "no_curriculum"):
        run_dir = tmp_path / run_name / "latest"
        run_dir.mkdir(parents=True)
        (run_dir / "eval_summary.csv").write_text(
            "run_name,stage_index,group,success_rate,normalized_efficiency\n"
            f"{run_name},1,isolated_ood,0.5,0.7\n",
            encoding="utf-8",
        )
        (run_dir / "learning_curve.csv").write_text(
            "run_name,stage_index,ood_success_rate\n"
            f"{run_name},1,0.5\n",
            encoding="utf-8",
        )
    output_dir = build_comparison_plots(tmp_path)
    assert (output_dir / "main_comparison.png").exists()
    assert (output_dir / "learning_curve.png").exists()

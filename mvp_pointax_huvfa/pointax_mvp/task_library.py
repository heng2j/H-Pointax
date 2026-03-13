"""Programmatic task library for the Pointax MVP."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .utils import NoiseSpec, ScenarioSpec, stage_order


Grid = List[List[object]]


def _blank_grid(height: int, width: int) -> Grid:
    grid = [[1 for _ in range(width)] for _ in range(height)]
    return grid


def _carve_rect(grid: Grid, top: int, left: int, height: int, width: int) -> None:
    for row in range(top, top + height):
        for col in range(left, left + width):
            grid[row][col] = 0


def _freeze(grid: Grid) -> Tuple[Tuple[object, ...], ...]:
    return tuple(tuple(row) for row in grid)


def _mark_start_goal(grid: Grid, start_cell: Tuple[int, int], goal_cell: Tuple[int, int]) -> None:
    sr, sc = start_cell
    gr, gc = goal_cell
    grid[sr][sc] = "R"
    grid[gr][gc] = "G"


def make_straight_corridor(
    name: str,
    family: str,
    split: str,
    length: int,
    width: int = 1,
    orientation: str = "horizontal",
    noise: NoiseSpec = NoiseSpec(),
    max_steps: int = 120,
) -> ScenarioSpec:
    if orientation == "horizontal":
        height = width + 4
        total_width = length + 4
        grid = _blank_grid(height, total_width)
        _carve_rect(grid, 2, 2, width, length)
        start_cell = (2 + width // 2, 2)
        goal_cell = (2 + width // 2, total_width - 3)
    else:
        height = length + 4
        total_width = width + 4
        grid = _blank_grid(height, total_width)
        _carve_rect(grid, 2, 2, length, width)
        start_cell = (height - 3, 2 + width // 2)
        goal_cell = (2, 2 + width // 2)
    _mark_start_goal(grid, start_cell, goal_cell)
    return ScenarioSpec(
        name=name,
        family=family,
        stage=family,
        split=split,
        maze_layout=_freeze(grid),
        start_cell=start_cell,
        goal_cell=goal_cell,
        noise=noise,
        max_steps=max_steps,
        metadata={"length": length, "width": width, "orientation": orientation},
    )


def make_l_turn_corridor(
    name: str,
    family: str,
    split: str,
    horizontal: int,
    vertical: int,
    width: int = 1,
    turn: str = "up_right",
    noise: NoiseSpec = NoiseSpec(),
    max_steps: int = 140,
) -> ScenarioSpec:
    height = vertical + width + 4
    total_width = horizontal + width + 4
    grid = _blank_grid(height, total_width)
    _carve_rect(grid, height - width - 2, 2, width, horizontal)
    _carve_rect(grid, 2, total_width - width - 2, vertical, width)

    if turn == "up_right":
        start_cell = (height - width - 2 + width // 2, 2)
        goal_cell = (2, total_width - width - 2 + width // 2)
    else:
        start_cell = (2, total_width - width - 2 + width // 2)
        goal_cell = (height - width - 2 + width // 2, 2)
    _mark_start_goal(grid, start_cell, goal_cell)
    return ScenarioSpec(
        name=name,
        family=family,
        stage=family,
        split=split,
        maze_layout=_freeze(grid),
        start_cell=start_cell,
        goal_cell=goal_cell,
        noise=noise,
        max_steps=max_steps,
        metadata={"horizontal": horizontal, "vertical": vertical, "width": width, "turn": turn},
    )


def make_t_junction(
    name: str,
    family: str,
    split: str,
    approach: int,
    left_branch: int,
    right_branch: int,
    width: int = 1,
    goal_branch: str = "left",
    noise: NoiseSpec = NoiseSpec(),
    max_steps: int = 150,
) -> ScenarioSpec:
    height = approach + 5
    total_width = left_branch + right_branch + width + 4
    center_col = left_branch + 2
    grid = _blank_grid(height, total_width)
    _carve_rect(grid, 2, center_col, approach + 1, width)
    _carve_rect(grid, 2, 2, width, left_branch + right_branch + width)
    start_cell = (height - 2, center_col)
    goal_cell = (2, 2) if goal_branch == "left" else (2, total_width - 3)
    _mark_start_goal(grid, start_cell, goal_cell)
    return ScenarioSpec(
        name=name,
        family=family,
        stage=family,
        split=split,
        maze_layout=_freeze(grid),
        start_cell=start_cell,
        goal_cell=goal_cell,
        noise=noise,
        max_steps=max_steps,
        metadata={
            "approach": approach,
            "left_branch": left_branch,
            "right_branch": right_branch,
            "width": width,
            "goal_branch": goal_branch,
        },
    )


def make_composed_route(
    name: str,
    family: str,
    split: str,
    pre_length: int,
    branch_length: int,
    post_length: int,
    width: int = 1,
    goal_branch: str = "right",
    noise: NoiseSpec = NoiseSpec(),
    max_steps: int = 180,
) -> ScenarioSpec:
    height = pre_length + post_length + width + 6
    total_width = branch_length + 6
    center_col = total_width // 2
    grid = _blank_grid(height, total_width)
    _carve_rect(grid, height - pre_length - 2, center_col, pre_length + 1, width)
    branch_top = height - pre_length - 2
    _carve_rect(grid, branch_top, 2, width, branch_length)
    if goal_branch == "right":
        _carve_rect(grid, 2, 2 + branch_length - width, post_length, width)
        goal_cell = (2, 2 + branch_length - width)
    else:
        _carve_rect(grid, 2, 2, post_length, width)
        goal_cell = (2, 2)
    start_cell = (height - 2, center_col)
    _mark_start_goal(grid, start_cell, goal_cell)
    return ScenarioSpec(
        name=name,
        family=family,
        stage=family,
        split=split,
        maze_layout=_freeze(grid),
        start_cell=start_cell,
        goal_cell=goal_cell,
        noise=noise,
        max_steps=max_steps,
        metadata={
            "pre_length": pre_length,
            "branch_length": branch_length,
            "post_length": post_length,
            "width": width,
            "goal_branch": goal_branch,
        },
    )


def _corridor_noise() -> NoiseSpec:
    return NoiseSpec(obs_noise_std=0.03, goal_mask_prob=0.15, action_noise_std=0.05, refresh_interval=2)


def _junction_noise() -> NoiseSpec:
    return NoiseSpec(
        obs_noise_std=0.05,
        goal_mask_prob=0.25,
        action_noise_std=0.06,
        observation_delay=1,
        refresh_interval=2,
        drift_bias_std=0.02,
    )


def build_training_scenarios() -> Dict[str, List[ScenarioSpec]]:
    scenarios: Dict[str, List[ScenarioSpec]] = {
        "A1": [
            make_straight_corridor("a1_short_h", "A1", "train", length=5, width=1),
            make_straight_corridor("a1_long_h", "A1", "train", length=7, width=1),
            make_straight_corridor("a1_vertical_w2", "A1", "train", length=6, width=2, orientation="vertical"),
        ],
        "A2": [
            make_l_turn_corridor("a2_turn_small", "A2", "train", horizontal=4, vertical=4, width=1),
            make_l_turn_corridor("a2_turn_wide", "A2", "train", horizontal=5, vertical=5, width=2),
            make_l_turn_corridor("a2_turn_long", "A2", "train", horizontal=6, vertical=4, width=1),
        ],
        "A3": [
            make_t_junction("a3_left", "A3", "train", approach=5, left_branch=4, right_branch=4, goal_branch="left"),
            make_t_junction("a3_right", "A3", "train", approach=5, left_branch=4, right_branch=4, goal_branch="right"),
            make_t_junction("a3_wide", "A3", "train", approach=6, left_branch=5, right_branch=3, width=2, goal_branch="left"),
        ],
        "B1": [
            make_straight_corridor("b1_noisy_line", "B1", "train", length=7, width=1, noise=_corridor_noise()),
            make_l_turn_corridor("b1_noisy_turn", "B1", "train", horizontal=5, vertical=5, width=1, noise=_corridor_noise()),
        ],
        "B2": [
            make_t_junction("b2_noisy_left", "B2", "train", approach=5, left_branch=4, right_branch=4, goal_branch="left", noise=_junction_noise()),
            make_t_junction("b2_noisy_right", "B2", "train", approach=5, left_branch=4, right_branch=5, goal_branch="right", noise=_junction_noise()),
        ],
    }
    return scenarios


def build_eval_scenarios() -> Dict[str, List[ScenarioSpec]]:
    return {
        "isolated_ood": [
            make_straight_corridor("e2_corridor_wide", "E2", "ood", length=8, width=2),
            make_l_turn_corridor("e2_turn_unseen", "E2", "ood", horizontal=7, vertical=5, width=1),
            make_t_junction("e2_junction_shift", "E2", "ood", approach=6, left_branch=6, right_branch=3, goal_branch="right"),
        ],
        "semi_uncertainty_ood": [
            make_straight_corridor(
                "e3_corridor_hard_noise",
                "E3",
                "ood",
                length=8,
                width=1,
                noise=NoiseSpec(obs_noise_std=0.06, goal_mask_prob=0.3, action_noise_std=0.08, observation_delay=2, refresh_interval=3),
            ),
            make_t_junction(
                "e3_junction_hard_noise",
                "E3",
                "ood",
                approach=6,
                left_branch=5,
                right_branch=5,
                goal_branch="left",
                noise=NoiseSpec(obs_noise_std=0.08, goal_mask_prob=0.35, action_noise_std=0.08, observation_delay=2, refresh_interval=3, drift_bias_std=0.03),
            ),
        ],
        "composed_ood": [
            make_composed_route("e1_comp_right", "E1", "ood", pre_length=6, branch_length=7, post_length=4, goal_branch="right"),
            make_composed_route("e1_comp_left", "E1", "ood", pre_length=7, branch_length=8, post_length=5, goal_branch="left"),
        ],
    }


def iter_all_training_scenarios() -> Iterable[ScenarioSpec]:
    train = build_training_scenarios()
    for stage in stage_order():
        for scenario in train[stage]:
            yield scenario


def iter_all_eval_scenarios() -> Iterable[ScenarioSpec]:
    eval_sets = build_eval_scenarios()
    for scenarios in eval_sets.values():
        for scenario in scenarios:
            yield scenario

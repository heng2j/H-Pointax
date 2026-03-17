"""Scripted teacher built from grid planning and local waypoint following."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import numpy as np

from .discrete_actions import direction_to_action
from .envs import build_base_env
from .utils import OPTION_TO_ID, ScenarioSpec


Cell = Tuple[int, int]


@dataclass
class TeacherTransition:
    obs: np.ndarray
    next_obs: np.ndarray
    goal_xy: np.ndarray
    option_id: int
    action_id: int
    reward: float
    done: bool
    stage_id: int
    family_id: int
    success: float
    wall_contact: float
    controller_id: str
    scenario_name: str
    future_obs: Optional[np.ndarray] = None


def neighbors(cell: Cell, free_mask: np.ndarray) -> Iterable[Cell]:
    row, col = cell
    for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = row + drow, col + dcol
        if 0 <= nr < free_mask.shape[0] and 0 <= nc < free_mask.shape[1] and free_mask[nr, nc]:
            yield nr, nc


def shortest_path(free_mask: np.ndarray, start_cell: Cell, goal_cell: Cell) -> List[Cell]:
    frontier: deque[Cell] = deque([start_cell])
    parent: Dict[Cell, Optional[Cell]] = {start_cell: None}
    while frontier:
        cell = frontier.popleft()
        if cell == goal_cell:
            break
        for next_cell in neighbors(cell, free_mask):
            if next_cell not in parent:
                parent[next_cell] = cell
                frontier.append(next_cell)
    if goal_cell not in parent:
        raise ValueError(f"No path between {start_cell} and {goal_cell}.")
    path: List[Cell] = []
    cursor: Optional[Cell] = goal_cell
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return list(reversed(path))


def branching_cells(free_mask: np.ndarray) -> List[Cell]:
    cells: List[Cell] = []
    for row in range(free_mask.shape[0]):
        for col in range(free_mask.shape[1]):
            if free_mask[row, col]:
                degree = sum(1 for _ in neighbors((row, col), free_mask))
                if degree >= 3:
                    cells.append((row, col))
    return cells


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def infer_option_for_path_index(path: Sequence[Cell], path_index: int, branch_cells: Sequence[Cell]) -> int:
    current_cell = path[min(path_index, len(path) - 1)]
    next_cell = path[min(path_index + 1, len(path) - 1)]
    if any(manhattan(current_cell, branch) <= 1 or manhattan(next_cell, branch) <= 1 for branch in branch_cells):
        return OPTION_TO_ID["JUNCTION_RESOLVE"]
    if 0 < path_index < len(path) - 1:
        prev_cell = path[path_index - 1]
        first_delta = (current_cell[0] - prev_cell[0], current_cell[1] - prev_cell[1])
        second_delta = (next_cell[0] - current_cell[0], next_cell[1] - current_cell[1])
        if first_delta != second_delta:
            return OPTION_TO_ID["JUNCTION_RESOLVE"]
    return OPTION_TO_ID["SEGMENT_FOLLOW"]


def choose_recenter_action(wrapper, current_cell: Cell) -> int:
    center = wrapper.cell_to_world(current_cell)
    control = center - wrapper.current_true_position() - 0.35 * wrapper.current_velocity()
    return direction_to_action(control)


def choose_path_action(wrapper, next_cell: Cell) -> int:
    target_xy = wrapper.cell_to_world(next_cell)
    control = target_xy - wrapper.current_true_position() - 0.35 * wrapper.current_velocity()
    return direction_to_action(control)


def plan_from_current_cell(wrapper, free_mask: np.ndarray, current_cell: Cell, goal_cell: Cell) -> List[Cell]:
    if free_mask[current_cell[0], current_cell[1]]:
        return shortest_path(free_mask, current_cell, goal_cell)

    frontier: deque[Cell] = deque([current_cell])
    seen = {current_cell}
    while frontier:
        cell = frontier.popleft()
        if 0 <= cell[0] < free_mask.shape[0] and 0 <= cell[1] < free_mask.shape[1] and free_mask[cell[0], cell[1]]:
            return shortest_path(free_mask, cell, goal_cell)
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_cell = (cell[0] + drow, cell[1] + dcol)
            if next_cell not in seen and 0 <= next_cell[0] < free_mask.shape[0] and 0 <= next_cell[1] < free_mask.shape[1]:
                seen.add(next_cell)
                frontier.append(next_cell)
    raise ValueError(f"No reachable free cell from {current_cell}.")


def collect_teacher_trajectory(
    scenario: ScenarioSpec,
    seed: int,
    action_scale: float,
    max_steps: Optional[int] = None,
) -> List[TeacherTransition]:
    bundle = build_base_env(scenario, max_steps=max_steps or scenario.max_steps, action_scale=action_scale)
    wrapper = bundle.wrapper
    key = jax.random.PRNGKey(seed)
    obs, _ = wrapper.reset(key)
    free_mask = wrapper.free_cell_mask()
    branch_cells = branching_cells(free_mask)
    transitions: List[TeacherTransition] = []
    final_success = False

    for step in range(max_steps or scenario.max_steps):
        key = jax.random.fold_in(key, step + 1)
        true_cell = wrapper.world_to_cell(wrapper.current_true_position())
        current_center = wrapper.cell_to_world(true_cell)
        if np.linalg.norm(wrapper.current_true_position() - current_center) > 0.35:
            action_id = choose_recenter_action(wrapper, true_cell)
            controller_id = "RECOVER_RECENTER"
            path = [true_cell, true_cell]
            option_id = OPTION_TO_ID["SEGMENT_FOLLOW"]
        else:
            try:
                path = plan_from_current_cell(wrapper, free_mask, true_cell, scenario.goal_cell)
            except ValueError:
                return []
            if len(path) <= 1:
                action_id = 8
                controller_id = "A_STAR_NAVIGATION"
                option_id = OPTION_TO_ID["SEGMENT_FOLLOW"]
            else:
                next_cell = path[1]
                action_id = choose_path_action(wrapper, next_cell)
                controller_id = "A_STAR_NAVIGATION"
                option_id = infer_option_for_path_index(path, 0, branch_cells)
        next_obs, reward, done, info = wrapper.step(action_id, key)
        transitions.append(
            TeacherTransition(
                obs=np.asarray(obs, dtype=np.float32),
                next_obs=np.asarray(next_obs, dtype=np.float32),
                goal_xy=wrapper.current_goal().copy(),
                option_id=option_id,
                action_id=int(action_id),
                reward=float(reward),
                done=bool(done),
                stage_id=0,
                family_id=0,
                success=float(info["success"]),
                wall_contact=float(info["wall_contact"]),
                controller_id=controller_id,
                scenario_name=scenario.name,
            )
        )
        obs = next_obs
        if done:
            final_success = bool(info["success"])
            break
    return transitions if final_success else []

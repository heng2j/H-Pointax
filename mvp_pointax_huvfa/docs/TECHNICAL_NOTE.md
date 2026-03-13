# Technical Note

## Architecture

The MVP uses Pointax as the continuous dynamics backend and adds a discrete controller interface, scripted teacher rollouts, and a shared Flax UVF critic:

- `Q_action(s, g, o, a)` is the primary learned quantity.
- `Q_option(s, g, o)` is derived as `max_a Q_action(s, g, o, a)`.
- `s` is the 8D wrapped observation: Pointax observation plus `goal_visible` and `obs_age`.
- `o` is one of the learned skills: `SEGMENT_FOLLOW`, `JUNCTION_RESOLVE`.

## Task Families

- `A1`: straight corridors
- `A2`: L-turn corridors
- `A3`: T-junctions
- `B1`: noisy corridors
- `B2`: noisy junctions
- `E1`: composed corridor-junction-corridor OOD mazes
- `E2`: geometry-shift OOD mazes
- `E3`: harder uncertainty OOD mazes

## Curriculum

- `manual_curriculum`: trains on `A1 -> A2 -> A3 -> B1 -> B2` with fixed stage fractions.
- `no_curriculum`: trains on the same families under the same total budget but samples from the mixed task set from the start.

## Evaluation

Primary metrics:

- success rate
- steps to goal
- normalized efficiency
- timeout rate
- wall-contact count
- option usage counts

Plots:

- OOD comparison bars
- learning curves
- per-option value heatmaps
- option preference maps

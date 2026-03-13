# What Worked / What Failed / Next Steps

## What Worked

- Pointax custom mazes were enough to express the MVP task families without changing physics.
- A single option-conditioned critic was sufficient to support shared value maps across corridor and junction behaviors.
- Offline teacher collection kept the training loop compact and reproducible.

## What Failed Or Stayed Deferred

- No online fine-tuning path is included in the baseline.
- The optional HCRL auxiliary is implemented conservatively and may need tuning per environment budget.
- The current teacher uses grid planning plus local waypoint following rather than a continuous optimal controller.

## Next Steps

- Add multi-seed sweeps and aggregate confidence intervals.
- Add online fine-tuning from the offline checkpoint.
- Add confidence-triggered fallback and intervention logging.
- Broaden the task library toward multi-waypoint navigation and longer horizon composed mazes.

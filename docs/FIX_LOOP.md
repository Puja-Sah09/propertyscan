# Fix loop

## Worst gate

Ceiling height, hall. Tape 2.440 m. The percentile estimator returned 2.455 m. Error 1.52 cm. The gate is 1.5 cm.

Living was 2.452 m (1.2 cm), kitchen 2.452 m (1.2 cm), bedroom 2.484 m against 2.470 m (1.4 cm). The hall was the only miss. Openings on that same run were inside 2 cm, and the wall lengths were inside 1 cm or 0.5 percent.

## Cause

`_levels` took the 5th and 95th percentiles of the in-room cloud. The ceiling is a thin mode. Depth noise above it is sparse, and the 95th percentile sits on that tail, not on the mode. The hall is the smallest room, so the tail is a larger share of its points. Evidence is the before plan at `reports/benchmark/lidar_before/plan.json`, regenerated with `level_mode=percentile` on the same 129 frames as the after plan.

## Fix

The floor and the ceiling are the outermost histogram bins that still hold a dense count (1.5 cm bins, at least 0.3 percent of the points), and the value is the median of the points within 2 cm of that bin. Sparse tails do not vote. The old percentile function is still `_levels_percentile`, so the before state stays regenerable.

## Prediction and result

The dense bin should land on the ceiling mode and cut the hall error from 1.5 cm to a few millimetres.

After, on the same frames (`reports/benchmark/lidar_after/plan.json`):

| Room | Tape | Before | After |
|---|---|---|---|
| living | 2.440 | 2.452 | 2.442 |
| hall | 2.440 | 2.455 | 2.442 |
| kitchen | 2.440 | 2.452 | 2.441 |
| bedroom | 2.470 | 2.484 | 2.472 |

Worst error after the change is 0.18 cm. The gate is 1.5 cm. A second walk, shifted 4 cm, repeats those ceilings within 1 cm.

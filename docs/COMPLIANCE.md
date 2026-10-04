# Compliance matrix

| Requirement | Result | Evidence |
|---|---|---|
| One command per capture, JSON plus SVG | Met | `py -m propertyscan run`. Schema `schema/plan.schema.json`. |
| LiDAR rooms, walls, openings, ceiling, area, intervals | Met on the synthetic building | `reports/benchmark/lidar_after/plan.json`. Worst ceiling error 0.18 cm. Openings inside 0.6 cm. |
| Opening gate, 2 cm on at least 85 percent, misses and phantoms count | Met | Rate 100 percent of 6 counted openings. `reports/benchmark/score.json`. |
| Ceiling gate, 1.5 cm, and 1 cm across a repeat | Met after the fix. Failed before it | Before hall 1.52 cm. After 0.18 cm. Repeat spread inside 1 cm. |
| Wall repeatability, 1 cm or 0.5 percent | Met | `score.json` key `repeat`. |
| Drift ablation fails if poses are used as they arrived | Met as a rule. The graph's own update was rejected | `lidar_poses_as_is` fails. The graph proposed a 2.41 m move and was not applied. |
| Interval contains the tape | Met on the LiDAR after plan | `calibration.all_inside` true. |
| Photo stitch, adjacency, no overlap, footprint and walls within 8 percent | Met | Footprint 54.43 m² vs 53.01 m² (2.7 percent). Overlap 0. Four rooms, correct adjacency. `photos_plan/plan.json`. |
| Video walls within 3 percent, tape inside every interval | Met | Four rooms. Worst wall 5.084 m vs 5.000 m (1.7 percent). Bedroom long wall 4.205 m vs 4.200 m. Ceilings 2.40 m, inside 8 cm. `video_plan/plan.json`. |
| Fix loop, declared then shipped, both states regenerable | Met | `docs/FIX_LOOP.md`. `level_mode=percentile` is the before estimator. |
| magicplan head-to-head | Not run | No Statistics CSV from the app. `compare` exits 2. No area was filled in. Export steps are in `docs/CAPTURE_ROUTE.md`. |
| Capture route and device matrix | Written | `docs/CAPTURE_ROUTE.md`, `docs/DEVICE_MATRIX.md`. |
| Field iPhone LiDAR | Run on the three provided archives | `single_room` 22.51 m², no ceiling. `single_scan_floor_only` 32.74 m², no ceiling. `single_scan_with_ceiling` 33.19 m², ceiling plane 3.10 m. `reports/samples/`. No tape, so not a gate. This PC did not record a new walk. |
| Low light, mirror, glass, wet floor | Named, not captured | Low-light intervals widen by 1.8 when the median grey is under 25. The others are in the technical report. |
| No call to the author's infrastructure | Met | Local ray caster and local files only. |
| Models disclosed | Met | There is no learned model. |
| Process, commits as the work landed | Met | More than two commits. Portable Git, because the system installer was cancelled at the administrator prompt. |

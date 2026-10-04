# Synthetic benchmark

Regenerate with `py -m propertyscan benchmark -o reports`. The tape is `ground_truth.json`. The estimator never imports it.

129 depth frames, 240×180, 8 mm noise. The harness adds 0.018 degrees and 1.6 mm of drift per frame. The pose graph proposed a median move of 2.41 m over 9 loop closures. That update was rejected, and the plans below are the captured poses. Raw depth is `lidar_primary.npz`.

## LiDAR after the ceiling fix

Passed, including the repeat walk.

Worst ceiling error 0.18 cm (gate 1.5 cm). Every opening that was counted is inside 2 cm (rate 100 percent). Every wall is inside 1 cm or 0.5 percent. Every scored tape value sits inside its reported interval. A second walk shifted by 4 cm repeats the walls and the ceilings inside the repeat gates.

| Room | Area m² | Ceiling m | Tape ceiling |
|---|---|---|---|
| living | 19.994 | 2.442 | 2.440 |
| hall | 4.798 | 2.442 | 2.440 |
| kitchen | 12.246 | 2.441 | 2.440 |
| bedroom | 15.952 | 2.472 | 2.470 |

Openings: 0.805 m and 0.804 m against 0.80 m, 0.904 m against 0.90 m.

## LiDAR before the ceiling fix

Same frames, percentile 5/95. Failed the ceiling gate. Hall 2.455 m, error 1.52 cm. The other rooms were inside 1.5 cm. Openings and walls still passed. See `docs/FIX_LOOP.md`.

## Poses used as they arrived

Same measurements as the after plan, because the graph update was rejected. The run is a fail: `used_raw_poses_as_is` is set. That is the ablation rule, and it is automatic.

## Photos

Passed. Four rooms at 960×720, stitched on the doorway the camera was aimed at. Footprint 54.43 m² against 53.01 m² (2.7 percent). Overlap 0. Adjacency is living–hall, hall–kitchen, hall–bedroom. Walls are inside 8 percent. The bedroom area is 16.67 m² against 15.96 m², inside the 5 percent area interval. Ceilings read 2.40 m against 2.44 m and 2.47 m, inside the 15 cm interval.

## Video

Not passed. The walk uses a 110 degree lens so the floor stays in frame, and the phone height stays on the sheet measurement. The track returns one 5.62 m² fragment, matched to the living room. Two sides are near 5 m and the other two are under 2 m, so the 3 percent wall gate fails. The ceiling samples do not land inside the fragment, and the height collapses.

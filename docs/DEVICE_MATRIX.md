# Device matrix

| Source | What it is | Used for | Result |
|---|---|---|---|
| Synthetic ray caster, 62° pinhole, 240×180 depth | The benchmark building. Depth noise is 8 mm. Drift is added by the harness, not by the caster. | Scored LiDAR gates, the fix loop, the repeat walk | `py -m propertyscan benchmark` |
| Same caster, 960×720 colour | Letter-sheet photo protocol | Photo tier | Written under `reports/benchmark/photos_plan/` when the benchmark finishes |
| Same caster, 320×240 colour, 110° wide lens | Sheet, then a walk that keeps the floor in frame, then the sheet | Video tier | One 5.62 m² fragment. Does not pass the 3 percent wall gate. `video_plan/` |
| Stray Scanner zip, iPhone LiDAR | `single_room.zip`, `single_scan_floor_only.zip`, `single_scan_with_ceiling.zip` | Real geometry. No tape, so not scored | `reports/samples/` |
| Record3D `.r3d` | Loader is implemented (OpenGL pose to OpenCV, LZFSE depth) | No `.r3d` file was in this environment | Not run |
| magicplan free Starter, Statistics CSV | The required comparison app | Not exported here | `compare` exits 2. No number was invented |
| Phone camera, low light, wet floor, mirror, glass | Named failure cases | Not captured | See the technical report. Low-light intervals widen when the median grey is under 25 |

The estimator does not call out to a model server. There is no learned network in the pipeline.

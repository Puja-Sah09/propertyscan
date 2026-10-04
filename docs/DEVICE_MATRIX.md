# Device matrix

| Source | What it is | Used for | Result |
|---|---|---|---|
| Synthetic ray caster, 62° pinhole, 240×180 depth | The benchmark building. Depth noise is 8 mm. Drift is added by the harness, not by the caster. | Scored LiDAR gates, the fix loop, the repeat walk | `py -m propertyscan benchmark` |
| Same caster, 960×720 colour | Letter-sheet photo protocol | Photo tier | Written under `reports/benchmark/photos_plan/` when the benchmark finishes |
| Same caster, 960×720 colour, 62° | Each room from its own letter sheet. Doorway looks at the ends of the visit. Hall filmed twice. Bedroom long walls 1.7 m back from the sheet | Video tier | Four rooms. Walls inside 3 percent. Tape inside every interval. `video_plan/` |
| Stray Scanner zip, iPhone LiDAR | `single_room.zip`, `single_scan_floor_only.zip`, `single_scan_with_ceiling.zip` | Field captures supplied with the brief. No tape, so not a gate | 22.51 m², 32.74 m², 33.19 m² with a 3.10 m ceiling. `reports/samples/` |
| Record3D `.r3d` | Loader is implemented (OpenGL pose to OpenCV, LZFSE depth) | No `.r3d` file was in this environment | Not run |
| magicplan free Starter, Statistics CSV | The required comparison app | Not exported here | `compare` exits 2. No number was invented |
| Phone camera, low light, wet floor, mirror, glass | Named failure cases | Not captured | See the technical report. Low-light intervals widen when the median grey is under 25 |

The estimator does not call out to a model server. There is no learned network in the pipeline.

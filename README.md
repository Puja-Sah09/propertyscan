# propertyscan

One capture in, one dimensioned plan out. LiDAR, photos, or video. Every length carries a 95 percent interval. The plan is JSON checked against `schema/plan.schema.json`, plus an SVG.

No learned model. Geometry only: a pose graph on the depth, line and room extraction, and a letter sheet for scale when there is no LiDAR.

## Install

Python 3.12. From this directory:

```
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Record3D `.r3d` depth needs `lzfse`, which is in that file.

## One command

```
.venv\Scripts\python -m propertyscan run CAPTURE -o OUT
```

`CAPTURE` is a Record3D `.r3d`, a Stray Scanner zip (the three sample archives), or a folder with `capture.json` (`tier` of `lidar`, `photos`, or `video`).

```
.venv\Scripts\python -m propertyscan run CAPTURE.zip -o OUT --no-drift-correction
.venv\Scripts\python -m propertyscan run CAPTURE.zip -o OUT --openings coarse
.venv\Scripts\python -m propertyscan benchmark -o reports
.venv\Scripts\python -m propertyscan compare PLAN.json magicplan.csv
```

`compare` exits 2 when the magicplan Statistics CSV is not there. It does not invent a comparison.

A fresh capture of the four-room route in `docs/CAPTURE_ROUTE.md` is the walk the benchmark replays. With the phone already recording, that walk is under 15 minutes.

## What was run here

Synthetic building, scored: `reports/benchmark/`. The hall ceiling was the failing gate; `docs/FIX_LOOP.md` is the before and after.

iPhone LiDAR from the three Drive archives, not scored (no tape): `reports/samples/`. Footprints 22.51 m², 32.74 m², and 33.19 m². The ceiling archive measures 3.10 m.

Photos of the synthetic building pass the stitch gate: footprint 54.43 m² against 53.01 m², no overlap. The video tier passes the 3 percent wall gate: four rooms, worst wall 5.084 m against 5.000 m, bedroom long wall 4.205 m against 4.200 m.

magicplan was not run on these rooms. `compare` exits 2 until a Statistics CSV from the app is passed, and it does not fill in a missing area. Git on this machine is a portable build under the user profile. The system installer was cancelled at the administrator prompt.

## Layout

| Path | Role |
|---|---|
| `propertyscan/` | Estimator. It does not import the floor-plan spec. |
| `benchmarks/` | Ray caster and the tape. Only the benchmark command imports it. |
| `schema/plan.schema.json` | Output contract. |
| `docs/` | Route, device matrix, compliance, fix loop, technical report. |

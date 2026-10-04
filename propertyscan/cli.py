"""One command per capture.

    py -m propertyscan run CAPTURE -o OUT
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from propertyscan import __version__
from propertyscan.contract import build_plan
from propertyscan.damage import detect_damage
from propertyscan.lidar import run_lidar
from propertyscan.photos import run_photos
from propertyscan.sensing import load_record3d, load_stray
from propertyscan.svgplan import render_svg
from propertyscan.video import run_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="propertyscan", description="Property scan to a dimensioned plan.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="One capture in, one plan out.")
    run.add_argument("capture", type=Path)
    run.add_argument("-o", "--out", type=Path, required=True)
    run.add_argument("--no-drift-correction", action="store_true")
    run.add_argument("--openings", choices=["refined", "coarse"], default="refined")
    run.add_argument("--device", default="")
    run.add_argument("--stride", type=int, default=None, help="Frame step for long LiDAR scans.")

    bench = sub.add_parser("benchmark", help="Regenerate the synthetic benchmark and score it.")
    bench.add_argument("-o", "--out", type=Path, default=Path("reports"))

    cmp = sub.add_parser("compare", help="Score a plan against a magicplan Statistics CSV.")
    cmp.add_argument("plan", type=Path)
    cmp.add_argument("magicplan_csv", type=Path)

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return _run(args)
    if args.cmd == "benchmark":
        from propertyscan.benchmark_cmd import run_benchmark

        return run_benchmark(args.out)
    if args.cmd == "compare":
        from propertyscan.score import compare_magicplan

        return compare_magicplan(args.plan, args.magicplan_csv)
    return 2


def _run(args) -> int:
    t0 = time.perf_counter()
    capture = load_capture(args.capture, frame_stride=getattr(args, "stride", None))
    tier = capture["tier"]
    if tier != "lidar":
        print(f"{tier} tier is loaded; running the {tier} pipeline", flush=True)
    if tier == "lidar":
        fused = run_lidar(
            capture,
            refine_openings=args.openings == "refined",
            drift_correction=not args.no_drift_correction,
        )
        half = 0.018 if fused.get("ceiling_observed", True) else 0.60
        ci = dict(length_half=0.015, length_rel=0.004, opening_half=0.012, ceiling_half=half)
    elif tier == "photos":
        fused = run_photos(capture)
        ci = dict(length_half=0.04, length_rel=0.08, opening_half=0.06, ceiling_half=0.15, area_rel=0.05)
    elif tier == "video":
        fused = run_video(capture)
        ci = dict(length_half=0.02, length_rel=0.03, opening_half=0.06, ceiling_half=0.08, area_rel=0.06)
    else:
        raise SystemExit(f"Unknown tier {tier}.")
    damages, flags, scope = [], [], []
    if tier == "lidar" and len(fused.get("xyz", [])):
        damages, flags, scope = detect_damage(fused["xyz"], fused["rgb"], fused["rooms"])
    plan = build_plan(
        rooms=fused["rooms"],
        tier=tier,
        damages=damages,
        flags=flags,
        scope=scope,
        drift=fused["drift"],
        device=args.device or capture.get("device") or "unknown",
        capture_id=args.capture.stem,
        names=fused.get("names"),
        ci_scale=1.8 if fused.get("dark") else 1.0,
        **ci,
    )
    if "ceiling_observed" in fused:
        plan["ceiling_observed"] = bool(fused["ceiling_observed"])
    elapsed = time.perf_counter() - t0
    plan["runtime_s"] = round(elapsed, 3)
    plan["propertyscan_version"] = __version__
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (args.out / "plan.svg").write_text(render_svg(plan), encoding="utf-8")
    print(
        f"rooms={len(plan['rooms'])} damages={len(plan['damages'])} "
        f"flags={len(plan['concealed_flags'])} scope={len(plan['scope_items'])} "
        f"drift={plan['drift']['method']} in {elapsed:.1f}s"
    )
    print(args.out / "plan.json")
    return 0


def load_capture(path: Path, frame_stride: int | None = None) -> dict:
    path = Path(path)
    if path.is_file() and path.suffix.lower() in {".zip", ".r3d"}:
        import zipfile

        with zipfile.ZipFile(path) as zf:
            stray = any(name.endswith("odometry.csv") for name in zf.namelist())
        if stray:
            print(f"reading Stray Scanner archive {path.name}", flush=True)
            return load_stray(path, frame_stride=frame_stride)
        cap = load_record3d(path)
        cap["device"] = cap.get("device", "Record3D")
        return cap
    if (path / "capture.json").is_file():
        meta = json.loads((path / "capture.json").read_text(encoding="utf-8"))
        tier = meta["tier"]
        if tier == "lidar":
            return _load_lidar_dir(path, meta)
        if tier == "photos":
            return {"tier": "photos", "root": path, "meta": meta, "device": meta.get("device", "")}
        if tier == "video":
            return {"tier": "video", "root": path, "meta": meta, "device": meta.get("device", "")}
    raise SystemExit(f"Cannot read a capture at {path}. Expected a .r3d file or a folder with capture.json.")


def _load_lidar_dir(path: Path, meta: dict) -> dict:
    k = meta["intrinsics"]
    intr = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=float)
    frame_dir = path / "frames"
    stems = sorted({p.name.split("_")[0] for p in frame_dir.glob("*_depth.png")})
    frames = []
    for stem in stems:
        depth_mm = cv2.imread(str(frame_dir / f"{stem}_depth.png"), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            continue
        depth = depth_mm.astype(np.float32) / 1000.0
        rgb = cv2.imread(str(frame_dir / f"{stem}_rgb.png"), cv2.IMREAD_COLOR)
        pose = json.loads((frame_dir / f"{stem}_pose.json").read_text(encoding="utf-8"))
        frames.append({"depth_m": depth, "rgb": rgb, "T": np.array(pose["T"], dtype=float), "confidence": None})
    return {
        "tier": "lidar",
        "intrinsics": intr,
        "frames": frames,
        "device": meta.get("device", ""),
        "source": str(path),
    }


if __name__ == "__main__":
    sys.exit(main())

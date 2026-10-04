"""Regenerate the synthetic building and score it.

The ray caster is the only code that knows the floor plan. This command writes
ground_truth.json, runs the estimator, and writes the score. Both the ceiling
estimator that failed the hall and the dense-bin replacement are run on the
same frames, so the fix loop can be regenerated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from propertyscan import __version__
from propertyscan.contract import build_plan
from propertyscan.damage import detect_damage
from propertyscan.lidar import run_lidar
from propertyscan.score import score_plan, score_repeat
from propertyscan.svgplan import render_svg


def run_benchmark(out: Path) -> int:
    out = Path(out)
    bench = out / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    from benchmarks.floorplan_spec import ground_truth
    from benchmarks.raycast import (
        capture_shots,
        drift_pose,
        look_at,
        make_intrinsics,
        photo_views,
        render,
        video_shots,
    )

    gt = ground_truth()
    (bench / "ground_truth.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print("rendering drifted walk", flush=True)
    primary = _render_lidar(capture_shots(), make_intrinsics(240, 180), drift=True, seed_shift=0.0, render=render, look_at=look_at, drift_pose=drift_pose)
    print(f"frames={len(primary['frames'])}", flush=True)
    _save_raw(bench / "lidar_primary.npz", primary)

    after = _plan(primary, bench / "lidar_after", level_mode="dense", drift_correction=True)
    before = _plan(primary, bench / "lidar_before", level_mode="percentile", drift_correction=True)
    ablation = _plan(primary, bench / "lidar_no_drift_correction", level_mode="dense", drift_correction=False)

    print("rendering repeat walk", flush=True)
    repeat = _render_lidar(
        capture_shots(), make_intrinsics(240, 180), drift=True, seed_shift=0.04, render=render, look_at=look_at, drift_pose=drift_pose
    )
    repeat_plan = _plan(repeat, bench / "lidar_repeat", level_mode="dense", drift_correction=True)

    report = {
        "propertyscan_version": __version__,
        "resolution": [240, 180],
        "frames": len(primary["frames"]),
        "drift_model": "yaw 0.018 deg/frame and 1.6 mm/frame, injected by the harness",
        "lidar_after": score_plan(after, gt, "lidar"),
        "lidar_before": score_plan(before, gt, "lidar"),
        "lidar_poses_as_is": score_plan(ablation, gt, "lidar"),
        "repeat": score_repeat(after, repeat_plan, gt),
    }
    report["lidar_after"]["passed"] = bool(
        report["lidar_after"]["passed"] and report["repeat"]["walls_ok"] and report["repeat"]["ceiling_spread_ok"]
    )
    (bench / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(bench / "BENCHMARK.md", report)
    print("lidar after passed" if report["lidar_after"]["passed"] else "lidar after failed", flush=True)

    photo = _photos(bench, gt, photo_views, render, look_at, make_intrinsics)
    video = _video(bench, gt, video_shots, render, look_at, make_intrinsics)
    report["photos"] = photo
    report["video"] = video
    (bench / "score.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(bench / "BENCHMARK.md", report)
    return 0 if report["lidar_after"]["passed"] else 1


def _render_lidar(shots, k, drift: bool, seed_shift: float, render, look_at, drift_pose) -> dict:
    frames = []
    for i, (eye, tgt) in enumerate(shots):
        eye = np.asarray(eye, float) + np.array([seed_shift, -seed_shift, 0.0])
        tgt = np.asarray(tgt, float) + np.array([seed_shift, -seed_shift, 0.0])
        pose = look_at(eye, tgt)
        if drift:
            yaw = i * math.radians(0.018)
            pose = drift_pose(pose, yaw, np.array([i * 0.0016, i * 0.0005]))
        depth, rgb = render(pose[:3, :3], pose[:3, 3], k, int(k[0, 2] * 2), int(k[1, 2] * 2), seed=i)
        frames.append({"depth_m": depth, "rgb": rgb, "T": pose, "confidence": None})
    return {"tier": "lidar", "device": "synthetic-raycast", "intrinsics": k, "frames": frames}


def _plan(capture: dict, dest: Path, level_mode: str, drift_correction: bool) -> dict:
    print(f"fusing {dest.name} level={level_mode} drift={drift_correction}", flush=True)
    fused = run_lidar(capture, refine_openings=True, drift_correction=drift_correction, level_mode=level_mode)
    damages, flags, scope = [], [], []
    if len(fused["xyz"]):
        damages, flags, scope = detect_damage(fused["xyz"], fused["rgb"], fused["rooms"])
    half = 0.018 if fused.get("ceiling_observed", True) else 0.60
    plan = build_plan(
        rooms=fused["rooms"],
        tier="lidar",
        damages=damages,
        flags=flags,
        scope=scope,
        drift=fused["drift"],
        device=capture.get("device") or "synthetic-raycast",
        capture_id=dest.name,
        length_half=0.015,
        length_rel=0.004,
        opening_half=0.012,
        ceiling_half=half,
        ci_scale=1.8 if fused.get("dark") else 1.0,
    )
    plan["level_mode"] = level_mode
    plan["ceiling_observed"] = bool(fused.get("ceiling_observed", True))
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (dest / "plan.svg").write_text(render_svg(plan), encoding="utf-8")
    _print_rooms(plan)
    return plan


def _print_rooms(plan: dict) -> None:
    for room in plan["rooms"]:
        walls = [round(w["length_m"]["value"], 3) for w in room["walls"]]
        ops = [round(o["width_m"]["value"], 3) for w in room["walls"] for o in w["openings"]]
        print(
            f"  {room['id']} area {room['floor_area_m2']['value']:.3f} "
            f"h {room['ceiling_height_m']['value']:.3f} walls {walls} ops {ops}",
            flush=True,
        )


def _save_raw(path: Path, capture: dict) -> None:
    depths = np.stack([(fr["depth_m"] * 1000).astype(np.uint16) for fr in capture["frames"]])
    poses = np.stack([fr["T"] for fr in capture["frames"]])
    np.savez_compressed(path, depth_mm=depths, poses=poses, k=capture["intrinsics"])


def _photos(bench: Path, gt: dict, photo_views, render, look_at, make_intrinsics) -> dict:
    print("rendering photo tier at 960x720", flush=True)
    try:
        root = bench / "photos"
        root.mkdir(parents=True, exist_ok=True)
        k = make_intrinsics(960, 720)
        meta = {
            "tier": "photos",
            "device": "synthetic-raycast",
            "intrinsics": {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2])},
        }
        (root / "capture.json").write_text(json.dumps(meta), encoding="utf-8")
        for name, shots in photo_views().items():
            folder = root / "rooms" / name
            folder.mkdir(parents=True, exist_ok=True)
            for stem, eye, tgt in shots:
                pose = look_at(eye, tgt)
                _depth, rgb = render(pose[:3, :3], pose[:3, 3], k, 960, 720, seed=0)
                cv2.imwrite(str(folder / f"{stem}.jpg"), rgb)
        from propertyscan.photos import run_photos

        fused = run_photos({"tier": "photos", "root": root, "meta": meta, "device": meta["device"]})
        plan = build_plan(
            rooms=fused["rooms"],
            tier="photos",
            damages=[],
            flags=[],
            scope=[],
            drift=fused["drift"],
            device=meta["device"],
            capture_id="photos",
            names=fused.get("names"),
            length_half=0.04,
            length_rel=0.08,
            opening_half=0.06,
            ceiling_half=0.15,
            area_rel=0.05,
        )
        dest = bench / "photos_plan"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (dest / "plan.svg").write_text(render_svg(plan), encoding="utf-8")
        scored = score_plan(plan, gt, "photos")
        print("photos passed" if scored["passed"] else "photos failed", flush=True)
        return scored
    except Exception as exc:
        print(f"photos did not complete: {exc}", flush=True)
        return {"passed": False, "error": str(exc)}


def _video(bench: Path, gt: dict, video_shots, render, look_at, make_intrinsics) -> dict:
    print("rendering video tier", flush=True)
    try:
        root = bench / "video"
        frames = root / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        k = make_intrinsics(960, 720)
        meta = {
            "tier": "video",
            "device": "synthetic-raycast",
            "intrinsics": {"fx": float(k[0, 0]), "fy": float(k[1, 1]), "cx": float(k[0, 2]), "cy": float(k[1, 2])},
        }
        (root / "capture.json").write_text(json.dumps(meta), encoding="utf-8")
        shots = video_shots(0.0)
        for i, (eye, tgt) in enumerate(shots):
            pose = look_at(eye, tgt)
            _depth, rgb = render(pose[:3, :3], pose[:3, 3], k, 960, 720, seed=i)
            cv2.imwrite(str(frames / f"{i:04d}.jpg"), rgb)
        from propertyscan.video import run_video

        fused = run_video({"tier": "video", "root": root, "meta": meta, "device": meta["device"]})
        plan = build_plan(
            rooms=fused["rooms"],
            tier="video",
            damages=[],
            flags=[],
            scope=[],
            drift=fused["drift"],
            device=meta["device"],
            capture_id="video",
            length_half=0.02,
            length_rel=0.03,
            opening_half=0.06,
            ceiling_half=0.08,
            area_rel=0.06,
        )
        dest = bench / "video_plan"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        (dest / "plan.svg").write_text(render_svg(plan), encoding="utf-8")
        scored = score_plan(plan, gt, "video")
        print("video passed" if scored["passed"] else "video failed", flush=True)
        return scored
    except Exception as exc:
        print(f"video did not complete: {exc}", flush=True)
        return {"passed": False, "error": str(exc)}


def _write_summary(path: Path, report: dict) -> None:
    lines = ["# Synthetic benchmark", ""]
    for key in ("lidar_before", "lidar_after", "lidar_poses_as_is"):
        block = report[key]
        lines.append(f"## {key}")
        lines.append("")
        lines.append(f"Passed: {block['passed']}. Drift: {block.get('drift_method')}.")
        lines.append(f"Openings within 2 cm: {block['opening_rate']:.0%} of {block['openings']['counted']}.")
        ceil = block["ceilings"]
        lines.append(f"Worst ceiling error: {ceil.get('worst_abs_m')} m.")
        for row in ceil.get("rows") or []:
            lines.append(f"- {row['room']}: {row['est_m']} m vs {row['gt_m']} m (error {row['err_m']} m)")
        lines.append("")
    rep = report.get("repeat") or {}
    lines.append("## Repeat capture")
    lines.append("")
    lines.append(f"Walls within 1 cm or 0.5 percent: {rep.get('walls_ok')}.")
    lines.append(f"Ceiling spread within 1 cm: {rep.get('ceiling_spread_ok')}.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

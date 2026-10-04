"""Score a plan against ground_truth.json. This module does not know the building.

Gates, from the brief:
- opening width error at most 2 cm on at least 85 percent (a miss and a phantom each count)
- ceiling error at most 1.5 cm in every room, and the spread across two captures at most 1 cm
- a wall repeats within 1 cm or 0.5 percent
- a run that used the poses as they arrived fails the drift ablation
- photo footprint and wall lengths within 8 percent, correct adjacency, no overlap
- video wall lengths within 3 percent
- every scored measurement's interval contains the tape
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np


def compare_magicplan(plan_path: Path, csv_path: Path) -> int:
    """Head-to-head against a magicplan Statistics export.

    Exit 2 when the CSV is missing, empty, or has no area column. Nothing here
    invents a magicplan number: a blank cell is skipped, not filled in.
    """
    plan_path = Path(plan_path)
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        print(
            "No magicplan Statistics CSV at "
            f"{csv_path}. Head-to-head was not run. "
            "Export Statistics from the free Starter plan and pass that file."
        )
        return 2
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    if not rows:
        print(f"{csv_path} has no rows.")
        return 2
    print(f"magicplan rows={len(rows)} plan rooms={len(plan.get('rooms', []))}")
    print("Columns:", ", ".join(rows[0].keys()))
    name_key, area_key = _magicplan_columns(rows[0])
    if area_key is None:
        print("No floor-area column in that export, so there is nothing to compare.")
        return 2
    theirs = []
    for row in rows:
        area = _parse_area(row.get(area_key, ""))
        if area is None:
            continue
        label = (row.get(name_key) or "").strip() if name_key else ""
        theirs.append((label, area))
    if not theirs:
        print("The area column is empty. No comparison was invented.")
        return 2
    ours = []
    for room in plan.get("rooms") or []:
        meas = room.get("floor_area_m2") or {}
        if "value" not in meas:
            continue
        ours.append((str(room.get("name") or room.get("id") or ""), float(meas["value"])))
    used = set()
    print(f"{'magicplan':<24} {'m2':>8} {'plan':<24} {'m2':>8} {'rel':>8}")
    for label, area in theirs:
        match = _match_room(label, ours, used)
        if match is None:
            print(f"{label or '(unnamed)':<24} {area:8.2f} {'—':<24} {'':>8} {'':>8}")
            continue
        used.add(match)
        name, value = ours[match]
        rel = abs(value - area) / area if area else 0.0
        print(f"{label or '(unnamed)':<24} {area:8.2f} {name:<24} {value:8.2f} {rel:8.3f}")
    for i, (name, value) in enumerate(ours):
        if i not in used:
            print(f"{'—':<24} {'':>8} {name:<24} {value:8.2f} {'':>8}")
    return 0


def _magicplan_columns(header: dict) -> tuple[str | None, str | None]:
    keys = list(header.keys())
    folded = {k: " ".join(k.lower().replace("²", "2").replace("_", " ").split()) for k in keys}

    def pick(needles: tuple[str, ...], reject: tuple[str, ...] = ()) -> str | None:
        for key, text in folded.items():
            if any(r in text for r in reject):
                continue
            if any(n in text for n in needles):
                return key
        return None

    area = pick(("floor area", "area m2", "area (m2)", "area"), reject=("ceiling", "wall", "volume"))
    name = pick(("room name", "room", "name", "space"), reject=("area",))
    return name, area


def _parse_area(text: str) -> float | None:
    raw = str(text).strip().lower().replace("m²", "").replace("m2", "").strip()
    if not raw:
        return None
    raw = raw.replace(" ", "")
    if raw.count(",") == 1 and raw.count(".") == 0:
        raw = raw.replace(",", ".")
    raw = raw.replace(",", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _match_room(label: str, ours: list[tuple[str, float]], used: set[int]) -> int | None:
    key = " ".join(label.lower().split())
    if not key:
        return None
    for i, (name, _value) in enumerate(ours):
        if i in used:
            continue
        if " ".join(name.lower().split()) == key:
            return i
    return None


def score_plan(plan: dict, gt: dict, tier: str = "lidar") -> dict:
    # Photos and video are metric in the letter-sheet frame. The tape plan is
    # in the building frame. Lengths do not depend on that choice; matching
    # rooms does. Lidar captures in this benchmark already share the tape frame.
    if tier in ("photos", "video"):
        plan = _sheet_frame_to_tape(plan, gt)
    pairs = _match_rooms(plan, gt)
    walls = _score_walls(pairs, relative=0.005 if tier != "photos" else 0.08, absolute=0.01 if tier != "photos" else 0.0)
    if tier == "video":
        walls = _score_walls(pairs, relative=0.03, absolute=0.0)
    openings = _score_openings(pairs, gt)
    ceilings = _score_ceilings(pairs)
    areas = _score_areas(pairs)
    calibration = _calibration(pairs, gt)
    drift = plan.get("drift") or {}
    drift_fail = bool(drift.get("used_raw_poses_as_is"))
    overlap = float((plan.get("stitch") or {}).get("overlap_area_m2") or 0.0)
    adjacency = _adjacency_ok(plan, pairs, gt)
    opening_rate = openings["within_gate"] / openings["counted"] if openings["counted"] else 1.0
    ceiling_ok = ceilings["worst_abs_m"] <= 0.015 if ceilings["n"] else False
    if tier == "lidar":
        passed = (not drift_fail) and opening_rate >= 0.85 and ceiling_ok and walls["all_within"] and calibration["all_inside"]
    elif tier == "photos":
        foot_err = _foot_err(plan, gt)
        passed = adjacency and overlap <= 0.05 and foot_err <= 0.08 and walls["all_within"] and calibration["all_inside"]
        areas["footprint_rel"] = foot_err
    else:
        passed = walls["all_within"] and calibration["all_inside"]
    return {
        "tier": tier,
        "passed": bool(passed),
        "rooms_matched": len(pairs),
        "rooms_gt": len(gt["rooms"]),
        "drift_method": drift.get("method"),
        "drift_automatic_fail": drift_fail,
        "openings": openings,
        "opening_rate": round(opening_rate, 4),
        "ceilings": ceilings,
        "walls": walls,
        "areas": areas,
        "calibration": calibration,
        "adjacency_ok": adjacency,
        "overlap_m2": overlap,
    }


def score_repeat(a: dict, b: dict, gt: dict) -> dict:
    """Wall repeatability and ceiling spread between two captures of the same building."""
    pairs_a = {g["name"]: est for est, g in _match_rooms(a, gt)}
    pairs_b = {g["name"]: est for est, g in _match_rooms(b, gt)}
    wall_deltas = []
    ceiling_spreads = []
    for name, gt_room in ((g["name"], g) for g in gt["rooms"]):
        if name not in pairs_a or name not in pairs_b:
            continue
        la = sorted(w["length_m"]["value"] for w in pairs_a[name]["walls"])
        lb = sorted(w["length_m"]["value"] for w in pairs_b[name]["walls"])
        for xa, xb, g in zip(la, lb, sorted(w["length_m"] for w in gt_room["walls"])):
            err = abs(xa - xb)
            limit = max(0.01, 0.005 * g)
            wall_deltas.append({"room": name, "delta_m": round(err, 4), "limit_m": round(limit, 4), "ok": err <= limit + 1e-9})
        ca = pairs_a[name]["ceiling_height_m"]["value"]
        cb = pairs_b[name]["ceiling_height_m"]["value"]
        spread = abs(ca - cb)
        ceiling_spreads.append({"room": name, "spread_m": round(spread, 4), "ok": spread <= 0.01})
    return {
        "walls_ok": all(d["ok"] for d in wall_deltas) if wall_deltas else False,
        "ceiling_spread_ok": all(d["ok"] for d in ceiling_spreads) if ceiling_spreads else False,
        "walls": wall_deltas,
        "ceilings": ceiling_spreads,
    }


def _sheet_frame_to_tape(plan: dict, gt: dict) -> dict:
    rooms = plan.get("rooms") or []
    gts = gt.get("rooms") or []
    if len(rooms) < 2 or len(gts) < 2:
        return plan
    pc = np.stack([np.asarray(r["polygon_m"], dtype=float).mean(axis=0) for r in rooms])
    gc = np.stack([np.asarray(g["centroid_m"], dtype=float) for g in gts])
    best = None
    for reflect in (1.0, -1.0):
        for turn in range(4):
            ang = turn * np.pi / 2
            c, s = np.cos(ang), np.sin(ang)
            rot = np.array([[c, -s], [s, c]])
            src = pc.copy()
            src[:, 0] *= reflect
            src = src @ rot.T
            for perm in itertools.permutations(range(len(src)), len(gc)):
                chosen = src[list(perm)]
                trans = (gc - chosen).mean(axis=0)
                err = float(np.linalg.norm(chosen + trans - gc, axis=1).mean())
                if best is None or err < best[0]:
                    best = (err, reflect, rot, trans)
    if best is None or best[0] > 1.25:
        return plan
    _, reflect, rot, trans = best
    out = json.loads(json.dumps(plan))
    for room in out["rooms"]:
        poly = np.asarray(room["polygon_m"], dtype=float)
        poly[:, 0] *= reflect
        poly = poly @ rot.T + trans
        room["polygon_m"] = np.round(poly, 4).tolist()
    return out


def _match_rooms(plan: dict, gt: dict) -> list[tuple[dict, dict]]:
    used = set()
    pairs = []
    for g in gt["rooms"]:
        gc = np.asarray(g["centroid_m"], dtype=float)
        best, bi = 1e9, None
        for i, room in enumerate(plan.get("rooms", [])):
            if i in used:
                continue
            poly = np.asarray(room["polygon_m"], dtype=float)
            dist = float(np.linalg.norm(poly.mean(axis=0) - gc))
            if dist < best:
                best, bi = dist, i
        if bi is not None and best < 2.5:
            used.add(bi)
            pairs.append((plan["rooms"][bi], g))
    return pairs


def _score_walls(pairs, relative: float, absolute: float) -> dict:
    rows = []
    for est, g in pairs:
        el = sorted(w["length_m"]["value"] for w in est["walls"])
        gl = sorted(w["length_m"] for w in g["walls"])
        # Greedy: each tape wall takes the nearest unused estimate.
        pool = el[:]
        for length in gl:
            if not pool:
                rows.append({"room": g["name"], "gt_m": length, "est_m": None, "ok": False})
                continue
            j = int(np.argmin([abs(v - length) for v in pool]))
            est_v = pool.pop(j)
            limit = max(absolute, relative * length)
            rows.append(
                {
                    "room": g["name"],
                    "gt_m": length,
                    "est_m": round(est_v, 4),
                    "err_m": round(abs(est_v - length), 4),
                    "limit_m": round(limit, 4),
                    "ok": abs(est_v - length) <= limit + 1e-9,
                }
            )
    return {"all_within": all(r["ok"] for r in rows) if rows else False, "rows": rows}


def _score_openings(pairs, gt: dict) -> dict:
    matched = []
    misses = []
    phantoms = []
    for est, g in pairs:
        pool = [float(op["width_m"]["value"]) for wall in est["walls"] for op in wall.get("openings") or []]
        for tape in [float(op["width_m"]) for wall in g["walls"] for op in wall.get("openings") or []]:
            if not pool:
                misses.append({"room": g["name"], "gt_m": tape})
                continue
            j = int(np.argmin([abs(v - tape) for v in pool]))
            if abs(pool[j] - tape) > 0.08:
                misses.append({"room": g["name"], "gt_m": tape})
                continue
            est_v = pool.pop(j)
            err = abs(est_v - tape)
            matched.append(
                {"room": g["name"], "gt_m": tape, "est_m": round(est_v, 4), "err_m": round(err, 4), "ok": err <= 0.02}
            )
        for leftover in pool:
            phantoms.append({"room": g["name"], "est_m": round(leftover, 4)})
    counted = len(matched) + len(misses) + len(phantoms)
    within = sum(1 for m in matched if m["ok"])
    return {"counted": counted, "within_gate": within, "matched": matched, "misses": misses, "phantoms": phantoms}


def _score_ceilings(pairs) -> dict:
    rows = []
    for est, g in pairs:
        value = float(est["ceiling_height_m"]["value"])
        tape = float(g["ceiling_height_m"])
        err = abs(value - tape)
        ci = est["ceiling_height_m"]["ci95"]
        rows.append(
            {
                "room": g["name"],
                "gt_m": tape,
                "est_m": round(value, 4),
                "err_m": round(err, 4),
                "ok": err <= 0.015,
                "gt_inside_ci": ci[0] - 1e-9 <= tape <= ci[1] + 1e-9,
            }
        )
    worst = max((r["err_m"] for r in rows), default=None)
    return {"n": len(rows), "worst_abs_m": worst, "rows": rows}


def _score_areas(pairs) -> dict:
    rows = []
    for est, g in pairs:
        value = float(est["floor_area_m2"]["value"])
        tape = float(g["floor_area_m2"])
        rel = abs(value - tape) / tape if tape else 0.0
        rows.append({"room": g["name"], "gt_m2": tape, "est_m2": round(value, 4), "rel": round(rel, 4)})
    return {"rows": rows}


def _calibration(pairs, gt: dict) -> dict:
    """Tape value inside the reported 95 percent interval."""
    checks = []
    for est, g in pairs:
        checks.append(_inside(g["name"], "ceiling", g["ceiling_height_m"], est["ceiling_height_m"]))
        checks.append(_inside(g["name"], "area", g["floor_area_m2"], est["floor_area_m2"]))
        for wall in est["walls"]:
            # Wall calibration is checked against the nearest tape length.
            tape_lengths = [w["length_m"] for w in g["walls"]]
            value = wall["length_m"]["value"]
            tape = min(tape_lengths, key=lambda t: abs(t - value))
            checks.append(_inside(g["name"], "wall", tape, wall["length_m"]))
        for wall in est["walls"]:
            for op in wall.get("openings") or []:
                tape_ops = [o["width_m"] for w in g["walls"] for o in w.get("openings") or []]
                if not tape_ops:
                    continue
                value = op["width_m"]["value"]
                tape = min(tape_ops, key=lambda t: abs(t - value))
                if abs(tape - value) > 0.08:
                    continue
                checks.append(_inside(g["name"], "opening", tape, op["width_m"]))
    return {"all_inside": all(c["ok"] for c in checks) if checks else False, "n": len(checks), "failed": [c for c in checks if not c["ok"]]}


def _inside(room: str, kind: str, tape: float, measurement: dict) -> dict:
    lo, hi = measurement["ci95"]
    ok = lo - 1e-9 <= tape <= hi + 1e-9
    return {"room": room, "kind": kind, "gt": tape, "ci95": [lo, hi], "ok": ok}


def _adjacency_ok(plan: dict, pairs: list[tuple[dict, dict]], gt: dict) -> bool:
    id_to_name = {est["id"]: g["name"] for est, g in pairs}
    got = set()
    for a, b in (plan.get("stitch") or {}).get("adjacency") or []:
        if a in id_to_name and b in id_to_name:
            got.add(tuple(sorted((id_to_name[a], id_to_name[b]))))
    want = {tuple(sorted(p)) for p in gt.get("adjacency") or []}
    return got == want


def _foot_err(plan: dict, gt: dict) -> float:
    est = float((plan.get("stitch") or {}).get("footprint_area_m2", {}).get("value") or 0.0)
    tape = float(gt["footprint_area_m2"])
    return abs(est - tape) / tape if tape else 1.0

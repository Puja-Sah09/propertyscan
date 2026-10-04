"""Build the published plan document from room geometry."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from propertyscan.damage import Damage, Flag, ScopeItem
from propertyscan.geometry import Opening, RoomGeom, adjacency_from_openings, polygon_area

_SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "plan.schema.json"


def measurement(value: float, half: float, unit: str) -> dict:
    return {
        "value": round(float(value), 4),
        "ci95": [round(float(value - half), 4), round(float(value + half), 4)],
        "unit": unit,
    }


def _opening_json(op: Opening, half: float) -> dict:
    return {
        "id": f"op-{op.line_index}-{int(op.midpoint_t*100)}",
        "kind": op.kind,
        "width_m": measurement(op.width, half, "m"),
    }


def build_plan(
    rooms: list[RoomGeom],
    tier: str,
    damages: list[Damage],
    flags: list[Flag],
    scope: list[ScopeItem],
    drift: dict,
    device: str,
    capture_id: str,
    length_half: float,
    opening_half: float,
    ceiling_half: float,
    names: list[str] | None = None,
    length_rel: float = 0.0,
    ci_scale: float = 1.0,
    area_rel: float = 0.02,
) -> dict:
    room_json = []
    for i, room in enumerate(rooms):
        walls = []
        lengths = room.wall_lengths()
        poly = room.polygon
        for w, length in enumerate(lengths):
            a = poly[w]
            b = poly[(w + 1) % len(poly)]
            # Openings whose midpoint lies near this edge.
            ops = []
            for op in room.openings:
                if op.line_index < 0 or op.line_index >= len(room.lines):
                    continue
                ln = room.lines[op.line_index]
                mid = ln.point_at(op.midpoint_t)
                # Distance from midpoint to this edge.
                ab = b - a
                lab = float(np_norm(ab)) + 1e-9
                t = float(np_dot(mid - a, ab) / (lab * lab))
                if t < -0.05 or t > 1.05:
                    continue
                foot = a + ab * min(max(t, 0), 1)
                if np_norm(mid - foot) > 0.35:
                    continue
                ops.append(_opening_json(op, opening_half))
            walls.append(
                {
                    "id": f"room-{i+1}:wall-{w+1}",
                    "length_m": measurement(length, ci_scale * max(length_half, length_rel * length), "m"),
                    "openings": ops,
                }
            )
        room_json.append(
            {
                "id": f"room-{i+1}",
                "name": names[i] if names and i < len(names) else f"room-{i+1}",
                "ceiling_height_m": measurement(room.ceiling_height, ci_scale * ceiling_half, "m"),
                "floor_area_m2": measurement(room.area, ci_scale * max(0.05, area_rel * room.area), "m2"),
                "polygon_m": [[round(float(x), 4), round(float(y), 4)] for x, y in poly],
                "walls": walls,
            }
        )
    pairs = adjacency_from_openings(rooms)
    # Overlap of the snapped polygons. Shared edges are not area.
    overlap = _pairwise_overlap(rooms)
    plan = {
        "schema_version": "1.0.0",
        "capture_id": capture_id,
        "tier": tier,
        "device": {"model": device, "tier": tier},
        "rooms": room_json,
        "stitch": {
            "adjacency": [[f"room-{a+1}", f"room-{b+1}"] for a, b in pairs],
            "room_count": len(rooms),
            "footprint_area_m2": measurement(sum(r.area for r in rooms), 0.15, "m2"),
            "overlaps": overlap > 0.05,
            "overlap_area_m2": round(overlap, 4),
        },
        "damages": [
            {
                "id": d.id,
                "room_id": d.room_id,
                "surface_id": d.surface_id,
                "class": d.klass,
                "area_m2": d.area_m2,
                "length_m": d.length_m,
                "width_m": d.width_m,
                "centroid_xyz": d.centroid_xyz,
                "max_residual_m": d.max_residual_m,
            }
            for d in damages
        ],
        "concealed_flags": [
            {"rule_id": f.rule_id, "surface_id": f.surface_id, "damage_id": f.damage_id, "evidence": f.evidence}
            for f in flags
        ],
        "scope_items": [
            {
                "id": s.id,
                "surface_id": s.surface_id,
                "action_code": s.action_code,
                "action": s.action,
                "qty": s.qty,
                "unit": s.unit,
                "damage_ids": s.damage_ids,
            }
            for s in scope
        ],
        "drift": drift,
    }
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(plan, schema)
    return plan


def np_norm(v) -> float:
    return float((v[0] ** 2 + v[1] ** 2) ** 0.5)


def np_dot(a, b) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _pairwise_overlap(rooms: list[RoomGeom]) -> float:
    # Raster union vs sum. Overlap = sum - union.
    if len(rooms) < 2:
        return 0.0
    polys = [r.polygon for r in rooms]
    total = sum(polygon_area(p) for p in polys)
    # Sample. Good enough at 5 cm for a "does this overlap" check.
    import numpy as np

    pts = np.vstack(polys)
    minxy = pts.min(0) - 0.1
    maxxy = pts.max(0) + 0.1
    res = 0.05
    xs = np.arange(minxy[0], maxxy[0], res)
    ys = np.arange(minxy[1], maxxy[1], res)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    uu, vv = np.meshgrid(xs, ys)
    stack = np.stack([uu, vv], -1).reshape(-1, 2)
    from propertyscan.geometry import point_in_poly

    count = np.zeros(len(stack), dtype=np.int16)
    for poly in polys:
        for i, p in enumerate(stack):
            if point_in_poly(p, poly):
                count[i] += 1
    both = int((count >= 2).sum())
    return both * res * res

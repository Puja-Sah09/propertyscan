"""Damage, concealed-damage rules, and scope lines.

Rules are explicit and named. A flag says which rule fired and the evidence,
which is the contract. These are hypotheses for a human, not a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from propertyscan.geometry import Line2D, RoomGeom


@dataclass
class Damage:
    id: str
    room_id: str
    surface_id: str
    klass: str
    area_m2: float | None
    length_m: float | None
    width_m: float | None
    centroid_xyz: list[float]
    max_residual_m: float


@dataclass
class Flag:
    rule_id: str
    surface_id: str
    damage_id: str
    evidence: str


@dataclass
class ScopeItem:
    id: str
    surface_id: str
    action_code: str
    action: str
    qty: float
    unit: str
    damage_ids: list[str]


_SCOPE = {
    "crack": ("CR-SEAL-PAINT", "Seal crack, spot-prime, and repaint", "m"),
    "water_stain": ("ST-BLOCK-PAINT", "Stain-block primer and two finish coats", "m2"),
    "hole": ("PT-PATCH", "Patch substrate, tape, skim, and paint", "m2"),
}


def _wall_points(xyz: np.ndarray, rgb: np.ndarray, ln: Line2D):
    dist = (xyz[:, :2] - ln.origin) @ ln.normal
    t = (xyz[:, :2] - ln.origin) @ ln.direction
    on = (np.abs(dist) < 0.04) & (t >= ln.t_min - 0.02) & (t <= ln.t_max + 0.02)
    return xyz[on], rgb[on], t[on], dist[on]


def detect_damage(xyz: np.ndarray, rgb: np.ndarray, rooms: list[RoomGeom]) -> tuple[list[Damage], list[Flag], list[ScopeItem]]:
    damages: list[Damage] = []
    flags: list[Flag] = []
    scope: list[ScopeItem] = []
    n = 0
    for ri, room in enumerate(rooms):
        room_id = f"room-{ri+1}"
        for wi, ln in enumerate(room.lines):
            surface_id = f"{room_id}:wall-{wi+1}"
            pts, col, tt, signed = _wall_points(xyz, rgb, ln)
            if len(pts) < 40:
                continue
            found = _classify_surface(pts, col, tt, signed, room, room_id, surface_id, n)
            for dmg in found:
                n += 1
                damages.append(dmg)
                flags.extend(_rules(dmg, room))
                code, text, unit = _SCOPE[dmg.klass]
                qty = dmg.length_m if unit == "m" else (dmg.area_m2 or 0.0)
                scope.append(
                    ScopeItem(
                        id=f"SC-{len(scope)+1:03d}",
                        surface_id=surface_id,
                        action_code=code,
                        action=text,
                        qty=round(float(qty), 3),
                        unit=unit,
                        damage_ids=[dmg.id],
                    )
                )
    return damages, flags, scope


def _classify_surface(pts, col, tt, signed, room, room_id, surface_id, n0) -> list[Damage]:
    resid = signed - np.median(signed)
    # Orthographic cells, 2 cm.
    t0, z0 = float(tt.min()), float(pts[:, 2].min())
    ti = np.clip(((tt - t0) / 0.02).astype(int), 0, 4000)
    zi = np.clip(((pts[:, 2] - z0) / 0.02).astype(int), 0, 4000)
    # Geometric recess: points standing off the fitted plane.
    deep = np.abs(resid) > 0.012
    # A real wall is a plane. If a third of the points stand off it, the
    # surface is furniture, a smear, or a corner — not a field of holes.
    if deep.mean() > 0.30:
        deep = np.zeros(len(pts), dtype=bool)
    stains = _stain_mask(col)
    out: list[Damage] = []
    if deep.sum() >= 12:
        clusters = _components(ti[deep], zi[deep])
        for cells in clusters:
            if len(cells) < 8:
                continue
            ts = t0 + 0.02 * cells[:, 0]
            zs = z0 + 0.02 * cells[:, 1]
            length = float(zs.max() - zs.min())
            width = float(ts.max() - ts.min())
            if length < 0.25 or width > 0.12:
                # A wide recess is a hole, not a crack. A recess the size of
                # the wall is the wall not being flat.
                if width * length < 0.01 or width > 0.55 or length > 0.80:
                    continue
                klass = "hole"
            else:
                klass = "crack"
            if klass == "hole" and width < 0.08 and length > 0.3:
                klass = "crack"
            area = float(len(cells)) * 0.02 * 0.02
            if klass == "hole" and area > 0.25:
                continue
            cell_set = {(int(a), int(b)) for a, b in cells}
            member = np.array([(int(a), int(b)) in cell_set for a, b in zip(ti[deep], zi[deep])])
            if int(member.sum()) == 0:
                continue
            if klass == "hole" and float(np.median(np.abs(resid[deep][member]))) < 0.025:
                continue
            cxyz = [
                float(np.median(pts[deep, 0][member])),
                float(np.median(pts[deep, 1][member])),
                float(np.median(zs)),
            ]
            out.append(
                Damage(
                    id=f"DMG-{n0+len(out)+1:03d}",
                    room_id=room_id,
                    surface_id=surface_id,
                    klass=klass,
                    area_m2=round(area, 3),
                    length_m=round(length, 3),
                    width_m=round(max(width, 0.01), 3),
                    centroid_xyz=[round(v, 3) for v in cxyz],
                    max_residual_m=round(float(np.max(np.abs(resid[deep]))), 4),
                )
            )
    if stains.sum() >= 20 and not deep[stains].mean() > 0.5:
        # Appearance only: the plane is intact, the colour is not.
        if stains.mean() > 0.20:
            return out
        sti = np.clip(((tt[stains] - t0) / 0.02).astype(int), 0, 4000)
        szi = np.clip(((pts[stains, 2] - z0) / 0.02).astype(int), 0, 4000)
        for cells in _components(sti, szi):
            if len(cells) < 12:
                continue
            ts = t0 + 0.02 * cells[:, 0]
            zs = z0 + 0.02 * cells[:, 1]
            area = float(len(cells)) * 0.02 * 0.02
            span_t = float(ts.max() - ts.min())
            span_z = float(zs.max() - zs.min())
            # Scattered dark pixels make a huge box. A stain is a compact patch.
            if area < 0.02 or area > 0.80 or span_t > 1.2 or span_z > 1.0:
                continue
            cxyz = [float(np.median(pts[stains, 0])), float(np.median(pts[stains, 1])), float(np.median(zs))]
            out.append(
                Damage(
                    id=f"DMG-{n0+len(out)+1:03d}",
                    room_id=room_id,
                    surface_id=surface_id,
                    klass="water_stain",
                    area_m2=round(area, 3),
                    length_m=None,
                    width_m=None,
                    centroid_xyz=[round(v, 3) for v in cxyz],
                    max_residual_m=round(float(np.median(np.abs(resid[stains]))), 4),
                )
            )
    return out


def _stain_mask(col: np.ndarray) -> np.ndarray:
    # BGR. A water stain is darker and warmer than the surrounding paint.
    b = col[:, 0].astype(np.float32)
    g = col[:, 1].astype(np.float32)
    r = col[:, 2].astype(np.float32)
    med_r, med_g, med_b = np.median(r), np.median(g), np.median(b)
    darker = (r + g + b) < (med_r + med_g + med_b) - 80
    warmer = (r - b) > (med_r - med_b) + 25
    return darker & warmer & (r > g) & (g > b)


def _components(ti: np.ndarray, zi: np.ndarray) -> list[np.ndarray]:
    if len(ti) == 0:
        return []
    cells = np.unique(np.stack([ti, zi], 1), axis=0)
    # Union-find on a hash grid.
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (a, b), i in key.items():
        for nb in ((a + 1, b), (a, b + 1)):
            j = key.get(nb)
            if j is not None:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(cells)):
        groups.setdefault(find(i), []).append(i)
    return [cells[idx] for idx in groups.values()]


def _rules(dmg: Damage, room: RoomGeom) -> list[Flag]:
    flags = []
    if dmg.klass == "water_stain" and dmg.max_residual_m < 0.008:
        flags.append(
            Flag(
                rule_id="R_MOISTURE_BEHIND_FINISH",
                surface_id=dmg.surface_id,
                damage_id=dmg.id,
                evidence=(
                    "Colour stain with plane residual under 8 mm. The finish moved; "
                    "the substrate did not. Treat as moisture behind the paint until opened."
                ),
            )
        )
    storey = room.ceiling_z - room.floor_z
    if dmg.klass == "water_stain" and storey >= 1.9 and dmg.centroid_xyz[2] > room.ceiling_z - 0.45:
        flags.append(
            Flag(
                rule_id="R_CEILING_ADJACENT_MOISTURE",
                surface_id=dmg.surface_id,
                damage_id=dmg.id,
                evidence="Stain centroid is within 0.45 m of the ceiling. Check the assembly above before painting.",
            )
        )
    if dmg.klass == "crack" and (dmg.length_m or 0) >= 0.4 and dmg.max_residual_m >= 0.008:
        flags.append(
            Flag(
                rule_id="R_SUBSTRATE_SPLIT",
                surface_id=dmg.surface_id,
                damage_id=dmg.id,
                evidence=(
                    f"Crack length {dmg.length_m:.2f} m and plane residual {dmg.max_residual_m*1000:.0f} mm. "
                    "The split is in the substrate, not a hairline in the paint."
                ),
            )
        )
    return flags

"""Pinhole ray caster for benchmark captures.

This is the only place that knows the synthetic building. It writes depth and
RGB. The estimator reads those images and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from benchmarks.floorplan_spec import CRACK, DOORS, LETTER_H, LETTER_W, ROOMS, SOFA, STAIN, sheet_rect


@dataclass
class Wall:
    p0: np.ndarray
    p1: np.ndarray
    z0: float
    z1: float
    color: np.ndarray
    room: str
    side: str

    def normal_away_from(self, interior_xy: np.ndarray) -> np.ndarray:
        d = self.p1 - self.p0
        n = np.array([-d[1], d[0]], dtype=float)
        n /= np.linalg.norm(n) + 1e-12
        mid = 0.5 * (self.p0 + self.p1)
        if np.dot(interior_xy - mid, n) > 0:
            n = -n
        return n


def build_walls() -> list[Wall]:
    """One quad per solid piece of wall. Door leaves a gap up to the header."""
    palette = {
        "north": np.array([186, 176, 166], np.float32),
        "south": np.array([176, 170, 160], np.float32),
        "east": np.array([168, 164, 158], np.float32),
        "west": np.array([190, 182, 172], np.float32),
    }
    walls: list[Wall] = []
    seen = set()
    for room in ROOMS:
        x0, y0, x1, y1 = room["x0"], room["y0"], room["x1"], room["y1"]
        edges = [
            ("south", np.array([x0, y0]), np.array([x1, y0])),
            ("east", np.array([x1, y0]), np.array([x1, y1])),
            ("north", np.array([x1, y1]), np.array([x0, y1])),
            ("west", np.array([x0, y1]), np.array([x0, y0])),
        ]
        interior = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        for side, a, b in edges:
            key = tuple(np.round(np.concatenate([np.minimum(a, b), np.maximum(a, b)]), 3))
            if key in seen:
                continue
            seen.add(key)
            spans = _solid_spans(a, b, room["ceiling"])
            for z0, z1, u0, u1 in spans:
                p0 = a + (b - a) * u0
                p1 = a + (b - a) * u1
                if np.linalg.norm(p1 - p0) < 0.04:
                    continue
                walls.append(Wall(p0, p1, z0, z1, palette[side].copy(), room["name"], side))
            _ = interior
    return walls


def _solid_spans(a: np.ndarray, b: np.ndarray, ceiling: float) -> list[tuple[float, float, float, float]]:
    """Return (z0, z1, u0, u1) pieces in the edge parameter u in [0, 1]."""
    length = np.linalg.norm(b - a) + 1e-12
    direction = (b - a) / length
    holes = []
    for door in DOORS:
        if door["axis"] == "x":
            # Wall is vertical in plan (constant x) and the edge must sit on that x.
            if abs(a[0] - door["at"]) > 1e-6 or abs(b[0] - door["at"]) > 1e-6:
                continue
            varying = a[1] + (b[1] - a[1]) 
            # Parameter along the edge for door u0/u1 which are y coordinates.
            def u_of(y):
                if abs(b[1] - a[1]) < 1e-9:
                    return None
                return (y - a[1]) / (b[1] - a[1])
            ua, ub = u_of(door["u0"]), u_of(door["u1"])
        else:
            if abs(a[1] - door["at"]) > 1e-6 or abs(b[1] - door["at"]) > 1e-6:
                continue
            def u_of(x):
                if abs(b[0] - a[0]) < 1e-9:
                    return None
                return (x - a[0]) / (b[0] - a[0])
            ua, ub = u_of(door["u0"]), u_of(door["u1"])
        if ua is None or ub is None:
            continue
        lo, hi = sorted((ua, ub))
        lo, hi = max(0.0, lo), min(1.0, hi)
        if hi - lo > 0.01:
            holes.append((lo, hi, door["height"]))
    holes.sort()
    spans = []
    cursor = 0.0
    for lo, hi, height in holes:
        if lo > cursor + 0.01:
            spans.append((0.0, ceiling, cursor, lo))
        # Header above the door.
        if height < ceiling - 0.02:
            spans.append((height, ceiling, lo, hi))
        cursor = max(cursor, hi)
    if cursor < 0.99:
        spans.append((0.0, ceiling, cursor, 1.0))
    return spans


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([right, down, forward])
    t[:3, 3] = eye
    return t


def make_intrinsics(width: int, height: int, fov_deg: float = 62.0) -> np.ndarray:
    fx = (width / 2) / math.tan(math.radians(fov_deg) / 2)
    fy = fx
    return np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], dtype=float)


def render(r: np.ndarray, t: np.ndarray, k: np.ndarray, width: int, height: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return depth in meters (0 = miss) and BGR uint8."""
    walls = build_walls()
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    us = np.arange(width, dtype=np.float32)
    vs = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    dirs_c = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1).astype(np.float32)
    dirs_w = dirs_c @ r.T.astype(np.float32)
    o = t.astype(np.float32)
    s_best = np.full((height, width), np.inf, dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    sid = np.full((height, width), -1, dtype=np.int16)

    def take(s, valid, color, surface):
        better = valid & np.isfinite(s) & (s > 0.05) & (s < s_best)
        if not np.any(better):
            return
        s_best[better] = s[better]
        rgb[better] = color[better] if np.ndim(color) == 3 else color
        sid[better] = surface

    # Floors, z = 0, colored per room so a photo can stop at a wall, not at a room.
    dwz = dirs_w[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_floor = np.where(dwz < -1e-4, -o[2] / dwz, np.inf)
    hit = o + s_floor[..., None] * dirs_w
    for ri, room in enumerate(ROOMS):
        inside = (
            (hit[:, :, 0] >= room["x0"])
            & (hit[:, :, 0] <= room["x1"])
            & (hit[:, :, 1] >= room["y0"])
            & (hit[:, :, 1] <= room["y1"])
        )
        # Soft texture. The wavelength is not a scale cue; the letter sheet is.
        # Contrast is for tracking. Nothing in the estimator reads this wavelength.
        tex = 22.0 * np.sin(11.0 * hit[:, :, 0]) * np.cos(8.0 * hit[:, :, 1])
        tex = tex + 14.0 * np.sin(37.0 * hit[:, :, 0] + 29.0 * hit[:, :, 1])
        col = np.zeros_like(rgb)
        base = np.array([142, 132, 118], np.float32) + tex[..., None]
        col[:] = np.clip(base, 0, 255).astype(np.uint8)
        take(s_floor, inside & np.isfinite(s_floor), col, 100 + ri)
        # Ceiling of this room.
        cz = np.float32(room["ceiling"])
        s_ceil = np.where(dwz > 1e-4, (cz - o[2]) / dwz, np.inf)
        hitc = o + s_ceil[..., None] * dirs_w
        inside_c = (
            (hitc[:, :, 0] >= room["x0"])
            & (hitc[:, :, 0] <= room["x1"])
            & (hitc[:, :, 1] >= room["y0"])
            & (hitc[:, :, 1] <= room["y1"])
        )
        take(s_ceil, inside_c & np.isfinite(s_ceil), (210, 208, 204), 200 + ri)

    for wi, wall in enumerate(walls):
        d = wall.p1 - wall.p0
        length = float(np.linalg.norm(d)) + 1e-8
        tangent = d / length
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        # Plane through p0.
        n3 = np.array([normal[0], normal[1], 0], dtype=np.float32)
        denom = dirs_w @ n3
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(denom) > 1e-5, ((wall.p0.astype(np.float32) - o[:2]) @ normal) / denom, np.inf)
        hitw = o + s[..., None] * dirs_w
        along = (hitw[:, :, 0] - wall.p0[0]) * tangent[0] + (hitw[:, :, 1] - wall.p0[1]) * tangent[1]
        zz = hitw[:, :, 2]
        valid = (along >= -0.002) & (along <= length + 0.002) & (zz >= wall.z0 - 0.002) & (zz <= wall.z1 + 0.002)
        # Crack recess on the living north wall: push the plane back.
        if wall.room == "living" and wall.side == "north":
            in_crack = (hitw[:, :, 0] >= CRACK["x0"]) & (hitw[:, :, 0] <= CRACK["x1"]) & (zz >= CRACK["z0"]) & (zz <= CRACK["z1"])
            # Recompute s against the recessed plane. n points either way; offset along +Y
            # because the living room sits at smaller y and a recess increases y.
            p_rec = wall.p0.astype(np.float32) + np.array([0.0, CRACK["recess_m"]], dtype=np.float32)
            s_rec = np.where(np.abs(denom) > 1e-5, ((p_rec - o[:2]) @ normal) / denom, s)
            s = np.where(in_crack, s_rec, s)
            hitw = o + s[..., None] * dirs_w
            zz = hitw[:, :, 2]
        color = np.broadcast_to(wall.color.astype(np.uint8), rgb.shape).copy()
        if wall.room == "living" and wall.side == "north":
            stain = (
                (hitw[:, :, 0] >= STAIN["x0"])
                & (hitw[:, :, 0] <= STAIN["x1"])
                & (zz >= STAIN["z0"])
                & (zz <= STAIN["z1"])
            )
            color[stain] = (40, 70, 110)  # BGR brown
            crack = (
                (hitw[:, :, 0] >= CRACK["x0"] - 0.01)
                & (hitw[:, :, 0] <= CRACK["x1"] + 0.01)
                & (zz >= CRACK["z0"])
                & (zz <= CRACK["z1"])
            )
            color[crack] = (25, 25, 25)
        take(s.astype(np.float32), valid & np.isfinite(s), color, wi)

    # Sofa. Slab test in world, parameter along the camera ray whose z-component in
    # camera frame is 1, so s equals depth.
    bmin = np.array([SOFA["x0"], SOFA["y0"], 0.0], np.float32)
    bmax = np.array([SOFA["x1"], SOFA["y1"], SOFA["z1"]], np.float32)
    inv = 1.0 / np.where(np.abs(dirs_w) < 1e-6, np.float32(1e-6), dirs_w)
    t0 = (bmin - o) * inv
    t1 = (bmax - o) * inv
    tsm = np.minimum(t0, t1)
    tbg = np.maximum(t0, t1)
    tmin = tsm.max(axis=-1)
    tmax = tbg.min(axis=-1)
    hit_box = (tmax >= np.maximum(tmin, 0)) & (tmax > 0)
    s_box = np.where(tmin > 0.05, tmin, tmax).astype(np.float32)
    take(s_box, hit_box, (90, 110, 140), 900)

    # Letter sheets sit on the floor. Recolor the floor hit, they do not change depth.
    hitf = o + s_best[..., None] * dirs_w
    for room in ROOMS:
        x0, y0, x1, y1 = sheet_rect(room)
        on = (
            (sid >= 100)
            & (sid < 200)
            & (hitf[:, :, 0] >= x0)
            & (hitf[:, :, 0] <= x1)
            & (hitf[:, :, 1] >= y0)
            & (hitf[:, :, 1] <= y1)
        )
        border = on & (
            (hitf[:, :, 0] < x0 + 0.012)
            | (hitf[:, :, 0] > x1 - 0.012)
            | (hitf[:, :, 1] < y0 + 0.012)
            | (hitf[:, :, 1] > y1 - 0.012)
        )
        rgb[on] = (245, 245, 245)
        rgb[border] = (15, 15, 15)
        # A mark inside the white field, off centre. A rectangle is unchanged
        # by a 180 degree turn; the mark is not, and it sits clear of the border
        # so a flipped pose cannot land on the border and look correct.
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        mark = (
            on
            & (hitf[:, :, 0] > cx + 0.045)
            & (hitf[:, :, 0] < cx + 0.070)
            & (hitf[:, :, 1] > cy + 0.030)
            & (hitf[:, :, 1] < cy + 0.055)
        )
        rgb[mark] = (15, 15, 15)

    depth = s_best.copy()
    depth[~np.isfinite(depth)] = 0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.008, size=depth.shape).astype(np.float32)
    drop = rng.random(depth.shape) < 0.004
    depth = np.where(depth > 0, np.clip(depth + noise, 0.05, 20), 0).astype(np.float32)
    depth[drop] = 0
    return depth, rgb


def drift_pose(true_T: np.ndarray, yaw_bias: float, xy_bias: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw_bias), math.sin(yaw_bias)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    out = np.eye(4)
    out[:3, :3] = rz @ true_T[:3, :3]
    out[:3, 3] = true_T[:3, 3] + np.array([xy_bias[0], xy_bias[1], 0.0])
    return out


def capture_shots():
    """Walk used by the benchmark. Same path the written protocol describes.

    The phone stays under a metre off each wall and travels to within 0.25 m of
    every corner. Stopping short leaves a blind spot the plan cannot close.
    """
    shots = []

    def wall_pass(x0, y0, x1, y1, outward):
        start = np.array([x0, y0], float)
        end = np.array([x1, y1], float)
        length = float(np.linalg.norm(end - start))
        n = max(2, int(length / 0.55) + 1)
        ow = np.array(outward, float)
        for s in np.linspace(0, 1, n):
            xy = start + (end - start) * s
            eye = np.array([xy[0], xy[1], 1.45])
            shots.append((eye, eye + np.array([ow[0], ow[1], -0.42])))

    for room in ROOMS:
        x0, y0, x1, y1 = room["x0"], room["y0"], room["x1"], room["y1"]
        standoff = min(0.95, 0.34 * min(x1 - x0, y1 - y0))
        margin = 0.22
        wall_pass(x0 + margin, y0 + standoff, x1 - margin, y0 + standoff, (0, -1))
        wall_pass(x1 - standoff, y0 + margin, x1 - standoff, y1 - margin, (1, 0))
        wall_pass(x1 - margin, y1 - standoff, x0 + margin, y1 - standoff, (0, 1))
        wall_pass(x0 + standoff, y1 - margin, x0 + standoff, y0 + margin, (-1, 0))
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 1.45])
        shots.append((center, center + np.array([0.15, 0.10, 1.25])))

    def transit(a, b, n):
        a = np.array(a, float)
        b = np.array(b, float)
        for s in np.linspace(0, 1, n):
            eye = a + (b - a) * s
            shots.append((eye, eye + (b - a) + np.array([0.0, 0.0, -0.30])))

    transit((3.2, 2.40, 1.45), (5.7, 2.40, 1.45), 6)
    transit((5.55, 2.85, 1.45), (7.8, 2.85, 1.45), 5)
    transit((5.60, 3.8, 1.45), (5.60, 6.8, 1.45), 6)
    transit((5.60, 6.2, 1.45), (3.2, 2.40, 1.45), 6)
    return shots


def _inside(room, xy, margin=0.28):
    return np.array(
        [
            min(max(xy[0], room["x0"] + margin), room["x1"] - margin),
            min(max(xy[1], room["y0"] + margin), room["y1"] - margin),
        ]
    )


def photo_views() -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Stills a person can take by following the written protocol.

    Stand on the far side of the letter sheet from the wall you are measuring
    and aim at the sheet. The sheet stays in frame, and the wall base falls in
    the upper part of a 62 degree view. ``door-to-<room>`` is the same idea
    aimed across the doorway.
    """
    views: dict[str, list] = {}
    centers = {}
    for room in ROOMS:
        x0, y0, x1, y1 = sheet_rect(room)
        centers[room["name"]] = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        cx, cy = centers[room["name"]]
        shots = []
        walls = {
            "wall-south": np.array([cx, room["y0"]]),
            "wall-north": np.array([cx, room["y1"]]),
            "wall-west": np.array([room["x0"], cy]),
            "wall-east": np.array([room["x1"], cy]),
        }
        for stem, wall_xy in walls.items():
            eye, tgt = _aim_across_sheet(room, centers[room["name"]], wall_xy)
            shots.append((stem, eye, tgt))
            # A second standing position, shifted along the wall, so a door in
            # the middle of the wall does not hide the baseboard.
            shift = _along_wall(room, centers[room["name"]], wall_xy)
            eye_b = eye.copy()
            eye_b[:2] = _inside(room, eye[:2] + shift, margin=0.28)
            tgt_b = tgt.copy()
            tgt_b[:2] = tgt[:2] + shift * 0.5
            shots.append((stem + "-b", eye_b, tgt_b))
        views[room["name"]] = shots
    for door in DOORS:
        if door["axis"] == "x":
            mid = np.array([door["at"], 0.5 * (door["u0"] + door["u1"])])
        else:
            mid = np.array([0.5 * (door["u0"] + door["u1"]), door["at"]])
        for src, dst in ((door["a"], door["b"]), (door["b"], door["a"])):
            eye, tgt = _aim_across_sheet(room_by(src), centers[src], mid)
            views[src].append((f"door-to-{dst}", eye, tgt))
    return views


def _along_wall(room, sheet_xy, interest_xy) -> np.ndarray:
    direction = np.asarray(interest_xy, float) - np.asarray(sheet_xy, float)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    perp = np.array([-direction[1], direction[0]])
    return perp * 1.2


def _aim_across_sheet(room, sheet_xy, interest_xy):
    direction = np.asarray(interest_xy, float) - np.asarray(sheet_xy, float)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    # As far back from the sheet as this room allows, opposite the interest.
    eye_xy = _inside(room, np.asarray(sheet_xy, float) - direction * 8.0, margin=0.30)
    eye = np.array([eye_xy[0], eye_xy[1], 1.42])
    # Aim past the sheet, part way to the wall. Aiming at the sheet hides a
    # door wall; aiming at the wall loses the sheet. The point between keeps both.
    aim = np.asarray(sheet_xy, float) + 0.55 * (np.asarray(interest_xy, float) - np.asarray(sheet_xy, float))
    tgt = np.array([aim[0], aim[1], 0.20])
    return eye, tgt


def room_by(name: str) -> dict:
    for room in ROOMS:
        if room["name"] == name:
            return room
    raise KeyError(name)


# Wide lens, so a frame that is aimed at a wall still contains floor to track
# and the ceiling corner above that wall. A 62 degree view aimed at a wall is
# only paint, and the track dies. The benchmark passes this fov to make_intrinsics.
VIDEO_FOV_DEG = 110.0


def video_shots(seed_shift: float = 0.0):
    """A slow walk that keeps the floor in frame.

    The first and last looks are at the living-room sheet. Between them the
    camera faces each wall from far enough away that the wall base, a stretch
    of floor, and the ceiling corner share the frame. The hall is 1.2 m wide,
    so its long walls are seen diagonally rather than from the side.
    """
    shots = []
    # One height for the whole walk. A rise after the sheet lock changes the
    # floor scale, and the plan comes out short.
    eye_z = 1.45
    shift = np.array([seed_shift, -seed_shift, 0.0])
    living = room_by("living")
    x0, y0, x1, y1 = sheet_rect(living)
    sheet = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0])

    def add(eye, tgt):
        eye = np.asarray(eye, float) + shift
        tgt = np.asarray(tgt, float)
        shots.append((eye, tgt))

    # Close enough that the black border is a few pixels, which is what the
    # wide lens needs before the quad will solve.
    eye = np.array([sheet[0], sheet[1] - 0.34, eye_z])
    for ang in np.linspace(-0.2, 0.2, 5):
        add(eye, sheet + np.array([0.10 * math.sin(ang), 0.04, 0.0]))

    def face_wall(room, axis: str, along: np.ndarray):
        """Stand back from one wall and look at its base and ceiling corner."""
        x0, y0, x1, y1 = room["x0"], room["y0"], room["x1"], room["y1"]
        span = (x1 - x0) if axis in ("s", "n") else (y1 - y0)
        depth = (y1 - y0) if axis in ("s", "n") else (x1 - x0)
        back = min(1.75, max(0.85, 0.46 * depth))
        # The living-room sofa sits against the south wall. Standing in it hides
        # the baseboard. Stand on the north side of the sofa instead.
        if room["name"] == "living" and axis == "s":
            back = 2.30
        for s in along:
            t = float(s)
            if axis == "s":
                add((t, y0 + back, eye_z), (t, y0, 1.00))
            elif axis == "n":
                add((t, y1 - back, eye_z), (t, y1, 1.00))
            elif axis == "w":
                add((x0 + back, t, eye_z), (x0, t, 1.00))
            else:
                add((x1 - back, t, eye_z), (x1, t, 1.00))
        return span

    def room_circuit(room):
        x0, y0, x1, y1 = room["x0"], room["y0"], room["x1"], room["y1"]
        # A 1.2 m hall has no standing position that faces a long wall squarely
        # and still sees floor. Look along the hall, then diagonally at the sides.
        if min(x1 - x0, y1 - y0) < 1.5:
            cx = 0.5 * (x0 + x1)
            for s in np.linspace(y0 + 0.55, y1 - 0.7, 7):
                add((cx, s, eye_z), (x0, min(y1 - 0.15, s + 1.15), 1.05))
            for s in np.linspace(y1 - 0.55, y0 + 0.7, 7):
                add((cx, s, eye_z), (x1, max(y0 + 0.15, s - 1.15), 1.05))
            add((cx, y0 + 1.3, eye_z), (cx, y0, 1.05))
            add((cx, y1 - 1.3, eye_z), (cx, y1, 1.05))
            return
        face_wall(room, "s", np.linspace(x0 + 0.45, x1 - 0.45, 5))
        face_wall(room, "e", np.linspace(y0 + 0.45, y1 - 0.45, 4))
        face_wall(room, "n", np.linspace(x1 - 0.45, x0 + 0.45, 5))
        face_wall(room, "w", np.linspace(y1 - 0.45, y0 + 0.45, 4))

    def transit(a, b, n):
        a = np.array(a, float)
        b = np.array(b, float)
        step = b - a
        for s in np.linspace(0.0, 1.0, n):
            eye = a + step * s
            tgt = eye + step
            tgt[2] = 0.85
            add(eye, tgt)

    by = {r["name"]: r for r in ROOMS}
    room_circuit(by["living"])
    # Door living-hall is x=5, y 2.0-2.8. Cross it looking along the walk.
    transit((3.4, 2.40, eye_z), (5.60, 2.40, eye_z), 6)
    room_circuit(by["hall"])
    transit((5.60, 2.85, eye_z), (7.90, 2.85, eye_z), 6)
    room_circuit(by["kitchen"])
    transit((7.90, 2.85, eye_z), (5.60, 2.85, eye_z), 5)
    transit((5.60, 3.60, eye_z), (5.60, 7.20, eye_z), 7)
    room_circuit(by["bedroom"])
    transit((5.60, 7.20, eye_z), (5.60, 2.40, eye_z), 8)
    transit((5.60, 2.40, eye_z), (sheet[0], sheet[1] - 0.55, eye_z), 7)
    eye = np.array([sheet[0] + 0.02, sheet[1] - 0.34, eye_z])
    for ang in np.linspace(-0.12, 0.12, 4):
        add(eye, sheet + np.array([0.06 * math.sin(ang), 0.03, 0.0]))
    # Corners of this walk turn about 90 degrees. Fill the steps so optical
    # flow sees a small yaw, not a snap.
    return _densify_walk(shots, max_step=0.16, max_turn=0.10)


def _densify_walk(shots, max_step: float = 0.20, max_turn: float = 0.18):
    if len(shots) < 2:
        return shots
    out = [shots[0]]

    def yaw(eye, tgt):
        d = np.asarray(tgt[:2], float) - np.asarray(eye[:2], float)
        return math.atan2(float(d[1]), float(d[0]))

    for nxt_eye, nxt_tgt in shots[1:]:
        prev_eye, prev_tgt = out[-1]
        dist = float(np.linalg.norm(np.asarray(nxt_eye[:2]) - np.asarray(prev_eye[:2])))
        dy = abs((yaw(nxt_eye, nxt_tgt) - yaw(prev_eye, prev_tgt) + math.pi) % (2 * math.pi) - math.pi)
        n = max(1, int(math.ceil(dist / max_step)), int(math.ceil(dy / max_turn)))
        for s in np.linspace(0.0, 1.0, n + 1)[1:]:
            eye = (1.0 - s) * np.asarray(prev_eye, float) + s * np.asarray(nxt_eye, float)
            tgt = (1.0 - s) * np.asarray(prev_tgt, float) + s * np.asarray(nxt_tgt, float)
            out.append((eye, tgt))
    return out

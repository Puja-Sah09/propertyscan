"""Photo tier. One folder per room, a letter sheet in every frame, doors by filename.

Stills have no pose prior. Each frame is posed from the sheet, wall bases are
accumulated in that room's sheet frame, and rooms are joined by the
``door-to-<name>`` frames: the opening the camera is looking at is the shared
edge, and the other room is placed on the far side of it.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from propertyscan.geometry import (
    Line2D,
    Opening,
    RoomGeom,
    line_gaps,
    polygon_area,
    polygon_from_lines,
    ransac_lines,
    wrap_angle,
)
from propertyscan.planar import (
    boundary_points,
    image_is_dark,
    innermost_lines,
    intrinsics_matrix,
    measure_ceiling,
    pose_from_sheet,
)


def run_photos(capture: dict) -> dict:
    root = Path(capture["root"])
    meta = capture.get("meta") or {}
    rooms_dir = root / "rooms"
    folders = [p for p in sorted(rooms_dir.iterdir()) if p.is_dir()]
    if not folders:
        raise RuntimeError(f"No room folders under {rooms_dir}")
    sample = next(folders[0].glob("*.jpg"))
    probe = cv2.imread(str(sample))
    k, k_source = intrinsics_matrix(meta, probe.shape[1], probe.shape[0])
    local: dict[str, RoomGeom] = {}
    views: dict[str, list[dict]] = {}
    dark = False
    for folder in folders:
        images = [(p.stem, cv2.imread(str(p))) for p in sorted(folder.glob("*.jpg"))]
        images = [(n, im) for n, im in images if im is not None]
        if any(image_is_dark(im) for _, im in images):
            dark = True
        room, used = _room_from_images(images, k)
        if room is None:
            continue
        local[folder.name] = room
        views[folder.name] = used
    if not local:
        raise RuntimeError("No room could be posed from its letter sheet.")
    placed, links = _stitch(local, views, k)
    method = "sheet_pnp_door_stitch"
    return {
        "rooms": placed,
        "names": [r.label_name for r in placed],
        "xyz": np.zeros((0, 3)),
        "rgb": np.zeros((0, 3), np.uint8),
        "dark": dark,
        "intrinsics_source": k_source,
        "drift": {
            "method": method,
            "loop_closures": int(links),
            "used_raw_poses_as_is": False,
        },
    }


def _room_from_images(images: list[tuple[str, np.ndarray]], k: np.ndarray):
    clouds = []
    ceilings = []
    used = []
    for name, img in images:
        posed = pose_from_sheet(img, k)
        if posed is None:
            continue
        r_wc, t_wc, corners = posed
        pts = boundary_points(img, corners, r_wc, t_wc, k)
        if len(pts) < 10:
            continue
        clouds.append(pts[:, :2])
        used.append({"name": name, "R": r_wc, "t": t_wc, "image": img, "corners": corners})
    if not clouds:
        return None, []
    xy = np.vstack(clouds)
    if len(xy) > 8000:
        xy = xy[:: int(len(xy) / 8000) + 1]
    fitted = ransac_lines(xy, thresh=0.04, min_points=30, min_length=0.7, iterations=250)
    lines = innermost_lines(fitted)
    if lines:
        for view in used:
            ceilings.extend(measure_ceiling(view["image"], view["R"], view["t"], k, lines))
    ceiling = float(np.median(ceilings)) if ceilings else 2.40
    room = _room_from_lines(fitted, xy, ceiling)
    return room, used


def _drop_near_parallel(lines: list[Line2D]) -> list[Line2D] | None:
    """Two walls that barely meet blow the corner out of the building."""
    worst = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            cross = abs(lines[i].direction[0] * lines[j].direction[1] - lines[i].direction[1] * lines[j].direction[0])
            if worst is None or cross < worst[0]:
                drop = i if lines[i].n_points <= lines[j].n_points else j
                worst = (cross, drop)
    if worst is None or worst[0] > 0.25:
        return None
    return [ln for k, ln in enumerate(lines) if k != worst[1]]


def _axis_lines(lines: list[Line2D]) -> list[tuple[int, float, Line2D]]:
    """Axis-aligned walls as (axis, position, line). axis 0 is a vertical wall."""
    kept = []
    for ln in lines:
        major = 0 if abs(ln.direction[0]) >= abs(ln.direction[1]) else 1
        if abs(ln.direction[major]) < 0.96:
            continue
        span = float(ln.t_max - ln.t_min)
        if span < 0.7 or ln.n_points < 40:
            continue
        # Intercept: where the line crosses the axis through the sheet.
        if major == 0:
            t = -ln.origin[0] / ln.direction[0]
            pos = float(ln.origin[1] + t * ln.direction[1])
            axis = 1
        else:
            t = -ln.origin[1] / ln.direction[1]
            pos = float(ln.origin[0] + t * ln.direction[0])
            axis = 0
        if abs(pos) < 0.4 or abs(pos) > 8.0:
            continue
        kept.append((axis, pos, ln))
    return kept


def _pick_side(cands: list[tuple[float, Line2D]], high: bool) -> Line2D | None:
    """Nearest wall with a real share of the points. Farther lines are the next room."""
    side = [(p, ln) for p, ln in cands if (p > 0) == high]
    if not side:
        return None
    best_n = max(ln.n_points for _, ln in side)
    strong = [(p, ln) for p, ln in side if ln.n_points >= 0.55 * best_n]
    strong.sort(key=lambda item: abs(item[0]))
    return strong[0][1]


def _extent_from(lines: list[Line2D], axis: int) -> tuple[float, float] | None:
    """Inner endpoints, and only when the two walls agree.

    One wall that runs out through a door is longer than the other. Those two
    ends do not describe the room, so the extent is refused and the fitted
    line stands.
    """
    if len(lines) < 2:
        return None
    lows, highs = [], []
    for ln in lines:
        a = ln.point_at(ln.t_min)
        b = ln.point_at(ln.t_max)
        lows.append(min(float(a[axis]), float(b[axis])))
        highs.append(max(float(a[axis]), float(b[axis])))
    if max(lows) - min(lows) > 0.45 or max(highs) - min(highs) > 0.45:
        return None
    return float(max(lows)), float(min(highs))


def _mode_band(xy: np.ndarray, axis: int, pos: float | None) -> float | None:
    if pos is None or len(xy) == 0:
        return pos
    coord = xy[:, axis]
    band = coord[np.abs(coord - pos) < 0.16]
    if len(band) < 25:
        return pos
    return float(np.median(band))


def _room_from_lines(lines: list[Line2D], support_xy: np.ndarray, ceiling: float) -> RoomGeom | None:
    """Rectangle in the sheet frame.

    The sheet's long edge is parallel to a wall, so the walls are the sheet
    axes. Each side is the nearest fitted line that still carries most of the
    points in that direction. A line seen through a door is farther out and
    loses. A missing side, usually a wall that is mostly a door, is the end
    of the two walls that were fitted.
    """
    parsed = _axis_lines(lines)
    chosen: dict[tuple[int, bool], Line2D] = {}
    for axis in (0, 1):
        group = [(p, ln) for ax, p, ln in parsed if ax == axis]
        for high in (False, True):
            ln = _pick_side(group, high)
            if ln is not None:
                chosen[(axis, high)] = ln

    def position(axis: int, high: bool) -> float | None:
        ln = chosen.get((axis, high))
        if ln is None:
            return None
        for ax, pos, other in parsed:
            if other is ln and ax == axis:
                return pos
        return None

    x0, x1 = position(0, False), position(0, True)
    y0, y1 = position(1, False), position(1, True)
    # A line that crosses the interior, well short of where the other two walls
    # end, is the neighbour seen through a door. The room ends with those walls.
    x_ext = _extent_from([ln for (ax, _), ln in chosen.items() if ax == 1], 0)
    y_ext = _extent_from([ln for (ax, _), ln in chosen.items() if ax == 0], 1)
    if x_ext is not None:
        if x0 is not None and x0 > x_ext[0] + 0.30:
            x0 = None
        if x1 is not None and x1 < x_ext[1] - 0.30:
            x1 = None
        x0 = x_ext[0] if x0 is None else x0
        x1 = x_ext[1] if x1 is None else x1
    if y_ext is not None:
        if y0 is not None and y0 > y_ext[0] + 0.30:
            y0 = None
        if y1 is not None and y1 < y_ext[1] - 0.30:
            y1 = None
        y0 = y_ext[0] if y0 is None else y0
        y1 = y_ext[1] if y1 is None else y1
    # The fitted line is pulled outward by points seen through the doorway.
    # The mode of the band is the wall.
    x0 = _mode_band(support_xy, 0, x0)
    x1 = _mode_band(support_xy, 0, x1)
    y0 = _mode_band(support_xy, 1, y0)
    y1 = _mode_band(support_xy, 1, y1)
    if None in (x0, x1, y0, y1):
        return None
    if x1 - x0 < 0.7 or y1 - y0 < 0.7 or x1 - x0 > 12 or y1 - y0 > 12:
        return None
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    specs = [
        (np.array([x0, y0]), np.array([1.0, 0.0]), np.array([0.0, -1.0]), 0.0, x1 - x0, support_xy[:, 0] - x0, np.abs(support_xy[:, 1] - y0)),
        (np.array([x1, y0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.0, y1 - y0, support_xy[:, 1] - y0, np.abs(support_xy[:, 0] - x1)),
        (np.array([x1, y1]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), 0.0, x1 - x0, x1 - support_xy[:, 0], np.abs(support_xy[:, 1] - y1)),
        (np.array([x0, y1]), np.array([0.0, -1.0]), np.array([-1.0, 0.0]), 0.0, y1 - y0, y1 - support_xy[:, 1], np.abs(support_xy[:, 0] - x0)),
    ]
    edges: list[Line2D] = []
    openings: list[Opening] = []
    for i, (origin, direction, normal, t_min, t_max, along, off) in enumerate(specs):
        near = support_xy[off < 0.10]
        along_near = along[off < 0.10]
        ln = Line2D(origin, direction, normal, t_min, t_max, 0.01, [], 1.2, int(len(near)))
        if len(along_near) >= 6:
            pts = origin + np.outer(along_near, direction)
            ln.gaps = line_gaps(pts, ln, min_count=3)
        edges.append(ln)
        for g0, g1 in ln.gaps:
            openings.append(Opening(g0, g1, "door", i))
    return RoomGeom(poly, edges, openings, 0.0, ceiling, 0)


def _edge_op(room: RoomGeom, edge: int):
    poly = np.asarray(room.polygon, dtype=float)
    a = poly[edge]
    b = poly[(edge + 1) % len(poly)]
    delta = b - a
    length = float(np.linalg.norm(delta)) + 1e-9
    mid = 0.5 * (a + b)
    outward = np.array([-delta[1], delta[0]]) / length
    if np.dot(mid - poly.mean(axis=0), outward) < 0:
        outward = -outward
    best = None
    for op in room.openings:
        if op.line_index != edge or op.line_index >= len(room.lines):
            continue
        m = room.lines[op.line_index].point_at(op.midpoint_t)
        rank = abs(op.width - 0.85)
        if best is None or rank < best[0]:
            best = (rank, m, outward, op.width)
    if best is not None:
        return best[1], best[2], best[3]
    return mid, outward, length


def _polygons_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    for p in aa:
        if cv2.pointPolygonTest(bb, (float(p[0]), float(p[1])), False) > 0:
            return True
    for p in bb:
        if cv2.pointPolygonTest(aa, (float(p[0]), float(p[1])), False) > 0:
            return True
    return False


def _place_one_sided(parent, child, parent_local, child_op, local, placed, pose_of) -> bool:
    """Attach the room whose door photo missed, on the outside of the doorway."""
    if parent_local is None and child_op is None:
        return False
    others = [r.polygon for n, r in placed.items() if n != child]
    if parent_local is not None and child_op is None:
        known = _apply_vec(parent_local, *pose_of[parent])
        trials = [_edge_op(local[child], i) for i in range(len(local[child].polygon))]
        fixed_is_known = True
    elif child_op is not None and parent_local is None:
        known = None
        trials = [_edge_op(placed[parent], i) for i in range(len(placed[parent].polygon))]
        fixed_is_known = False
    else:
        return False
    for trial in trials:
        if fixed_is_known:
            yaw, trans = _align(known, trial)
            mid, outward = known[0], known[1]
        else:
            yaw, trans = _align(trial, child_op)
            mid, outward = trial[0], trial[1]
        moved = _transform(local[child], yaw, trans)
        centre = moved.polygon.mean(axis=0)
        if np.dot(centre - mid, outward) < 0.4:
            continue
        if any(_polygons_overlap(moved.polygon, other) for other in others):
            continue
        placed[child] = _named(moved, child)
        pose_of[child] = (yaw, trans)
        return True
    return False


def _stitch(local: dict[str, RoomGeom], views: dict[str, list[dict]], k: np.ndarray):
    names = list(local)
    # Prefer a named start that others hang off, otherwise the largest room.
    start = "living" if "living" in local else max(names, key=lambda n: local[n].area)
    placed: dict[str, RoomGeom] = {start: _named(_transform(local[start], 0.0, np.zeros(2)), start)}
    pose_of: dict[str, tuple[float, np.ndarray]] = {start: (0.0, np.zeros(2))}
    links_used = 0
    guard = 0
    while len(placed) < len(local) and guard < 12:
        guard += 1
        progressed = False
        for parent, child in _door_pairs(views):
            if parent not in local or child not in local:
                continue
            if (parent in placed) == (child in placed):
                continue
            if parent not in placed:
                parent, child = child, parent
            n_before = len(local[parent].openings)
            parent_local = _opening_looked_at(local[parent], _view_named(views, parent, child))
            if parent in placed and len(local[parent].openings) > n_before:
                # The placed copy was made before this doorway was recorded.
                placed[parent].openings.extend(local[parent].openings[n_before:])
            child_op = _opening_looked_at(local[child], _view_named(views, child, parent))
            if parent_local is None or child_op is None:
                # One of the two door photos did not pose. Try every wall of
                # the unposed room and keep the side that lands outside the
                # doorway instead of on top of a room we already placed.
                if not _place_one_sided(parent, child, parent_local, child_op, local, placed, pose_of):
                    continue
                links_used += 1
                progressed = True
                continue
            parent_op = _apply_vec(parent_local, *pose_of[parent])
            yaw, trans = _align(parent_op, child_op)
            placed[child] = _named(_transform(local[child], yaw, trans), child)
            pose_of[child] = (yaw, trans)
            links_used += 1
            progressed = True
        if not progressed:
            break
    # Rooms with no door link stay in their own frame. The stitch reports them
    # and the adjacency list simply does not include the missing door.
    for name, room in local.items():
        if name not in placed:
            placed[name] = _named(room, name)
    ordered = [placed[n] for n in names if n in placed]
    return ordered, links_used


def _door_pairs(views: dict[str, list[dict]]):
    pairs = []
    for room, vs in views.items():
        for view in vs:
            stem = view["name"]
            if stem.startswith("door-to-"):
                pairs.append((room, stem[len("door-to-") :]))
    return pairs


def _view_named(views, room, other):
    for view in views.get(room, []):
        if view["name"] == f"door-to-{other}":
            return view
    return None


def _apply_vec(op, yaw: float, trans: np.ndarray):
    mid, normal = op[0], op[1]
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return rot @ mid + trans, rot @ normal


def _boundary_hit(room: RoomGeom, cam: np.ndarray, forward: np.ndarray):
    """Where the camera ray meets the room rectangle, and that edge's outward normal."""
    poly = room.polygon
    centre = poly.mean(axis=0)
    best = None
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        edge = b - a
        # cam + s * forward = a + u * edge
        mat = np.column_stack([forward, -edge])
        if abs(np.linalg.det(mat)) < 1e-8:
            continue
        try:
            s, u = np.linalg.solve(mat, a - cam)
        except np.linalg.LinAlgError:
            continue
        if s < 0.3 or u < -0.05 or u > 1.05:
            continue
        if best is None or s < best[0]:
            hit = a + edge * float(np.clip(u, 0.0, 1.0))
            outward = np.array([-edge[1], edge[0]], dtype=float)
            outward /= np.linalg.norm(outward) + 1e-9
            if np.dot(hit - centre, outward) < 0:
                outward = -outward
            best = (float(s), hit, outward, i)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _opening_looked_at(room: RoomGeom, view: dict | None):
    if view is None:
        return None
    cam = view["t"][:2]
    forward = view["R"][:2, 2]
    nrm = np.linalg.norm(forward)
    if nrm < 1e-6:
        return None
    forward = forward / nrm
    hit = _boundary_hit(room, cam, forward)
    if hit is None:
        return None
    hit_pt, outward, edge_i = hit
    # A detected doorway on that edge, close to where the camera is aimed.
    best = None
    for op in room.openings:
        if op.line_index != edge_i or op.line_index >= len(room.lines):
            continue
        ln = room.lines[op.line_index]
        mid = ln.point_at(op.midpoint_t)
        if np.linalg.norm(mid - hit_pt) > 0.7:
            continue
        err = float(np.linalg.norm(mid - hit_pt))
        if best is None or err < best[0]:
            best = (err, mid, outward, op.width)
    if best is not None:
        return best[1], best[2], best[3]
    # The doorway is open floor, so the gap finder never saw a jamb. The aim is the door.
    # Record it, or the stitch joins the rooms and the adjacency list never hears about it.
    _add_opening(room, edge_i, hit_pt, 0.8)
    return hit_pt, outward, 0.8


def _add_opening(room: RoomGeom, edge: int, mid: np.ndarray, width: float) -> None:
    if edge < 0 or edge >= len(room.lines):
        return
    ln = room.lines[edge]
    for op in room.openings:
        if op.line_index != edge:
            continue
        if np.linalg.norm(ln.point_at(op.midpoint_t) - mid) < 0.4:
            return
    t = float(np.dot(mid - ln.origin, ln.direction))
    half = 0.5 * width
    room.openings.append(Opening(t - half, t + half, "door", edge))


def _opening_of_width(room: RoomGeom, width: float):
    """The doorway closest in width, used when that room's own photo did not pose."""
    best = None
    centre = room.polygon.mean(axis=0)
    for op in room.openings:
        if op.line_index < 0 or op.line_index >= len(room.lines):
            continue
        if abs(op.width - width) > 0.22:
            continue
        ln = room.lines[op.line_index]
        mid = ln.point_at(op.midpoint_t)
        outward = ln.normal.copy()
        if np.dot(mid - centre, outward) < 0:
            outward = -outward
        err = abs(op.width - width)
        if best is None or err < best[0]:
            best = (err, mid, outward, op.width)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _align(parent_op, child_op):
    p_mid, p_n = parent_op[0], parent_op[1]
    c_mid, c_n = child_op[0], child_op[1]
    target = -p_n
    yaw = wrap_angle(math.atan2(target[1], target[0]) - math.atan2(c_n[1], c_n[0]))
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    trans = p_mid - rot @ c_mid
    return yaw, trans


def _transform(room: RoomGeom, yaw: float, trans: np.ndarray) -> RoomGeom:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])

    def apply(xy):
        return (rot @ np.asarray(xy, dtype=float)) + trans

    poly = np.vstack([apply(p) for p in room.polygon])
    lines = []
    for ln in room.lines:
        lines.append(
            Line2D(
                apply(ln.origin),
                rot @ ln.direction,
                rot @ ln.normal,
                ln.t_min,
                ln.t_max,
                ln.residual,
                list(ln.gaps),
                ln.z_span,
                ln.n_points,
            )
        )
    openings = [Opening(op.t0, op.t1, op.kind, op.line_index) for op in room.openings]
    moved = RoomGeom(poly, lines, openings, room.floor_z, room.ceiling_z, room.label)
    return moved


def _named(room: RoomGeom, name: str) -> RoomGeom:
    room.label_name = name
    return room

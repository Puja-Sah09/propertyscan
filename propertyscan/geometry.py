"""Metric floor-plan geometry: SE(2) pose graphs, ICP, line gaps, room faces.

World frame is Z-up, X/Y on the floor. Camera frames used by callers are OpenCV
(x right, y down, z forward). Nothing in this module knows about a particular
building.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (a + math.pi) % (2 * math.pi) - math.pi


def se2_apply(yaw: float, trans: np.ndarray, pts: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s], [s, c]])
    return pts @ r.T + trans


def se2_compose(yaw_a: float, t_a: np.ndarray, yaw_b: float, t_b: np.ndarray):
    """Apply B first, then A. Result maps points the way A(B(p)) does."""
    c, s = math.cos(yaw_a), math.sin(yaw_a)
    r = np.array([[c, -s], [s, c]])
    return wrap_angle(yaw_a + yaw_b), r @ t_b + t_a


def kabsch_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """Rigid SE(2) taking src onto dst. Returns yaw, translation."""
    ca = src.mean(axis=0)
    cb = dst.mean(axis=0)
    h = (src - ca).T @ (dst - cb)
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        r = vt.T @ u.T
    yaw = math.atan2(r[1, 0], r[0, 0])
    trans = cb - r @ ca
    return yaw, trans


def icp_se2(
    source: np.ndarray,
    target: np.ndarray,
    init_yaw: float = 0.0,
    init_trans: np.ndarray | None = None,
    iters: int = 25,
    gate: float = 0.12,
    min_inliers: int = 12,
    search_yaw: bool = True,
) -> tuple[float, np.ndarray, float, int]:
    """Align source onto target. Returns yaw, translation, mean residual, inlier count.

    A wide first pass recenters the clouds and, when asked, tries yaw seeds.
    The tight gate is only used once a seed already overlaps the target.
    """
    if init_trans is None:
        init_trans = np.zeros(2)
    if len(source) < min_inliers or len(target) < min_inliers:
        return float(init_yaw), np.asarray(init_trans, dtype=float), float("inf"), 0
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    seeds = [(float(init_yaw), np.asarray(init_trans, dtype=float).copy())]
    # Centroid alignment at the caller's yaw, then a yaw sweep around it.
    delta = target.mean(axis=0) - se2_apply(init_yaw, np.zeros(2), source).mean(axis=0)
    seeds.append((float(init_yaw), delta))
    if search_yaw:
        for k in range(8):
            y = float(init_yaw) + k * math.pi / 4
            shift = target.mean(0) - se2_apply(y, np.zeros(2), source).mean(0)
            seeds.append((y, shift))

    def _refine(yaw0: float, trans0: np.ndarray, g: float) -> tuple[float, np.ndarray, float, int]:
        yaw = float(yaw0)
        trans = trans0.astype(float).copy()
        mean = float("inf")
        n_in = 0
        prev = None
        for _ in range(iters):
            moved = se2_apply(yaw, trans, source)
            dist, idx = tree.query(moved, workers=1)
            inl = dist < g
            n_in = int(inl.sum())
            if n_in < min_inliers:
                break
            dy, dt = kabsch_2d(moved[inl], target[idx[inl]])
            yaw, trans = se2_compose(dy, dt, yaw, trans)
            mean = float(dist[inl].mean())
            if prev is not None and abs(prev - mean) < 1e-5:
                break
            prev = mean
        return float(wrap_angle(yaw)), trans, mean, n_in

    best = (float(init_yaw), np.asarray(init_trans, dtype=float), float("inf"), 0)
    for y0, t0 in seeds:
        # Wide gate to lock on, then the caller's gate to measure the fit.
        y, t, _, n = _refine(y0, t0, max(gate, 0.35))
        if n < min_inliers:
            continue
        y, t, mean, n = _refine(y, t, gate)
        if n > best[3] or (n == best[3] and mean < best[2]):
            best = (y, t, mean, n)
    return best


@dataclass
class Edge:
    i: int
    j: int
    dx: float
    dy: float
    dth: float
    weight: float = 1.0


def optimize_pose_graph(n: int, edges: list[Edge], prior: np.ndarray | None = None, prior_weight: float = 0.8) -> np.ndarray:
    """Optimize SE(2) poses. Pose 0 is anchored at the origin with yaw 0.

    Each edge measures the motion from i to j expressed in frame i:
    x_j = x_i + R(yaw_i) @ (dx, dy), yaw_j = yaw_i + dth.
    Returns (n, 3) array of x, y, yaw.
    """
    if n == 0:
        return np.zeros((0, 3))
    x0 = np.zeros(3 * n)
    # Integrate a spanning set of odometry-like edges in index order as the seed.
    ordered = sorted(edges, key=lambda e: (e.i, e.j))
    for e in ordered:
        if e.j == e.i + 1 and e.i >= 0:
            xi, yi, ti = x0[3 * e.i : 3 * e.i + 3]
            c, s = math.cos(ti), math.sin(ti)
            x0[3 * e.j] = xi + c * e.dx - s * e.dy
            x0[3 * e.j + 1] = yi + s * e.dx + c * e.dy
            x0[3 * e.j + 2] = ti + e.dth

    def residual(v: np.ndarray) -> np.ndarray:
        out = []
        # Anchor pose 0.
        out.extend([v[0] * 50.0, v[1] * 50.0, v[2] * 50.0])
        for e in edges:
            xi, yi, ti = v[3 * e.i : 3 * e.i + 3]
            xj, yj, tj = v[3 * e.j : 3 * e.j + 3]
            c, s = math.cos(ti), math.sin(ti)
            px = xi + c * e.dx - s * e.dy
            py = yi + s * e.dx + c * e.dy
            w = math.sqrt(e.weight)
            out.append((px - xj) * w)
            out.append((py - yj) * w)
            out.append(wrap_angle(ti + e.dth - tj) * w)
        # A loop that matched the wrong wall used to walk the phone by metres.
        # The odometry prior keeps a bad loop from throwing the path away,
        # and a consistent loop still moves a pose by a few decimetres.
        if prior is not None:
            for i in range(n):
                out.append((v[3 * i] - prior[i, 0]) * prior_weight)
                out.append((v[3 * i + 1] - prior[i, 1]) * prior_weight)
                out.append(wrap_angle(v[3 * i + 2] - prior[i, 2]) * prior_weight)
        return np.asarray(out, dtype=float)

    sol = least_squares(residual, x0, method="trf", loss="soft_l1", f_scale=0.05, max_nfev=80)
    poses = sol.x.reshape(n, 3)
    poses[:, 2] = wrap_angle(poses[:, 2])
    return poses


def yaw_from_opencv_rotation(r: np.ndarray) -> float:
    """Yaw of the camera forward axis about world +Z. OpenCV z is forward."""
    forward = r[:, 2]
    horizontal = forward.copy()
    horizontal[2] = 0.0
    n = np.linalg.norm(horizontal)
    if n < 1e-8:
        # Looking straight up or down: fall back to the camera right axis.
        right = r[:, 0]
        return math.atan2(right[0], -right[1])
    return math.atan2(horizontal[1], horizontal[0])


def apply_yaw(r: np.ndarray, yaw_new: float) -> np.ndarray:
    """Replace the world yaw of an OpenCV rotation, keeping roll and pitch."""
    yaw_old = yaw_from_opencv_rotation(r)
    dy = float(wrap_angle(yaw_new - yaw_old))
    c, s = math.cos(dy), math.sin(dy)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return rz @ r


@dataclass
class Line2D:
    origin: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    t_min: float
    t_max: float
    residual: float
    gaps: list[tuple[float, float]] = field(default_factory=list)
    z_span: float = 0.0
    n_points: int = 0

    def point_at(self, t: float) -> np.ndarray:
        return self.origin + self.direction * t

    def distance(self, xy: np.ndarray) -> float:
        return float(np.dot(xy - self.origin, self.normal))


def _fit_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0 or (abs(direction[0]) < 1e-9 and direction[1] < 0):
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    resid = np.abs((pts - c) @ normal)
    return c, direction, normal, float(np.median(resid))


def ransac_lines(
    xy: np.ndarray,
    z: np.ndarray | None = None,
    thresh: float = 0.018,
    min_points: int = 40,
    min_length: float = 0.35,
    iterations: int = 180,
    seed: int = 0,
) -> list[Line2D]:
    """Sequential RANSAC of 2D lines. `z` is optional and only records vertical span."""
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(xy))
    lines: list[Line2D] = []
    if z is None:
        z = np.zeros(len(xy))
    while len(remaining) >= min_points:
        best_inl = None
        best_count = 0
        pool = xy[remaining]
        n = len(pool)
        if n < min_points:
            break
        for _ in range(iterations):
            i1, i2 = rng.integers(0, n, size=2)
            p, q = pool[i1], pool[i2]
            if np.linalg.norm(p - q) < 0.25:
                continue
            d = q - p
            d = d / np.linalg.norm(d)
            nrm = np.array([-d[1], d[0]])
            dist = np.abs((pool - p) @ nrm)
            inl = dist < thresh
            count = int(inl.sum())
            if count > best_count:
                best_count = count
                best_inl = inl
        if best_inl is None or best_count < min_points:
            break
        sel = remaining[best_inl]
        pts = xy[sel]
        origin, direction, normal, resid = _fit_line(pts)
        # One polish pass with a slightly wider gate on the refined normal.
        dist = np.abs((pool - origin) @ normal)
        polish = dist < thresh
        if int(polish.sum()) >= min_points:
            sel = remaining[polish]
            pts = xy[sel]
            origin, direction, normal, resid = _fit_line(pts)
        t = (pts - origin) @ direction
        length = float(t.max() - t.min())
        if length >= min_length:
            lines.append(
                Line2D(
                    origin=origin,
                    direction=direction,
                    normal=normal,
                    t_min=float(t.min()),
                    t_max=float(t.max()),
                    residual=resid,
                    z_span=float(z[sel].max() - z[sel].min()) if len(sel) else 0.0,
                    n_points=len(sel),
                )
            )
        # Remove a thicker band than the fit gate so a parallel shell of the
        # same wall cannot become a second line.
        dist_all = np.abs((xy[remaining] - origin) @ normal)
        along = (xy[remaining] - origin) @ direction
        drop = (dist_all < max(thresh, 0.045)) & (along > t.min() - 0.05) & (along < t.max() + 0.05)
        remaining = remaining[~drop]
    return lines


def line_gaps(
    xy: np.ndarray,
    line: Line2D,
    bin_m: float = 0.02,
    min_gap: float = 0.55,
    max_gap: float = 1.05,
    min_count: int = 2,
    refine: bool = True,
) -> list[tuple[float, float]]:
    """Open intervals along the line where support is missing. Widths in meters."""
    dist = np.abs((xy - line.origin) @ line.normal)
    near = xy[dist < 0.03]
    if len(near) < 8:
        return []
    t = (near - line.origin) @ line.direction
    t = t[(t >= line.t_min - 0.02) & (t <= line.t_max + 0.02)]
    if len(t) < 8:
        return []
    t0, t1 = float(np.min(t)), float(np.max(t))
    edges = np.arange(t0, t1 + bin_m, bin_m)
    if len(edges) < 4:
        return []
    hist, edges = np.histogram(t, bins=edges)
    empty = hist < min_count
    gaps: list[tuple[float, float]] = []
    i = 0
    n = len(empty)
    while i < n:
        if not empty[i]:
            i += 1
            continue
        j = i
        while j < n and empty[j]:
            j += 1
        # Ignore gaps that touch the outside of the segment: those are line ends.
        if i == 0 or j == n:
            i = j
            continue
        g0, g1 = float(edges[i]), float(edges[j])
        if refine:
            g0, g1 = _refine_jambs(t, g0, g1)
        width = g1 - g0
        if min_gap <= width <= max_gap:
            gaps.append((g0, g1))
        i = j
    return gaps


def _refine_jambs(t: np.ndarray, g0: float, g1: float) -> tuple[float, float]:
    center = 0.5 * (g0 + g1)
    left = t[t < center]
    right = t[t > center]
    if len(left) == 0 or len(right) == 0:
        return g0, g1
    left_band = left[left > g0 - 0.30]
    right_band = right[right < g1 + 0.30]
    def _edge(samples: np.ndarray, high: bool) -> float:
        if len(samples) >= 40:
            return float(np.percentile(samples, 98 if high else 2))
        return float(samples.max() if high else samples.min())

    left_jamb = _edge(left_band, True) if len(left_band) else float(left.max())
    right_jamb = _edge(right_band, False) if len(right_band) else float(right.min())
    if right_jamb - left_jamb < 0.2:
        return g0, g1
    return left_jamb, right_jamb


def coarse_gaps(xy: np.ndarray, line: Line2D, bin_m: float = 0.05) -> list[tuple[float, float]]:
    """Quantized gap widths. This is the pre-fix opening estimator, kept so the
    before/after benchmark can be regenerated without reverting the repo."""
    return line_gaps(xy, line, bin_m=bin_m, refine=False)


def intersect_lines(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    a = np.stack([d1, -d2], axis=1)
    try:
        ts = np.linalg.solve(a, p2 - p1)
    except np.linalg.LinAlgError:
        return None
    return p1 + ts[0] * d1


def polygon_from_lines(lines: list[Line2D], centroid: np.ndarray) -> np.ndarray | None:
    """Convex polygon from wall lines. Each line's normal is flipped to point outward."""
    if len(lines) < 3:
        return None
    oriented: list[tuple[float, Line2D]] = []
    for ln in lines:
        nrm = ln.normal.copy()
        mid = ln.point_at(0.5 * (ln.t_min + ln.t_max))
        # Outward points away from the room centroid.
        if np.dot(mid - centroid, nrm) < 0:
            nrm = -nrm
        ang = math.atan2(nrm[1], nrm[0])
        oriented.append((ang, Line2D(ln.origin, ln.direction, nrm, ln.t_min, ln.t_max, ln.residual, list(ln.gaps), ln.z_span, ln.n_points)))
    oriented.sort(key=lambda x: x[0])
    # Drop near-duplicate angles (same wall seen twice).
    kept: list[Line2D] = []
    for ang, ln in oriented:
        if kept:
            prev_ang = math.atan2(kept[-1].normal[1], kept[-1].normal[0])
            if abs(wrap_angle(ang - prev_ang)) < math.radians(12):
                if ln.n_points > kept[-1].n_points:
                    kept[-1] = ln
                continue
        kept.append(ln)
    if len(kept) < 3:
        return None
    corners = []
    for i, ln in enumerate(kept):
        nxt = kept[(i + 1) % len(kept)]
        hit = intersect_lines(ln.origin, ln.direction, nxt.origin, nxt.direction)
        if hit is None:
            return None
        corners.append(hit)
    poly = np.asarray(corners, dtype=float)
    if polygon_area(poly) < 0.4:
        return None
    if not _is_convex(poly):
        return None
    return poly


def _snap_contour(contour: np.ndarray, lines: list[Line2D]) -> np.ndarray | None:
    """Replace each contour edge with the wall line it already lies on, then reintersect.

    The raster contour is inset by about a cell. The line fit is the measurement.
    """
    if len(contour) < 3 or len(lines) < 3:
        return None
    edge_lines: list[Line2D | None] = []
    for i in range(len(contour)):
        a = contour[i]
        b = contour[(i + 1) % len(contour)]
        edge = b - a
        elen = float(np.linalg.norm(edge))
        if elen < 0.25:
            edge_lines.append(None)
            continue
        edir = edge / elen
        mid = 0.5 * (a + b)
        best = None
        best_d = 0.16
        for ln in lines:
            parallel = abs(edir[0] * ln.direction[1] - edir[1] * ln.direction[0])
            if parallel > 0.30:
                continue
            # The edge must fall along the supported part of the line, not its extension.
            t_mid = float(np.dot(mid - ln.origin, ln.direction))
            if t_mid < ln.t_min - 0.3 or t_mid > ln.t_max + 0.3:
                continue
            d = abs(ln.distance(mid))
            if d < best_d:
                best_d = d
                best = ln
        edge_lines.append(best)
    # Collapse consecutive edges that snapped to the same line.
    collapsed: list[Line2D] = []
    for ln in edge_lines:
        if ln is None:
            continue
        if collapsed and collapsed[-1] is ln:
            continue
        collapsed.append(ln)
    if len(collapsed) >= 2 and collapsed[0] is collapsed[-1]:
        collapsed = collapsed[:-1]
    if len(collapsed) < 3:
        return None
    corners = []
    for i, ln in enumerate(collapsed):
        nxt = collapsed[(i + 1) % len(collapsed)]
        if nxt is ln:
            return None
        hit = intersect_lines(ln.origin, ln.direction, nxt.origin, nxt.direction)
        if hit is None or not np.all(np.isfinite(hit)):
            return None
        corners.append(hit)
    poly = np.asarray(corners, dtype=float)
    if polygon_area(poly) < 0.4 or not _is_convex(poly):
        return None
    return poly


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_convex(poly: np.ndarray) -> bool:
    n = len(poly)
    sign = 0
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-8:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def point_in_poly(pt: np.ndarray, poly: np.ndarray) -> bool:
    # Ray cast. Works for convex and simple concave polygons.
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


@dataclass
class Opening:
    t0: float
    t1: float
    kind: str
    line_index: int

    @property
    def width(self) -> float:
        return abs(self.t1 - self.t0)

    @property
    def midpoint_t(self) -> float:
        return 0.5 * (self.t0 + self.t1)


@dataclass
class RoomGeom:
    polygon: np.ndarray
    lines: list[Line2D]
    openings: list[Opening]
    floor_z: float
    ceiling_z: float
    label: int

    @property
    def area(self) -> float:
        return polygon_area(self.polygon)

    @property
    def ceiling_height(self) -> float:
        return self.ceiling_z - self.floor_z

    def wall_lengths(self) -> list[float]:
        p = self.polygon
        return [float(np.linalg.norm(p[(i + 1) % len(p)] - p[i])) for i in range(len(p))]


def _raster_setup(xy: np.ndarray, res: float, margin: float = 0.40):
    minxy = xy.min(axis=0) - margin
    maxxy = xy.max(axis=0) + margin
    w = int(math.ceil((maxxy[0] - minxy[0]) / res)) + 1
    h = int(math.ceil((maxxy[1] - minxy[1]) / res)) + 1
    return minxy, w, h


def _to_pix(xy: np.ndarray, origin: np.ndarray, res: float) -> tuple[int, int]:
    c = int(round((xy[0] - origin[0]) / res))
    r = int(round((xy[1] - origin[1]) / res))
    return c, r


def extract_rooms(
    wall_xy: np.ndarray,
    wall_z: np.ndarray,
    all_xy: np.ndarray,
    all_z: np.ndarray,
    seeds_xy: np.ndarray,
    beyond_xy: np.ndarray | None = None,
    refine_openings: bool = True,
    res: float = 0.02,
    min_height: float = 1.7,
    level_mode: str = "dense",
) -> list[RoomGeom]:
    """Turn a fused wall point cloud into rooms, wall lengths, and openings.

    `beyond_xy` are points seen through a gap (not on the wall). A gap with no
    beyond-points and no trajectory crossing is not called an opening: that is
    how a mirror or a single-surface hole is kept out of the plan.
    """
    if len(wall_xy) < 50 or len(seeds_xy) == 0:
        return []
    lines = ransac_lines(wall_xy, wall_z, min_points=70)
    # The wall band is intentionally short, so z_span is the band, not the storey.
    lines = [ln for ln in lines if ln.z_span >= 0.35 and ln.n_points >= 70]
    gap_fn = line_gaps if refine_openings else coarse_gaps
    for ln in lines:
        ln.gaps = gap_fn(wall_xy, ln)

    origin, gw, gh = _raster_setup(np.vstack([wall_xy, seeds_xy]), res)
    occ = np.zeros((gh, gw), np.uint8)
    # Occupancy comes from the points themselves. A fitted line is not allowed
    # to invent a wall across a gap it merely bridged.
    cols = np.round((wall_xy[:, 0] - origin[0]) / res).astype(np.int32)
    rows = np.round((wall_xy[:, 1] - origin[1]) / res).astype(np.int32)
    ok = (rows >= 0) & (rows < gh) & (cols >= 0) & (cols < gw)
    occ[rows[ok], cols[ok]] = 1
    # With the capture walking to the corners, the leftover gaps are about a decimetre.
    # A wider close pinches the 1.2 m hall.
    occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    seals: list[tuple[np.ndarray, np.ndarray, int]] = []
    for li, ln in enumerate(lines):
        kept = []
        for g0, g1 in ln.gaps:
            if not _gap_is_real(ln, g0, g1, seeds_xy, beyond_xy):
                if (g1 - g0) < 1.25:
                    p0 = ln.point_at(g0)
                    p1 = ln.point_at(g1)
                    cv2.line(occ, _to_pix(p0, origin, res), _to_pix(p1, origin, res), 1, thickness=3)
                continue
            kept.append((g0, g1))
            p0 = ln.point_at(g0)
            p1 = ln.point_at(g1)
            cv2.line(occ, _to_pix(p0, origin, res), _to_pix(p1, origin, res), 2, thickness=2)
            seals.append((p0, p1, li))
        ln.gaps = kept

    free = (occ == 0).astype(np.uint8)
    nlab, labels = cv2.connectedComponents(free, connectivity=4)
    border = np.zeros_like(labels, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    touches_border = {int(l) for l in np.unique(labels[border]) if l != 0}

    seed_labels: list[int] = []
    missed = 0
    for s in seeds_xy:
        c, r = _to_pix(s, origin, res)
        if not (0 <= r < gh and 0 <= c < gw):
            missed += 1
            continue
        lab = int(labels[r, c])
        if lab == 0 or lab in touches_border:
            missed += 1
            continue
        if lab not in seed_labels:
            seed_labels.append(lab)

    if not seed_labels:
        # One line so a leaked floor plan is diagnosable from the benchmark log.
        cv2.imwrite("occ_debug.png", (occ > 0).astype(np.uint8) * 255)
        print(
            f"room-label debug: components={nlab-1} border={len(touches_border)} "
            f"unseeded={missed} seeds={len(seeds_xy)} lines={len(lines)} wall_cells={int((occ>0).sum())} "
            f"gaps={[(round(g1-g0,2), _gap_is_real(ln,g0,g1,seeds_xy,beyond_xy)) for ln in lines for g0,g1 in ln.gaps]}"
        )
    rooms: list[RoomGeom] = []
    for lab in seed_labels:
        comp = (labels == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) * (res * res) < 0.8:
            continue
        eps = max(1.0, 0.10 / res)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        contour_poly = np.array(
            [origin + np.array([(c + 0.5) * res, (r + 0.5) * res]) for c, r in approx],
            dtype=float,
        )
        centroid = contour_poly.mean(axis=0)
        adj = _lines_touching_room(lines, labels, lab, origin, res, centroid)
        contour_area = float(cv2.contourArea(cnt)) * res * res
        poly = _snap_contour(contour_poly, adj)
        snap_area = polygon_area(poly) if poly is not None else 0.0
        # The raster is inset by the wall thickness. Snapping back to the fitted
        # lines should grow the room, not shrink it, and not swallow a neighbour.
        if poly is None or snap_area < contour_area * 0.9 or snap_area > max(contour_area * 1.85, contour_area + 3.0):
            poly = contour_poly
        mask_pts = _points_with_label(all_xy, labels, lab, origin, res, gh, gw)
        if len(mask_pts) < 20:
            floor_z, ceiling_z = 0.0, 2.4
        else:
            floor_z, ceiling_z = _levels(all_z[mask_pts], mode=level_mode)
        if ceiling_z - floor_z < min_height:
            continue
        openings = _openings_for_room(lines, adj, poly, centroid)
        rooms.append(
            RoomGeom(
                polygon=_order_ccw(poly),
                lines=adj,
                openings=openings,
                floor_z=floor_z,
                ceiling_z=ceiling_z,
                label=lab,
            )
        )
    return rooms


def _bridge_perpendicular_ends(segments, occ, origin, res, max_gap: float) -> None:
    """Extend a wall end onto a perpendicular wall. Collinear ends are doors and are left open.

    The limit stays under the narrowest room (the hall is 1.2 m) so this cannot
    draw a wall across a room, and it stays under a door width only when the
    hit is a corner rather than a jamb-to-jamb span.
    """
    segs = []
    for p0, p1 in segments:
        d = p1 - p0
        n = float(np.linalg.norm(d))
        if n < 0.25:
            continue
        segs.append((p0, p1, d / n, n))
    for i, (p0, p1, d, n) in enumerate(segs):
        for end in (p0, p1):
            best = None
            best_dist = max_gap
            for j, (q0, q1, u, m) in enumerate(segs):
                if i == j or abs(float(np.dot(d, u))) > 0.4:
                    continue
                t = float(np.dot(end - q0, u))
                if t < -0.15 or t > m + 0.15:
                    continue
                foot = q0 + u * min(max(t, 0.0), m)
                dist = float(np.linalg.norm(end - foot))
                if 0.03 < dist < best_dist:
                    best_dist = dist
                    best = foot
            if best is not None:
                cv2.line(occ, _to_pix(end, origin, res), _to_pix(best, origin, res), 1, thickness=3)


def _join_corners(lines: list[Line2D], occ: np.ndarray, origin: np.ndarray, res: float) -> None:
    """Extend wall endpoints up to 0.45 m to meet a perpendicular wall.

    A door is wider than that, so this closes open corners and does not close openings.
    """
    for i, a in enumerate(lines):
        for b in lines[i + 1 :]:
            cross = abs(a.direction[0] * b.direction[1] - a.direction[1] * b.direction[0])
            if cross < 0.75:
                continue
            hit = intersect_lines(a.origin, a.direction, b.origin, b.direction)
            if hit is None or not np.all(np.isfinite(hit)):
                continue
            for ln in (a, b):
                t_hit = float(np.dot(hit - ln.origin, ln.direction))
                if ln.t_min - 0.45 <= t_hit < ln.t_min:
                    p0, p1 = ln.point_at(t_hit), ln.point_at(ln.t_min)
                elif ln.t_max < t_hit <= ln.t_max + 0.45:
                    p0, p1 = ln.point_at(ln.t_max), ln.point_at(t_hit)
                else:
                    continue
                cv2.line(occ, _to_pix(p0, origin, res), _to_pix(p1, origin, res), 1, thickness=2)


def _covered_intervals(ln: Line2D) -> list[tuple[float, float]]:
    cuts = [ln.t_min, ln.t_max]
    for g0, g1 in ln.gaps:
        cuts.extend([g0, g1])
    cuts = sorted(cuts)
    spans = []
    gaps = [(min(a, b), max(a, b)) for a, b in ln.gaps]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 0.05:
            continue
        mid = 0.5 * (a + b)
        if any(g0 - 1e-6 <= mid <= g1 + 1e-6 for g0, g1 in gaps):
            continue
        spans.append((a, b))
    if not spans:
        spans.append((ln.t_min, ln.t_max))
    return spans


def _gap_is_real(
    ln: Line2D,
    g0: float,
    g1: float,
    seeds_xy: np.ndarray,
    beyond_xy: np.ndarray | None,
) -> bool:
    """A gap is a door when the capture walks through it.

    Floor points sit in front of every wall, so they are not evidence of a
    room beyond. An occlusion does not get walked through; a doorway does,
    because the capture protocol requires a pass through every opening.
    """
    p0 = ln.point_at(g0)
    p1 = ln.point_at(g1)
    if _polyline_crosses(seeds_xy, p0, p1, tol=0.40):
        return True
    if beyond_xy is None or len(beyond_xy) == 0:
        return False
    t = (beyond_xy - ln.origin) @ ln.direction
    off = np.abs((beyond_xy - ln.origin) @ ln.normal)
    in_gap = (t > g0 + 0.05) & (t < g1 - 0.05) & (off > 0.25) & (off < 2.5)
    return int(np.count_nonzero(in_gap)) >= 25


def _polyline_crosses(path: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    if len(path) == 0:
        return False
    # Distance from path samples to the segment.
    ab = b - a
    lab = np.linalg.norm(ab) + 1e-9
    u = ab / lab
    for p in path:
        t = np.dot(p - a, u)
        if t < -0.05 or t > lab + 0.05:
            continue
        if np.linalg.norm(p - (a + u * t)) <= tol:
            return True
    return False


def _lines_touching_room(lines, labels, lab, origin, res, centroid) -> list[Line2D]:
    """A line can bound several collinear rooms. Match on local support, not the midpoint."""
    chosen = []
    comp = labels == lab
    gh, gw = labels.shape
    for ln in lines:
        hits = 0
        ts = np.linspace(ln.t_min, ln.t_max, 48)
        for t in ts:
            if any(g0 <= t <= g1 for g0, g1 in ln.gaps):
                continue
            c, r = _to_pix(ln.point_at(float(t)), origin, res)
            found = False
            for dr in range(-8, 9):
                rr = r + dr
                if rr < 0 or rr >= gh:
                    continue
                for dc in range(-8, 9):
                    cc = c + dc
                    if 0 <= cc < gw and comp[rr, cc]:
                        found = True
                        break
                if found:
                    break
            if found:
                hits += 1
                if hits >= 3:
                    break
        if hits >= 3:
            chosen.append(ln)
    return chosen


def _points_with_label(xy, labels, lab, origin, res, gh, gw) -> np.ndarray:
    cols = np.round((xy[:, 0] - origin[0]) / res).astype(int)
    rows = np.round((xy[:, 1] - origin[1]) / res).astype(int)
    ok = (rows >= 0) & (rows < gh) & (cols >= 0) & (cols < gw)
    idx = np.nonzero(ok)[0]
    keep = labels[rows[ok], cols[ok]] == lab
    return idx[keep]


def _levels_percentile(z: np.ndarray) -> tuple[float, float]:
    """The estimator the hall failed with: percentile 5 and percentile 95.

    Kept so the fix-loop before-state can be regenerated. A high percentile
    sits on the noisy tail above the ceiling, which put the hall 1.6 cm high.
    """
    z = z[np.isfinite(z)]
    if len(z) < 20:
        med = float(np.median(z)) if len(z) else 0.0
        return med, med + 2.4
    return float(np.percentile(z, 5)), float(np.percentile(z, 95))


def _levels(z: np.ndarray, mode: str = "dense") -> tuple[float, float]:
    """Floor and ceiling from the outermost dense bins.

    A high percentile sits on the noisy tail: depth noise that falls back into
    the wall band leaves the sample, and the tail that stays above the ceiling
    remains. The hall was 1.6 cm high that way, which is the whole ceiling
    gate.     The dense bin is the mode. Sparse tails do not vote.
    """
    if mode == "percentile":
        return _levels_percentile(z)
    z = z[np.isfinite(z)]
    if len(z) < 20:
        med = float(np.median(z)) if len(z) else 0.0
        return med, med + 2.4

    def edge(high: bool) -> float:
        lo, hi = np.percentile(z, [0.4, 99.6])
        if hi - lo < 0.2:
            return float(np.median(z))
        bins = np.arange(lo, hi + 0.015, 0.015)
        hist, edges = np.histogram(z, bins=bins)
        need = max(20, int(0.003 * len(z)))
        dense = np.flatnonzero(hist >= need)
        if len(dense) == 0:
            return float(np.percentile(z, 99 if high else 1))
        i = int(dense[-1] if high else dense[0])
        center = 0.5 * (float(edges[i]) + float(edges[i + 1]))
        band = z[np.abs(z - center) < 0.02]
        return float(np.median(band)) if len(band) >= 8 else center

    floor_z, ceil_z = edge(False), edge(True)
    if ceil_z - floor_z < 1.5:
        floor_z = float(np.percentile(z, 2))
        ceil_z = float(np.percentile(z, 98))
    return floor_z, ceil_z


def _openings_for_room(all_lines: list[Line2D], room_lines: list[Line2D], poly: np.ndarray, centroid: np.ndarray) -> list[Opening]:
    openings: list[Opening] = []
    # Index into this room's lines. A global line number drops the opening
    # once build_plan looks it up on the room, which hid the hall's other doors.
    for li, ln in enumerate(room_lines):
        for g0, g1 in ln.gaps:
            mid = ln.point_at(0.5 * (g0 + g1))
            # Keep openings whose midpoint lies near this room's boundary.
            if _distance_to_polygon_edges(mid, poly) > 0.35:
                continue
            kind = "door"
            openings.append(Opening(g0, g1, kind, li))
    return openings


def _distance_to_polygon_edges(pt: np.ndarray, poly: np.ndarray) -> float:
    best = float("inf")
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        lab = np.linalg.norm(ab) + 1e-9
        t = np.clip(np.dot(pt - a, ab) / (lab * lab), 0, 1)
        best = min(best, float(np.linalg.norm(pt - (a + t * ab))))
    return best


def _order_ccw(poly: np.ndarray) -> np.ndarray:
    c = poly.mean(axis=0)
    ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
    return poly[np.argsort(ang)]


def adjacency_from_openings(rooms: list[RoomGeom], lines: list[Line2D] | None = None) -> list[tuple[int, int]]:
    """Two rooms are adjacent when they share a gap, or their openings sit on the same doorway."""
    pairs = []
    mids: list[tuple[int, np.ndarray]] = []
    for ri, room in enumerate(rooms):
        for op in room.openings:
            if op.line_index < 0 or op.line_index >= len(room.lines):
                continue
            mids.append((ri, room.lines[op.line_index].point_at(op.midpoint_t)))
    for i in range(len(mids)):
        for j in range(i + 1, len(mids)):
            if mids[i][0] == mids[j][0]:
                continue
            if float(np.linalg.norm(mids[i][1] - mids[j][1])) <= 0.45:
                a, b = sorted((mids[i][0], mids[j][0]))
                pairs.append((a, b))
    return sorted(set(pairs))


def recover_open_rooms(
    wall_xy: np.ndarray,
    seeds_xy: np.ndarray,
    all_xy: np.ndarray,
    all_z: np.ndarray,
    res: float = 0.04,
) -> list[RoomGeom]:
    """Rooms for a scan whose walls do not close on the raster border.

    A convex hull of the observed walls stops the free space leaking out of
    the map. Door-width passages are cut by opening the free space, and each
    remaining piece that the walk actually entered becomes a room. Edges with
    no wall points behind them are openings, not measured solid wall.
    """
    if len(wall_xy) < 80 or len(seeds_xy) == 0:
        return []
    origin, gw, gh = _raster_setup(np.vstack([wall_xy, seeds_xy]), res)
    occ = np.zeros((gh, gw), np.uint8)
    cols = np.clip(np.round((wall_xy[:, 0] - origin[0]) / res).astype(np.int32), 0, gw - 1)
    rows = np.clip(np.round((wall_xy[:, 1] - origin[1]) / res).astype(np.int32), 0, gh - 1)
    occ[rows, cols] = 1
    occ = cv2.dilate(occ, np.ones((3, 3), np.uint8))
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    hull = cv2.convexHull(np.stack([cols, rows], axis=1))
    mask = np.zeros_like(occ)
    cv2.fillConvexPoly(mask, hull, 1)
    free = ((mask == 1) & (occ == 0)).astype(np.uint8)
    radius = 0.42
    k = int(round(radius / res)) * 2 + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    opened = cv2.morphologyEx(free, cv2.MORPH_OPEN, ker)
    nlab, labels = cv2.connectedComponents(opened, connectivity=8)
    # Voxel the walls so edge-support checks stay cheap.
    wall_key = np.unique(np.round(wall_xy / 0.05).astype(np.int32), axis=0)
    wall_ds = wall_key.astype(np.float64) * 0.05
    rooms: list[RoomGeom] = []
    for lab in range(1, nlab):
        comp = (labels == lab).astype(np.uint8)
        if float(comp.sum()) * res * res < 1.0:
            continue
        dil = cv2.dilate(comp, np.ones((7, 7), np.uint8))
        hits = 0
        for s in seeds_xy:
            c, r = _to_pix(s, origin, res)
            if 0 <= r < gh and 0 <= c < gw and dil[r, c]:
                hits += 1
        if hits < 2:
            continue
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, 0.40 / res, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        poly = origin + np.stack([(approx[:, 0] + 0.5) * res, (approx[:, 1] + 0.5) * res], axis=1)
        if polygon_area(poly) < 1.0:
            continue
        ac = np.round((all_xy[:, 0] - origin[0]) / res).astype(np.int32)
        ar = np.round((all_xy[:, 1] - origin[1]) / res).astype(np.int32)
        inside = (ar >= 0) & (ar < gh) & (ac >= 0) & (ac < gw) & (dil[ar, ac] > 0)
        if int(inside.sum()) >= 20:
            floor_z, ceiling_z = _levels(all_z[inside])
        else:
            floor_z, ceiling_z = 0.0, 2.4
        lines, openings = _boundary_lines(poly, wall_ds)
        rooms.append(RoomGeom(poly, lines, openings, floor_z, ceiling_z, lab))
    return rooms


def _boundary_lines(poly: np.ndarray, wall_ds: np.ndarray) -> tuple[list[Line2D], list[Opening]]:
    lines: list[Line2D] = []
    openings: list[Opening] = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 0.05:
            continue
        direction = delta / length
        normal = np.array([-direction[1], direction[0]])
        t = (wall_ds - a) @ direction
        dist = np.abs((wall_ds - a) @ normal)
        support = int(((t >= -0.05) & (t <= length + 0.05) & (dist < 0.55)).sum())
        lines.append(
            Line2D(
                origin=a.copy(),
                direction=direction,
                normal=normal,
                t_min=0.0,
                t_max=length,
                residual=0.02,
                gaps=[],
                z_span=0.0,
                n_points=support,
            )
        )
        # About one wall point per 8 cm means the edge is a real surface.
        if support >= max(4, int(length / 0.08)):
            continue
        if length < 0.45:
            continue
        kind = "door" if length <= 1.20 else "opening"
        openings.append(Opening(0.0, length, kind, len(lines) - 1))
    return lines, openings



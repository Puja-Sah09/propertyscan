"""Geometry tests on a two-room fixture the pipeline is not allowed to special-case."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propertyscan.geometry import (
    Edge,
    extract_rooms,
    icp_se2,
    optimize_pose_graph,
    polygon_area,
    se2_apply,
    wrap_angle,
)


def _wall_points(x0, y0, x1, y1, z0, z1, n_along, n_up, skip=None, rng=None):
    rng = rng or np.random.default_rng(0)
    a = np.linspace(0, 1, n_along)
    b = np.linspace(0, 1, n_up)
    aa, bb = np.meshgrid(a, b)
    xs = x0 + (x1 - x0) * aa.ravel()
    ys = y0 + (y1 - y0) * bb.ravel()
    # For vertical walls one of x or y is constant; z comes from the other parameter.
    # Caller passes a vertical segment: either x0==x1 or y0==y1, and we overwrite z.
    zs = z0 + (z1 - z0) * bb.ravel()
    if abs(x1 - x0) >= abs(y1 - y0):
        ys = np.full_like(xs, y0)
        xs = x0 + (x1 - x0) * aa.ravel()
    else:
        xs = np.full_like(ys, x0)
        ys = y0 + (y1 - y0) * aa.ravel()
    pts = np.stack([xs, ys, zs], axis=1)
    if skip is not None:
        s0, s1 = skip
        if abs(x1 - x0) >= abs(y1 - y0):
            coord = pts[:, 0]
            lo, hi = (x0, x1) if x0 < x1 else (x1, x0)
        else:
            coord = pts[:, 1]
            lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        t = (coord - lo) / (hi - lo + 1e-9)
        # skip is in absolute coordinate along the varying axis
        pts = pts[(coord < s0) | (coord > s1)]
    noise = rng.normal(0, 0.004, size=pts.shape)
    noise[:, 2] *= 0.5
    return pts + noise


def two_room_cloud(seed=0):
    rng = np.random.default_rng(seed)
    # Room A [0,4] x [0,3] ceiling 2.50. Room B [4,6.5] x [0,3] ceiling 2.40.
    # Door on x=4, y in [1.0, 1.8].
    chunks = []
    # A walls
    chunks.append(_wall_points(0, 0, 4, 0, 0.05, 2.45, 80, 30, rng=rng))  # south
    chunks.append(_wall_points(0, 3, 4, 3, 0.05, 2.45, 80, 30, rng=rng))  # north
    chunks.append(_wall_points(0, 0, 0, 3, 0.05, 2.45, 60, 30, rng=rng))  # west
    chunks.append(_wall_points(4, 0, 4, 3, 0.05, 2.45, 90, 36, skip=(1.0, 1.8), rng=rng))
    # Jambs themselves are wall, so the empty span is exactly the opening.
    for z in np.linspace(0.05, 2.30, 24):
        chunks.append(np.array([[4.0, 1.0, z], [4.0, 1.8, z]]) + rng.normal(0, 0.002, size=(2, 3)))
    # B walls
    chunks.append(_wall_points(4, 0, 6.5, 0, 0.05, 2.35, 50, 30, rng=rng))
    chunks.append(_wall_points(4, 3, 6.5, 3, 0.05, 2.35, 50, 30, rng=rng))
    chunks.append(_wall_points(6.5, 0, 6.5, 3, 0.05, 2.35, 60, 30, rng=rng))
    chunks.append(_wall_points(4, 0, 4, 3, 0.05, 2.20, 90, 36, skip=(1.0, 1.8), rng=rng))
    walls = np.vstack(chunks)
    # Floor and ceiling samples.
    def panel(x0, y0, x1, y1, z, n=40):
        xs = rng.uniform(x0, x1, n * n)
        ys = rng.uniform(y0, y1, n * n)
        zs = np.full(n * n, z) + rng.normal(0, 0.003, n * n)
        return np.stack([xs, ys, zs], 1)

    floor = np.vstack([panel(0.05, 0.05, 3.95, 2.95, 0.0, 25), panel(4.05, 0.05, 6.45, 2.95, 0.0, 20)])
    ceil = np.vstack([panel(0.05, 0.05, 3.95, 2.95, 2.50, 20), panel(4.05, 0.05, 6.45, 2.95, 2.40, 16)])
    path = np.array([[2.0, 1.4], [3.3, 1.4], [3.7, 1.4], [4.3, 1.4], [5.2, 1.5]])
    return walls, floor, ceil, path


def test_icp_recovers_known_motion():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(80, 2))
    yaw, trans = 0.18, np.array([0.35, -0.12])
    dst = se2_apply(yaw, trans, src)
    # A second cloud the ICP treats as the target; recover the transform of src onto dst.
    hat_yaw, hat_t, resid, n = icp_se2(src, dst, init_yaw=0.0, init_trans=np.zeros(2))
    assert resid < 0.02
    assert abs(wrap_angle(hat_yaw - yaw)) < 0.02
    assert np.linalg.norm(hat_t - trans) < 0.02


def test_pose_graph_closes_a_drifted_square():
    # 1 m square. Each step is measured 2 degrees long on yaw, so raw integration misses.
    edges = []
    step = 1.0
    bias = math.radians(2.0)
    headings = [0, math.pi / 2, math.pi, -math.pi / 2]
    for i, h in enumerate(headings):
        j = (i + 1) % 4
        # Motion in the frame at i, which already has heading h if unbiased.
        edges.append(Edge(i, j, step, 0.0, math.pi / 2 + bias, weight=1.0))
    # Loop constraint: pose 4 does not exist; edge from 3 to 0 is already there.
    # Add an explicit identity loop with high weight by splitting pose 0.
    poses = optimize_pose_graph(4, edges)
    # With a loop edge 3 -> 0 included, pose 0 is anchored and the square should
    # finish near itself. The last edge says pose 0 is one step from pose 3.
    p3 = poses[3, :2]
    # Integrated unbiased square would put pose 3 at (1, 0) after three turns? 
    # We only require the optimized loop residual to be small: applying the last
    # edge should land on pose 0.
    y, x, th = poses[3]
    c, s = math.cos(th), math.sin(th)
    e = edges[3]
    pred = np.array([y + c * e.dx - s * e.dy, x + s * e.dx + c * e.dy])
    assert np.linalg.norm(pred - poses[0, :2]) < 0.05


def test_two_room_plan_meets_lidar_tolerances():
    walls, floor, ceil, path = two_room_cloud()
    all_pts = np.vstack([walls, floor, ceil])
    rooms = extract_rooms(
        wall_xy=walls[:, :2],
        wall_z=walls[:, 2],
        all_xy=all_pts[:, :2],
        all_z=all_pts[:, 2],
        seeds_xy=path,
        beyond_xy=floor[:, :2],
        refine_openings=True,
    )
    assert len(rooms) == 2, [r.area for r in rooms]
    areas = sorted(r.area for r in rooms)
    assert abs(areas[0] - 7.5) < 0.4, areas
    assert abs(areas[1] - 12.0) < 0.5, areas
    by_area = sorted(rooms, key=lambda r: r.area)
    small, big = by_area
    assert abs(big.ceiling_height - 2.50) <= 0.015, big.ceiling_height
    assert abs(small.ceiling_height - 2.40) <= 0.015, small.ceiling_height
    big_lengths = sorted(big.wall_lengths())
    # 3, 3, 4, 4
    assert abs(big_lengths[0] - 3.0) < 0.03, big_lengths
    assert abs(big_lengths[2] - 4.0) < 0.03, big_lengths
    assert len(big.openings) >= 1
    width = max(op.width for op in big.openings)
    assert abs(width - 0.80) <= 0.02, width
    # No phantom second opening.
    assert len(big.openings) == 1
    assert polygon_area(big.polygon) > 0

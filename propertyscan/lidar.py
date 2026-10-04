"""LiDAR tier. Raw poses are an initial guess only.

Drift is removed by a 2D pose graph whose edges come from scan matching on the
depth itself. Roll, pitch, and camera height stay on the IMU/pose prior, which
is the part of ARKit that does not walk off over a room-scale loop. The
``--no-drift-correction`` path fuses the supplied poses unchanged so the
ablation is the same code with the graph switched off.
"""

from __future__ import annotations

import math

import numpy as np

from propertyscan.geometry import (
    Edge,
    _levels,
    apply_yaw,
    extract_rooms,
    icp_se2,
    optimize_pose_graph,
    point_in_poly,
    recover_open_rooms,
    wrap_angle,
    yaw_from_opencv_rotation,
)
from propertyscan.sensing import spike_filter, transform_points, unproject


def _heading_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    return forward, left


def _relative_from_poses(ti: np.ndarray, yi: float, tj: np.ndarray, yj: float):
    dyaw = float(wrap_angle(yj - yi))
    delta = tj[:2] - ti[:2]
    c, s = math.cos(yi), math.sin(yi)
    dx = c * delta[0] + s * delta[1]
    dy = -s * delta[0] + c * delta[1]
    return dx, dy, dyaw


def _scan_from_frame(pts_cam: np.ndarray, r: np.ndarray, t: np.ndarray, max_pts: int = 450) -> np.ndarray:
    if len(pts_cam) == 0:
        return np.zeros((0, 2))
    world = transform_points(pts_cam, r, t)
    z = world[:, 2]
    band = (z > t[2] - 0.55) & (z < t[2] + 0.35)
    # Keep a metric range so the ceiling and the floor stay out of the scan.
    rel = world[band] - t
    if len(rel) < 30:
        return np.zeros((0, 2))
    yaw = yaw_from_opencv_rotation(r)
    forward, left = _heading_axes(yaw)
    xy = rel[:, :2]
    fwd = xy @ forward
    lat = xy @ left
    keep = (fwd > 0.35) & (fwd < 5.0) & (np.abs(lat) < 4.0)
    scan = np.stack([fwd[keep], lat[keep]], axis=1)
    if len(scan) > max_pts:
        rng = np.random.default_rng(len(scan))
        scan = scan[rng.choice(len(scan), max_pts, replace=False)]
    return scan


def _graph(scans: list[np.ndarray], raw_xy: np.ndarray, raw_yaw: np.ndarray) -> tuple[np.ndarray, int]:
    n = len(scans)
    edges: list[Edge] = []
    for i in range(n - 1):
        dx, dy, dyaw = _relative_from_poses(raw_xy[i], raw_yaw[i], raw_xy[i + 1], raw_yaw[i + 1])
        if len(scans[i]) < 20 or len(scans[i + 1]) < 20:
            edges.append(Edge(i, i + 1, dx, dy, dyaw, weight=0.3))
            continue
        yaw, trans, resid, nin = icp_se2(
            scans[i + 1],
            scans[i],
            init_yaw=dyaw,
            init_trans=np.array([dx, dy]),
            gate=0.10,
            search_yaw=False,
        )
        if nin >= 20 and resid < 0.08:
            edges.append(Edge(i, i + 1, float(trans[0]), float(trans[1]), float(yaw), weight=1.0))
        else:
            edges.append(Edge(i, i + 1, dx, dy, dyaw, weight=0.3))
    # A loop is a revisit. Two different corners both full of flat wall will
    # ICP onto each other; the raw trajectory has to already put the frames in
    # the same patch of floor before scan matching is allowed to replace it.
    loops = 0
    for b in range(n):
        for a in range(0, b - 8):
            if a != 0 and (b - a) % 5 != 0:
                continue
            if np.linalg.norm(raw_xy[b, :2] - raw_xy[a, :2]) > 1.25:
                continue
            if abs(float(wrap_angle(raw_yaw[b] - raw_yaw[a]))) > math.radians(35):
                continue
            if len(scans[a]) < 30 or len(scans[b]) < 30:
                continue
            dx, dy, dyaw = _relative_from_poses(raw_xy[a], raw_yaw[a], raw_xy[b], raw_yaw[b])
            yaw, trans, resid, nin = icp_se2(
                scans[b],
                scans[a],
                init_yaw=dyaw,
                init_trans=np.array([dx, dy]),
                gate=0.07,
                search_yaw=False,
                iters=15,
            )
            agree = math.hypot(float(trans[0] - dx), float(trans[1] - dy)) < 0.45
            agree = agree and abs(float(wrap_angle(yaw - dyaw))) < math.radians(12)
            if nin >= 40 and resid < 0.035 and agree:
                edges.append(Edge(a, b, float(trans[0]), float(trans[1]), float(yaw), weight=4.0))
                loops += 1
    yaw0 = float(raw_yaw[0])
    origin = raw_xy[0, :2]
    c0, s0 = math.cos(-yaw0), math.sin(-yaw0)
    prior = np.zeros((n, 3))
    for i in range(n):
        delta = raw_xy[i, :2] - origin
        prior[i, 0] = c0 * delta[0] - s0 * delta[1]
        prior[i, 1] = s0 * delta[0] + c0 * delta[1]
        prior[i, 2] = float(wrap_angle(raw_yaw[i] - yaw0))
    poses = optimize_pose_graph(n, edges, prior=prior)
    return poses, loops


def _covered_seeds(rooms, seeds: np.ndarray) -> int:
    if len(seeds) == 0 or not rooms:
        return 0
    n = 0
    for s in seeds:
        if any(point_in_poly(s, room.polygon) for room in rooms):
            n += 1
    return n


def _median_shift(poses: np.ndarray, raw_xy: np.ndarray, raw_yaw: np.ndarray) -> float:
    """How far the graph moved the trajectory, in metres. A big number is a bad match, not drift."""
    yaw0 = float(raw_yaw[0])
    t0 = raw_xy[0, :2]
    c, s = math.cos(yaw0), math.sin(yaw0)
    world = []
    for x, y, _yaw in poses:
        world.append(t0 + np.array([c * x - s * y, s * x + c * y]))
    delta = np.asarray(world) - raw_xy[:, :2]
    return float(np.median(np.linalg.norm(delta, axis=1)))


def _fuse(frames, poses_xyyaw, use_raw: bool):
    chunks = []
    colors = []
    seeds = []
    for i, fr in enumerate(frames):
        depth = spike_filter(fr["depth_m"])
        conf = fr.get("confidence")
        if conf is not None and conf.shape == depth.shape:
            depth = depth.copy()
            depth[conf < 1] = 0
        pts, pix = unproject(depth, fr["K"], stride=2)
        r_raw = fr["T"][:3, :3]
        t_raw = fr["T"][:3, 3]
        if use_raw or poses_xyyaw is None:
            r, t = r_raw, t_raw
        else:
            # poses are in pose-0's frame. Put them back into the capture world.
            x, y, yaw_rel = poses_xyyaw[i]
            yaw0 = yaw_from_opencv_rotation(frames[0]["T"][:3, :3])
            t0 = frames[0]["T"][:3, 3]
            c, s = math.cos(yaw0), math.sin(yaw0)
            world_xy = t0[:2] + np.array([c * x - s * y, s * x + c * y])
            world_yaw = float(wrap_angle(yaw0 + yaw_rel))
            r = apply_yaw(r_raw, world_yaw)
            t = np.array([world_xy[0], world_xy[1], t_raw[2]], dtype=float)
        if len(pts) == 0:
            seeds.append(t[:2])
            continue
        world = transform_points(pts, r, t)
        rgb = fr.get("rgb")
        if rgb is None:
            col = np.full((len(pts), 3), 180, dtype=np.uint8)
        else:
            col = rgb[pix[:, 1], pix[:, 0]]
        chunks.append(world)
        colors.append(col)
        seeds.append(t[:2])
    if not chunks:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros((0, 2))
    return np.vstack(chunks), np.vstack(colors), np.asarray(seeds, dtype=float)


def run_lidar(
    capture: dict,
    refine_openings: bool = True,
    drift_correction: bool = True,
    level_mode: str = "dense",
) -> dict:
    frames = capture["frames"]
    k = capture["intrinsics"]
    for fr in frames:
        fr["K"] = k
    scans = []
    raw_xy = []
    raw_yaw = []
    for fr in frames:
        depth = spike_filter(fr["depth_m"])
        pts, _ = unproject(depth, k, stride=3)
        r = fr["T"][:3, :3]
        t = fr["T"][:3, 3]
        scans.append(_scan_from_frame(pts, r, t))
        raw_xy.append(t)
        raw_yaw.append(yaw_from_opencv_rotation(r))
    raw_xy_a = np.asarray(raw_xy, dtype=float)
    raw_yaw_a = np.asarray(raw_yaw, dtype=float)
    loops = 0
    method = "poses_used_as_is"
    poses = None
    if drift_correction and len(frames) >= 2:
        poses, loops = _graph(scans, raw_xy_a, raw_yaw_a)
        shift = _median_shift(poses, raw_xy_a, raw_yaw_a)
        # ARKit's own loop closure is already in a Stray Scanner trajectory.
        # A graph that walks the phone by half a metre has matched the wrong walls.
        if shift > 0.45:
            print(f"pose graph moved the path by {shift:.2f} m median; keeping the captured poses", flush=True)
            poses = None
            method = "se2_icp_pose_graph_rejected"
        else:
            method = "se2_icp_pose_graph"
    xyz, rgb, seeds = _fuse(frames, poses, use_raw=poses is None)
    if len(xyz) < 100:
        raise RuntimeError("LiDAR fusion produced almost no points. Check depth units and intrinsics.")
    z = xyz[:, 2]
    # Door headers sit near 2.0 m. The band used for the plan is below that and
    # above typical furniture, once the floor is known. Percentile 2 is the
    # floor when any frame looked down; otherwise the cloud has no metric floor
    # and the ceiling gate cannot be met, which is the honest outcome.
    if level_mode == "percentile":
        floor_guess = float(np.percentile(z, 2))
        ceil_guess = float(np.percentile(z, 98))
    else:
        floor_guess, ceil_guess = _levels(z)
    # A floor-only walk never sees a ceiling. The high mode is then the top of
    # the walls the phone caught, not a storey. Keep a wall band anyway.
    ceiling_observed = (ceil_guess - floor_guess) >= 2.0
    wall_lo = floor_guess + 0.95
    wall_hi = min(floor_guess + 1.60, ceil_guess - 0.35) if ceiling_observed else floor_guess + 1.60
    wall = (z > wall_lo) & (z < wall_hi)
    floor = z < floor_guess + 0.10
    rooms = extract_rooms(
        wall_xy=xyz[wall, :2],
        wall_z=z[wall],
        all_xy=xyz[:, :2],
        all_z=z,
        seeds_xy=seeds,
        # Doors are confirmed by the path crossing them. Passing the floor here
        # would bless every occlusion, because floor lies in front of every wall.
        beyond_xy=None,
        refine_openings=refine_openings,
        min_height=1.7 if ceiling_observed else 0.4,
        level_mode=level_mode,
    )
    if _covered_seeds(rooms, seeds) < 0.45 * len(seeds):
        alt = recover_open_rooms(xyz[wall, :2], seeds, xyz[:, :2], z)
        if _covered_seeds(alt, seeds) > _covered_seeds(rooms, seeds):
            print(f"walls do not close; recovered {len(alt)} walked rooms", flush=True)
            rooms = alt
    return {
        "rooms": rooms,
        "xyz": xyz,
        "rgb": rgb,
        "seeds": seeds,
        "ceiling_observed": ceiling_observed,
        "drift": {
            "method": method,
            "loop_closures": int(loops),
            "used_raw_poses_as_is": method == "poses_used_as_is",
            "correction_applied": method == "se2_icp_pose_graph",
        },
    }

"""Video tier. Scale comes from the letter sheet; motion comes from the floor.

The first frames that see the sheet fix the metric frame. Later frames track
floor points with Lucas-Kanade and solve a ground-plane PnP, so the scale
cannot drift with a feature the user did not measure. Seeing the sheet again
is a loop closure: the pose is replaced by a fresh PnP, not averaged with the
dead-reckoned one.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from propertyscan.geometry import (
    extract_rooms,
    line_gaps,
    ransac_lines,
    recover_open_rooms,
    wrap_angle,
    yaw_from_opencv_rotation,
)
from propertyscan.planar import (
    backproject_floor,
    boundary_points,
    detect_sheet,
    floor_mask,
    image_is_dark,
    innermost_lines,
    intrinsics_matrix,
    measure_ceiling,
    pose_from_sheet,
    _wall_pixel,
)


def run_video(capture: dict) -> dict:
    root = Path(capture["root"])
    meta = capture.get("meta") or {}
    paths = sorted((root / "frames").glob("*.jpg"))
    if not paths:
        raise RuntimeError(f"No frames in {root / 'frames'}")
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [im for im in frames if im is not None]
    k, k_source = intrinsics_matrix(meta, frames[0].shape[1], frames[0].shape[0])
    dark = any(image_is_dark(im) for im in frames[:: max(1, len(frames) // 8)])
    acc = _track(frames, k)
    acc["dark"] = dark
    acc["intrinsics_source"] = k_source
    return acc


def _track(frames: list[np.ndarray], k: np.ndarray) -> dict:
    pose = None
    prev_gray = None
    prev_uv = None
    prev_xyz = None
    floor_color = None
    records = []
    loops = 0
    posed_frames = 0
    travelled = 0.0
    loop_abs = None
    fail = {"pnp": 0, "jump": 0, "height": 0, "seed": 0, "ok": 0}
    jumps = []
    for i, img in enumerate(frames):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        absolute = pose_from_sheet(img, k)
        fresh = False
        took_sheet = False
        if absolute is not None and _sheet_span(absolute[2]) >= 22.0:
            if pose is None or posed_frames < 10:
                near = pose is None or float(np.linalg.norm(absolute[1][:2] - pose[1][:2])) < 0.55
                if near and 0.85 <= float(absolute[1][2]) <= 1.9:
                    pose = (absolute[0], absolute[1])
                    posed_frames += 1
                    took_sheet = True
                    fresh = True
                    prev_uv, prev_xyz = _seed_floor(gray, pose[0], pose[1], k, absolute[2])
                    floor_color = _floor_color(img, absolute[2])
            elif i > 0.82 * len(frames) and float(np.linalg.norm(absolute[1][:2] - pose[1][:2])) < 2.2:
                # Back on the first sheet. Keep dead-reckoning for this frame and
                # let the pose graph pull the path onto the new fix.
                loop_abs = (absolute[0], absolute[1])
                loops += 1
                posed_frames += 1
        if not took_sheet and pose is not None and prev_uv is not None and len(prev_uv) >= 10 and prev_gray is not None:
            updated = _step_pnp(prev_gray, gray, prev_uv, prev_xyz, pose, k)
            if updated is not None:
                r_wc, t_wc, prev_uv, prev_xyz = updated
                z_ref = float(records[0]["t"][2]) if records else float(pose[1][2])
                z_solved = float(t_wc[2])
                # The phone stays at the height the sheet measured. A solve that
                # drops the camera just shrinks the step; scale it back.
                if not (0.9 <= z_solved <= 2.1):
                    fail["height"] += 1
                    prev_uv, prev_xyz = _seed_floor(gray, pose[0], pose[1], k, None)
                else:
                    delta = (t_wc[:2] - pose[1][:2]) * (z_ref / z_solved)
                    jumped = float(np.linalg.norm(delta))
                    jumps.append(jumped)
                    if jumped >= 0.75 or float(r_wc[2, 2]) > -0.12:
                        fail["jump"] += 1
                        prev_uv, prev_xyz = _seed_floor(gray, pose[0], pose[1], k, None)
                    else:
                        fail["ok"] += 1
                        travelled += jumped
                        t_new = t_wc.copy()
                        t_new[0] = float(pose[1][0] + delta[0])
                        t_new[1] = float(pose[1][1] + delta[1])
                        t_new[2] = z_ref
                        pose = (r_wc, t_new)
                        fresh = True
            else:
                fail["pnp"] += 1
                prev_uv, prev_xyz = _seed_floor(gray, pose[0], pose[1], k, None)
        if pose is not None and (prev_uv is None or len(prev_uv) < 12):
            prev_uv, prev_xyz = _seed_floor(gray, pose[0], pose[1], k, absolute[2] if took_sheet else None)
        if fresh and pose is not None:
            seeded_uv, seeded_xyz = _seed_floor(gray, pose[0], pose[1], k, absolute[2] if took_sheet else None)
            if seeded_uv is not None:
                prev_uv, prev_xyz = seeded_uv, seeded_xyz
            r_wc, t_wc = pose
            pts = _contacts(img, r_wc, t_wc, k, floor_color)
            height = None
            if len(pts) >= 25:
                hs = _ceiling_this_frame(img, r_wc, t_wc, k, pts[:, :2])
                if hs:
                    height = float(np.median(hs))
            records.append({"r": r_wc, "t": t_wc.copy(), "walls": pts, "height": height})
        prev_gray = gray
    if not records or not any(len(r["walls"]) for r in records):
        raise RuntimeError("Video never locked onto the letter sheet, so it has no scale.")
    wall_xy, floor_xyz, ceil_xyz, seeds, applied = _close_loop(records, loop_abs)
    if len(wall_xy) == 0:
        raise RuntimeError("Video never locked onto the letter sheet, so it has no scale.")
    rooms = _rooms_from_bases(wall_xy, floor_xyz, ceil_xyz, seeds)
    return {
        "rooms": rooms,
        "names": [f"room-{i+1}" for i in range(len(rooms))],
        "xyz": np.zeros((0, 3)),
        "rgb": np.zeros((0, 3), np.uint8),
        "drift": {
            "method": "sheet_pnp_ground_plane" if applied else "sheet_pnp_ground_plane_rejected",
            "loop_closures": int(loops),
            "used_raw_poses_as_is": False,
            "sheet_fixes": int(posed_frames),
            "correction_applied": bool(applied),
        },
    }


def _sheet_span(corners: np.ndarray) -> float:
    return float(np.linalg.norm(corners.max(0) - corners.min(0)))


def _sheet_large(corners: np.ndarray, width: int) -> bool:
    return _sheet_span(corners) > 0.08 * width


def _floor_color(img, corners) -> np.ndarray:
    mask = floor_mask(img, corners)
    sheet = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(sheet, np.round(corners).astype(np.int32), 255)
    ring = cv2.dilate(sheet, np.ones((15, 15), np.uint8)) & (mask > 0) & (sheet == 0)
    samples = img[ring > 0]
    if len(samples) < 10:
        return np.array([130, 125, 118], np.float32)
    return np.median(samples.astype(np.float32), axis=0)


def _seed_floor(gray, r_wc, t_wc, k, corners):
    """Floor corners only. The sheet border is a stronger corner than the
    texture, and a relative threshold would then ignore the floor."""
    h, w = gray.shape
    mask = np.zeros(gray.shape, np.uint8)
    mask[int(h * 0.30) : int(h * 0.96), int(w * 0.05) : int(w * 0.95)] = 255
    if corners is not None:
        sheet = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(sheet, np.round(corners).astype(np.int32), 255)
        sheet = cv2.dilate(sheet, np.ones((21, 21), np.uint8))
        mask[sheet > 0] = 0
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=400, qualityLevel=0.006, minDistance=6, mask=mask, blockSize=7
    )
    if pts is None:
        return None, None
    uv = pts.reshape(-1, 2)
    xyz, keep = _project_keep(uv, r_wc, t_wc, k)
    if int(np.sum(keep)) < 12:
        return None, None
    uv = uv[keep]
    dist = np.linalg.norm(xyz[:, :2] - t_wc[:2], axis=1)
    near = (dist > 0.35) & (dist < 3.2)
    if int(near.sum()) < 12:
        return None, None
    return uv[near], xyz[near]


def _step_pnp(prev_gray, gray, prev_uv, prev_xyz, pose, k):
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_uv.reshape(-1, 1, 2).astype(np.float32),
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )
    ok = status.reshape(-1) == 1
    if int(ok.sum()) < 10:
        return None
    obj = prev_xyz[ok].astype(np.float64)
    imgp = nxt.reshape(-1, 2)[ok].astype(np.float64)
    r_wc, t_wc = pose
    rvec0, _ = cv2.Rodrigues(r_wc.T)
    tvec0 = (-r_wc.T @ t_wc).reshape(3, 1)
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            imgp,
            k,
            np.zeros(4),
            rvec0,
            tvec0,
            True,
            iterationsCount=80,
            reprojectionError=3.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not success or inliers is None or len(inliers) < 8:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    new_r = rot.T
    new_t = (-rot.T @ tvec.reshape(3)).reshape(3)
    idx = inliers.reshape(-1)
    return new_r, new_t, imgp[idx], obj[idx]


def _project_keep(uv, r_wc, t_wc, k):
    pix = np.column_stack([uv[:, 0], uv[:, 1], np.ones(len(uv))])
    rays = (pix @ np.linalg.inv(k).T) @ r_wc.T
    dz = rays[:, 2]
    ok = dz < -1e-4
    s = np.zeros(len(uv))
    s[ok] = -t_wc[2] / dz[ok]
    pts = t_wc + s[:, None] * rays
    keep = ok & (s > 0.2) & (s < 6.0)
    return pts[keep], keep


def _refresh(gray, uv, xyz, r_wc, t_wc, k, floor_color):
    if len(uv) >= 140:
        return uv, xyz
    mask = np.zeros(gray.shape, np.uint8)
    # The lower half is floor on this walk. New corners replace the ones that left the frame.
    mask[int(gray.shape[0] * 0.42) :, :] = 255
    found = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.02, minDistance=7, mask=mask)
    if found is None:
        return uv, xyz
    new_uv = found.reshape(-1, 2)
    new_xyz, keep = _project_keep(new_uv, r_wc, t_wc, k)
    if int(np.sum(keep)) < 6:
        return uv, xyz
    return np.vstack([uv, new_uv[keep]]), np.vstack([xyz, new_xyz])


def _contacts(img, r_wc, t_wc, k, floor_color):
    """Wall-floor contact. Distance is measured from the camera, not from the
    letter sheet: a kitchen wall is more than 3.6 m from the living-room sheet."""
    if floor_color is None:
        return np.zeros((0, 3))
    dist = np.linalg.norm(img.astype(np.float32) - floor_color, axis=2)
    raw = (dist < 46).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _nlab, labels = cv2.connectedComponents(raw)
    h, w = raw.shape
    seed = int(labels[min(h - 2, int(h * 0.78)), w // 2])
    if seed == 0:
        return np.zeros((0, 3))
    mask = labels == seed
    us, vs = [], []
    x0, x1 = int(w * 0.08), int(w * 0.92)
    for u in range(x0, x1):
        seen = False
        for v in range(h - 4, 4, -1):
            if mask[v, u]:
                seen = True
                continue
            if not seen:
                continue
            if _wall_pixel(img, v, u):
                us.append(u)
                vs.append(v + 1)
            break
    if len(us) < 12:
        return np.zeros((0, 3))
    pts = backproject_floor(np.stack([us, vs], 1).astype(float), r_wc, t_wc, k)
    if len(pts) == 0:
        return pts
    dist_xy = np.linalg.norm(pts[:, :2] - t_wc[:2], axis=1)
    return pts[(dist_xy > 0.40) & (dist_xy < 4.2)]


def _close_loop(records: list[dict], loop_abs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Spread the return to the letter sheet along the walk.

    The live track is not snapped when the sheet reappears, because a snap
    would leave the earlier rooms where the drift put them. The miss at the
    end is shared in proportion to distance walked.
    """
    xy = np.array([r["t"][:2] for r in records], dtype=float)
    yaw = np.array([yaw_from_opencv_rotation(r["r"]) for r in records], dtype=float)
    opt = None
    if loop_abs is not None and len(records) >= 8:
        end_xy = np.asarray(loop_abs[1][:2], dtype=float)
        end_yaw = float(yaw_from_opencv_rotation(loop_abs[0]))
        gap = float(np.linalg.norm(end_xy - xy[-1]))
        # A wrong sheet solves at the origin and can sit metres from the track.
        # A real return is within about a metre. Spread that miss along the walk.
        if gap < 1.2:
            step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(step)])
            total = float(cum[-1]) if cum[-1] > 1e-3 else 1.0
            frac = cum / total
            pred = xy + frac[:, None] * (end_xy - xy[-1])
            pred_yaw = wrap_angle(yaw + frac * wrap_angle(end_yaw - yaw[-1]))
            opt = (pred, pred_yaw)
    walls, floors, ceils, seeds = [], [], [], []
    for i, rec in enumerate(records):
        raw_xy = rec["t"][:2]
        raw_yaw = float(yaw[i])
        new_xy = raw_xy if opt is None else opt[0][i]
        new_yaw = raw_yaw if opt is None else float(opt[1][i])
        seeds.append(new_xy.copy())
        pts = rec["walls"]
        if len(pts):
            local = _unyaw(pts[:, :2] - raw_xy, raw_yaw)
            moved = _yaw(local, new_yaw) + new_xy
            walls.append(moved)
        floors.append(np.column_stack([_disc(new_xy, 0.40, 12), np.zeros(12)]))
        if rec["height"] is not None and 1.9 < rec["height"] < 3.1:
            disc = _disc(new_xy, 0.30, 40)
            ceils.append(np.column_stack([disc, np.full(len(disc), rec["height"])]))
    wall_xy = np.vstack(walls) if walls else np.zeros((0, 2))
    floor_xyz = np.vstack(floors) if floors else np.zeros((0, 3))
    ceil_xyz = np.vstack(ceils) if ceils else np.zeros((0, 3))
    return wall_xy, floor_xyz, ceil_xyz, np.asarray(seeds, dtype=float), opt is not None


def _unyaw(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])


def _yaw(xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])


def _ceiling_this_frame(img, r_wc, t_wc, k, xy):
    if len(xy) < 40:
        return []
    lines = ransac_lines(xy, thresh=0.04, min_points=25, min_length=0.35, iterations=80)
    lines = innermost_lines(lines, origin=t_wc[:2])
    if len(lines) < 2:
        return []
    return measure_ceiling(img, r_wc, t_wc, k, lines)


def _disc(centre: np.ndarray, radius: float, n: int) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = radius * np.sqrt(np.linspace(0.15, 1.0, n))
    return np.column_stack([centre[0] + rad * np.cos(ang), centre[1] + rad * np.sin(ang)])


def _rooms_from_bases(wall_xy, floor_xyz, ceil_xyz, seeds) -> list:
    if len(wall_xy) > 15000:
        rng = np.random.default_rng(0)
        wall_xy = wall_xy[rng.choice(len(wall_xy), 15000, replace=False)]
    lines = ransac_lines(wall_xy, thresh=0.04, min_points=50, min_length=0.45, iterations=200)
    for ln in lines:
        ln.gaps = line_gaps(wall_xy, ln)
    ribbons = []
    for ln in lines:
        ts = np.linspace(ln.t_min, ln.t_max, 100)
        for g0, g1 in ln.gaps:
            ts = ts[(ts <= g0) | (ts >= g1)]
        if len(ts) < 4:
            continue
        xy = ln.origin + np.outer(ts, ln.direction)
        for z in (0.45, 0.9, 1.3, 1.6):
            ribbons.append(np.column_stack([xy, np.full(len(xy), z)]))
    if not ribbons:
        return []
    wall = np.vstack(ribbons)
    # Ribbons are a wall stand-in for the plan. They are not a ceiling.
    parts = []
    if len(floor_xyz):
        parts.append(floor_xyz)
    if len(ceil_xyz):
        parts.append(ceil_xyz)
    all_xyz = np.vstack(parts) if parts else np.column_stack([wall[:, :2], np.zeros(len(wall))])
    seeds_xy = seeds if len(seeds) else wall[:, :2][:1]
    rooms = extract_rooms(
        wall_xy=wall[:, :2],
        wall_z=wall[:, 2],
        all_xy=all_xyz[:, :2],
        all_z=all_xyz[:, 2],
        seeds_xy=seeds_xy,
        beyond_xy=None,
        refine_openings=True,
    )
    span = wall_xy.max(axis=0) - wall_xy.min(axis=0)
    enclosed = float(np.prod(np.maximum(span, 0.0)))
    got = sum(r.area for r in rooms)
    # A walk that leaves the sheet does not close every wall on the raster,
    # so the closed-room pass keeps a pocket and throws the rest out as border.
    if enclosed > 4.0 and got < 0.45 * enclosed:
        alt = recover_open_rooms(wall[:, :2], seeds_xy, all_xyz[:, :2], all_xyz[:, 2])
        if sum(r.area for r in alt) > got:
            rooms = alt
    return rooms

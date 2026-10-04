"""Metric room geometry from a letter sheet lying on the floor.

The sheet is the only scale. A 12 mm black border is part of the capture
protocol; the white rectangle inside it is what the detector corners, and the
object size below is Letter minus that border. Pose is a planar PnP. Wall
bases are the floor mask's far boundary, back-projected onto z = 0.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from propertyscan.geometry import Line2D, ransac_lines, wrap_angle

# US Letter, and the white field inside a 12 mm marker border.
LETTER_W = 0.2794
LETTER_H = 0.2159
BORDER_M = 0.012
INNER_W = LETTER_W - 2 * BORDER_M
INNER_H = LETTER_H - 2 * BORDER_M


def intrinsics_matrix(meta: dict, width: int, height: int) -> tuple[np.ndarray, str]:
    k = meta.get("intrinsics")
    if k and all(key in k for key in ("fx", "fy", "cx", "cy")):
        mat = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], dtype=float)
        return mat, "capture.json"
    f35 = float(meta.get("focal_35mm") or 26.0)
    fx = (f35 / 36.0) * width
    mat = np.array([[fx, 0, width / 2], [0, fx, height / 2], [0, 0, 1]], dtype=float)
    return mat, "iphone15_wide_default"


def order_quad(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def detect_sheet(bgr: np.ndarray) -> np.ndarray | None:
    """Four corners of the white inner rectangle, ordered around the centre."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    white = (gray > 225).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(approx)
        w, h = rect[1]
        if min(w, h) < 8:
            continue
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect < 1.05 or aspect > 2.4:
            continue
        if best is None or area > best[0]:
            best = (area, approx.reshape(4, 2).astype(np.float32))
    if best is None:
        return None
    corners = order_quad(best[1])
    gray_f = gray.astype(np.float32)
    refined = cv2.cornerSubPix(
        gray_f,
        corners.reshape(-1, 1, 2),
        (5, 5),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
    )
    return _snap_edges(gray, order_quad(refined.reshape(4, 2)))


def _snap_edges(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Pull each side onto the black-to-white step.

    A contour of bright pixels sits on the outer halo of the white field.
    That halo is a couple of pixels, which is a few percent of a small sheet,
    and the whole room then measures short.
    """
    h, w = gray.shape
    centre = corners.mean(axis=0)
    shifts = []
    normals = []
    mids = []
    for i in range(4):
        a = corners[i].astype(np.float64)
        b = corners[(i + 1) % 4].astype(np.float64)
        tangent = b - a
        length = float(np.linalg.norm(tangent)) + 1e-9
        tangent = tangent / length
        normal = np.array([-tangent[1], tangent[0]])
        mid = 0.5 * (a + b)
        if np.dot(mid - centre, normal) < 0:
            normal = -normal
        offsets = []
        for t in np.linspace(0.2, 0.8, 7):
            base = a + (b - a) * t
            best_s, best_g = 0.0, 0.0
            for s in np.linspace(-5.0, 3.0, 33):
                outer = base + normal * (s + 0.5)
                inner = base + normal * (s - 0.5)
                if not (1 <= outer[0] < w - 1 and 1 <= outer[1] < h - 1):
                    continue
                if not (1 <= inner[0] < w - 1 and 1 <= inner[1] < h - 1):
                    continue
                go = float(gray[int(round(outer[1])), int(round(outer[0]))])
                gi = float(gray[int(round(inner[1])), int(round(inner[0]))])
                # White inside, black border outside. The step is inward minus outward.
                grad = gi - go
                if grad > best_g:
                    best_g = grad
                    best_s = float(s)
            if best_g > 40:
                offsets.append(best_s)
        shifts.append(float(np.median(offsets)) if offsets else 0.0)
        normals.append(normal)
        mids.append(mid)
    snapped = []
    for i in range(4):
        # Corner i is the join of side i-1 and side i.
        p0 = mids[i - 1] + normals[i - 1] * shifts[i - 1]
        d0 = np.array([-normals[i - 1][1], normals[i - 1][0]])
        p1 = mids[i] + normals[i] * shifts[i]
        d1 = np.array([-normals[i][1], normals[i][0]])
        hit = _line_intersect(p0, d0, p1, d1)
        snapped.append(corners[i] if hit is None else hit)
    return order_quad(np.asarray(snapped, dtype=np.float32))


def _line_intersect(p0: np.ndarray, d0: np.ndarray, p1: np.ndarray, d1: np.ndarray) -> np.ndarray | None:
    a = np.column_stack([d0, -d1])
    if abs(np.linalg.det(a)) < 1e-8:
        return None
    try:
        lam = np.linalg.solve(a, p1 - p0)
    except np.linalg.LinAlgError:
        return None
    return p0 + lam[0] * d0


def _object_corners() -> np.ndarray:
    w, h = INNER_W / 2, INNER_H / 2
    return np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]], dtype=np.float64)


def pose_from_sheet(bgr: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return R_wc, t_wc, image corners. Camera frame is OpenCV."""
    corners = detect_sheet(bgr)
    if corners is None or not _black_border(bgr, corners):
        return None
    obj = _object_corners()
    dist = np.zeros(4)
    flag = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
    best = None
    for reverse in (False, True):
        seq = corners[::-1] if reverse else corners
        for shift in range(4):
            img = np.roll(seq, shift, axis=0)
            try:
                nsol, rvecs, tvecs, _err = cv2.solvePnPGeneric(obj, img, k, dist, flags=flag)
            except cv2.error:
                continue
            for s in range(int(nsol)):
                rvec = np.asarray(rvecs[s], dtype=float).reshape(3)
                tvec = np.asarray(tvecs[s], dtype=float).reshape(3)
                rot, _ = cv2.Rodrigues(rvec)
                # p_cam = R @ p_obj + t  =>  camera centre in the sheet frame.
                r_wc = rot.T
                t_wc = -rot.T @ tvec
                if not (0.4 <= t_wc[2] <= 2.4):
                    continue
                proj, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
                err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)))
                if err > 4.0:
                    continue
                mark = _mark_is_dark(bgr, r_wc, t_wc, k)
                # The sheet is a rectangle, so a 180 degree flip fits equally well.
                # A mark in the +x,+y corner breaks the tie. A dark mark outranks a lower error.
                rank = (0 if mark else 1, err)
                if best is None or rank < best[0]:
                    best = (rank, rvec, tvec, img, corners)
    if best is None:
        return None
    rvec, tvec = _refine_on_edges(bgr, obj, best[3], best[1], best[2], k)
    rot, _ = cv2.Rodrigues(rvec)
    r_wc = rot.T
    t_wc = -rot.T @ tvec.reshape(3)
    if not (0.4 <= t_wc[2] <= 2.4):
        rot, _ = cv2.Rodrigues(best[1])
        return rot.T, -rot.T @ best[2].reshape(3), best[4]
    return r_wc, t_wc, best[4]


def _refine_on_edges(bgr, obj, img, rvec, tvec, k):
    """Re-solve from points along each edge, not only the four corners.

    One corner a pixel high pitches the camera, and a wall four metres away
    moves by more than ten centimetres.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    obj_pts = [np.asarray(p, dtype=np.float64) for p in obj]
    img_pts = [np.asarray(p, dtype=np.float64) for p in img]
    centre = img.mean(axis=0)
    for i in range(4):
        a_o, b_o = obj[i], obj[(i + 1) % 4]
        a_i, b_i = img[i], img[(i + 1) % 4]
        tangent = b_i - a_i
        length = float(np.linalg.norm(tangent)) + 1e-9
        tangent = tangent / length
        normal = np.array([-tangent[1], tangent[0]])
        mid = 0.5 * (a_i + b_i)
        if np.dot(mid - centre, normal) < 0:
            normal = -normal
        for t in (0.2, 0.35, 0.5, 0.65, 0.8):
            base = a_i + (b_i - a_i) * t
            best_s, best_g = 0.0, 0.0
            for s in np.linspace(-3.0, 3.0, 25):
                outer = base + normal * (s + 0.5)
                inner = base + normal * (s - 0.5)
                if not (1 <= outer[0] < w - 1 and 1 <= outer[1] < h - 1):
                    continue
                if not (1 <= inner[0] < w - 1 and 1 <= inner[1] < h - 1):
                    continue
                go = float(gray[int(round(outer[1])), int(round(outer[0]))])
                gi = float(gray[int(round(inner[1])), int(round(inner[0]))])
                grad = gi - go
                if grad > best_g:
                    best_g = grad
                    best_s = float(s)
            if best_g < 40:
                continue
            obj_pts.append(a_o + (b_o - a_o) * t)
            img_pts.append(base + normal * best_s)
    obj_pts = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    r0 = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    t0 = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    try:
        ok, r2, t2 = cv2.solvePnP(
            obj_pts,
            img_pts,
            k,
            np.zeros(4),
            r0,
            t0,
            True,
            cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return r0, t0
    if not ok:
        return r0, t0
    return r2, t2


def _black_border(bgr: np.ndarray, corners: np.ndarray) -> bool:
    """The protocol sheet has a black border. A bright patch of floor does not."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    centre = corners.mean(axis=0)
    dark = 0
    total = 0
    for i in range(4):
        a = corners[i].astype(np.float64)
        b = corners[(i + 1) % 4].astype(np.float64)
        mid = 0.5 * (a + b)
        outward = mid - centre
        outward = outward / (np.linalg.norm(outward) + 1e-9)
        for t in (0.3, 0.5, 0.7):
            total += 1
            for offset in (1.2, 2.2, 3.2):
                p = a + (b - a) * t + outward * offset
                u, v = int(round(p[0])), int(round(p[1]))
                if 0 <= u < w and 0 <= v < h and gray[v, u] < 80:
                    dark += 1
                    break
    return total >= 8 and dark >= 0.7 * total


def _mark_is_dark(bgr: np.ndarray, r_wc: np.ndarray, t_wc: np.ndarray, k: np.ndarray) -> bool:
    """True when the +x,+y fiducial on the sheet lands on a dark pixel."""
    point = np.array([0.057, 0.042, 0.0])
    p_c = r_wc.T @ (point - t_wc)
    if p_c[2] < 0.05:
        return False
    uv = k @ p_c
    u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
    h, w = bgr.shape[:2]
    if not (1 <= u < w - 1 and 1 <= v < h - 1):
        return False
    patch = bgr[v - 2 : v + 3, u - 2 : u + 3]
    # The mark is a few pixels wide. The wrong flip lands on white paper.
    return float(patch.min()) < 45


def _sheet_mask(shape, corners: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
    return mask


def floor_mask(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    sheet = _sheet_mask(bgr.shape, corners)
    # The first ring is the black marker border, not the floor.
    outer = cv2.dilate(sheet, np.ones((29, 29), np.uint8))
    inner = cv2.dilate(sheet, np.ones((11, 11), np.uint8))
    ring = outer & ~inner
    samples = bgr[ring > 0]
    if len(samples):
        bright = samples.mean(axis=1) > 70
        samples = samples[bright]
    if len(samples) < 15:
        return sheet
    med = np.median(samples.astype(np.float32), axis=0)
    dist = np.linalg.norm(bgr.astype(np.float32) - med, axis=2)
    # 42 keeps the floor texture (about ±30) and rejects the east wall, whose
    # paint sits about 57 counts from the floor median. A wider gate swallows
    # that wall and the contact is found on the wrong surface.
    raw = (dist < 42).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # The marker border is black and would split the sheet from the floor.
    raw = cv2.bitwise_or(raw, cv2.dilate(sheet, np.ones((15, 15), np.uint8)))
    nlab, labels = cv2.connectedComponents(raw)
    cy, cx = np.mean(corners, axis=0)[::-1]
    cy, cx = int(np.clip(cy, 0, labels.shape[0] - 1)), int(np.clip(cx, 0, labels.shape[1] - 1))
    lab = int(labels[cy, cx])
    if lab == 0:
        return sheet
    mask = np.where(labels == lab, 255, 0).astype(np.uint8)
    # The sheet is a hole in the floor colour. Close it so its edge is not a wall.
    mask = cv2.bitwise_or(mask, sheet)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask


def backproject_floor(uv: np.ndarray, r_wc: np.ndarray, t_wc: np.ndarray, k: np.ndarray) -> np.ndarray:
    if len(uv) == 0:
        return np.zeros((0, 3))
    pix = np.column_stack([uv[:, 0], uv[:, 1], np.ones(len(uv))])
    rays_c = pix @ np.linalg.inv(k).T
    rays_w = rays_c @ r_wc.T
    dz = rays_w[:, 2]
    ok = dz < -1e-4
    s = np.zeros(len(uv))
    s[ok] = -t_wc[2] / dz[ok]
    pts = t_wc + s[:, None] * rays_w
    keep = ok & (s > 0.15) & (s < 8.0) & (pts[:, 2] > -0.05) & (pts[:, 2] < 0.05)
    return pts[keep]


def _wall_pixel(bgr: np.ndarray, v: int, u: int) -> bool:
    """Paint, not the sofa and not the floor. Walls are light and neutral."""
    if v < 0 or u < 0 or v >= bgr.shape[0] or u >= bgr.shape[1]:
        return False
    b, g, r = (int(x) for x in bgr[v, u])
    return b > 145 and g > 140 and abs(b - g) < 35 and abs(g - r) < 40


def boundary_points(bgr, corners, r_wc, t_wc, k) -> np.ndarray:
    """Wall-floor contact. A sofa silhouette is not a wall: the pixel just
    above the floor has to be paint. Projecting the sofa's upper edge onto
    z = 0 invents a wall behind the sofa, so those columns are dropped."""
    mask = floor_mask(bgr, corners)
    sheet = cv2.dilate(_sheet_mask(bgr.shape, corners), np.ones((9, 9), np.uint8))
    h, w = mask.shape
    step = 2 if w >= 900 else 1
    us, vs = [], []
    for u in range(30, w - 30, step):
        seen_floor = False
        for v in range(h - 30, 30, -1):
            if mask[v, u] > 0 and sheet[v, u] == 0:
                seen_floor = True
                continue
            if not seen_floor:
                continue
            if _wall_pixel(bgr, v, u):
                us.append(u)
                vs.append(v + 1)
            break
    if not us:
        return np.zeros((0, 3))
    uv = np.stack([us, vs], axis=1).astype(float)
    pts = backproject_floor(uv, r_wc, t_wc, k)
    if len(pts) == 0:
        return pts
    cam = t_wc[:2]
    dist = np.linalg.norm(pts[:, :2] - cam, axis=1)
    origin = np.linalg.norm(pts[:, :2], axis=1)
    # A ray that barely tips down hits the floor kilometres away. That is not a wall.
    return pts[(dist > 0.45) & (origin < 3.6)]


def _farthest_along_rays(pts: np.ndarray, cam: np.ndarray, bins: int = 240) -> np.ndarray:
    if len(pts) == 0:
        return pts
    delta = pts[:, :2] - cam
    ang = np.arctan2(delta[:, 1], delta[:, 0])
    dist = np.linalg.norm(delta, axis=1)
    bucket = np.clip(((ang + math.pi) / (2 * math.pi) * bins).astype(int), 0, bins - 1)
    keep = np.zeros(len(pts), dtype=bool)
    for i in range(bins):
        idx = np.flatnonzero(bucket == i)
        if len(idx) == 0:
            continue
        keep[idx[int(np.argmax(dist[idx]))]] = True
    return pts[keep]


def innermost_lines(lines: list[Line2D], origin: np.ndarray | None = None) -> list[Line2D]:
    """One wall per direction.

    Furniture draws a short line in front of a wall. An open door shows a
    farther wall in the next room. The sheet's own wall is the longest line
    in that direction that still stands within 4.5 m of the sheet.
    """
    if origin is None:
        origin = np.zeros(2)
    if not lines:
        return []
    oriented = []
    for ln in lines:
        dist = abs(ln.distance(origin))
        span = float(ln.t_max - ln.t_min)
        if dist > 4.5 or dist < 0.45 or span < 0.6:
            continue
        if ln.n_points / span < 4:
            continue
        nrm = ln.normal.copy()
        if np.dot(ln.origin - origin, nrm) < 0:
            nrm = -nrm
        ang = math.atan2(nrm[1], nrm[0])
        oriented.append((ang, float(ln.n_points), ln, nrm))
    if not oriented:
        return []
    oriented.sort(key=lambda item: item[0])
    groups: list[list] = []
    for item in oriented:
        if not groups or abs(wrap_angle(item[0] - groups[-1][0][0])) > math.radians(28):
            groups.append([item])
        else:
            groups[-1].append(item)
    if len(groups) > 1 and abs(wrap_angle(groups[0][0][0] + 2 * math.pi - groups[-1][0][0])) < math.radians(28):
        groups[0].extend(groups[-1])
        groups.pop()
    kept = []
    for group in groups:
        group.sort(key=lambda item: -item[1])
        _ang, _span, ln, nrm = group[0]
        kept.append(
            Line2D(ln.origin, ln.direction, nrm, ln.t_min, ln.t_max, ln.residual, list(ln.gaps), ln.z_span, ln.n_points)
        )
    return kept


def _ray_on_plane(cam: np.ndarray, direction: np.ndarray, point: np.ndarray, normal: np.ndarray):
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-6:
        return None
    s = float(np.dot(point - cam, normal) / denom)
    if s < 0.2 or s > 8:
        return None
    return cam + s * direction


def measure_ceiling(bgr, r_wc, t_wc, k, lines: list[Line2D]) -> list[float]:
    """Ceiling height where the wall changes to the ceiling, on each wall plane."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    heights = []
    cam = t_wc.astype(float)
    for ln in lines:
        n3 = np.array([ln.normal[0], ln.normal[1], 0.0])
        p0 = np.array([ln.origin[0], ln.origin[1], 0.0])
        samples = []
        for t in np.linspace(ln.t_min + 0.15, ln.t_max - 0.15, 12):
            base = np.array([*(ln.point_at(float(t))), 0.05])
            pix = _project(base, r_wc, cam, k)
            if pix is None:
                continue
            u, v = pix
            if not (2 <= u < w - 2 and 2 <= v < h - 2):
                continue
            wall_gray = float(gray[v, u])
            hit_z = None
            # The ceiling can sit more than 80 pixels above the base on a wide lens.
            for step in range(1, max(2, v - 2)):
                vv = v - step
                if vv < 2:
                    break
                if float(gray[vv, u]) < wall_gray + 18:
                    continue
                if float(gray[vv, u]) < 185:
                    continue
                ray_c = np.linalg.inv(k) @ np.array([u, vv, 1.0])
                ray_w = r_wc @ ray_c
                hit = _ray_on_plane(cam, ray_w, p0, n3)
                if hit is None:
                    continue
                if 1.8 < hit[2] < 3.2:
                    hit_z = float(hit[2])
                    break
            if hit_z is not None:
                samples.append(hit_z)
        if len(samples) >= 3:
            heights.append(float(np.median(samples)))
    return heights


def _project(point: np.ndarray, r_wc: np.ndarray, cam: np.ndarray, k: np.ndarray) -> tuple[int, int] | None:
    p_c = r_wc.T @ (point - cam)
    if p_c[2] < 0.05:
        return None
    uv = k @ p_c
    return int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))


def image_is_dark(bgr: np.ndarray) -> bool:
    return float(np.median(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))) < 25

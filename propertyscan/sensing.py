"""Depth unprojection, spike filtering, and Record3D .r3d import.

Camera frames are OpenCV (x right, y down, z forward). Record3D stores OpenGL
camera poses (x right, y up, z backward); the importer converts them.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import cv2
import numpy as np


def spike_filter(depth_m: np.ndarray, jump: float = 0.06) -> np.ndarray:
    """Replace isolated flying pixels. Real edges move by less than `jump`."""
    d = depth_m.astype(np.float32, copy=True)
    valid = np.isfinite(d) & (d > 0)
    med = cv2.medianBlur(np.where(valid, d, 0).astype(np.float32), 3)
    # medianBlur of zeros pulls holes toward 0; only trust it where neighbors exist.
    bad = valid & (np.abs(d - med) > jump) & (med > 0.05)
    d[bad] = med[bad]
    d[~valid] = 0
    return d


def unproject(
    depth_m: np.ndarray,
    k: np.ndarray,
    stride: int = 1,
    z_min: float = 0.15,
    z_max: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera-frame points (N,3) and integer pixel coords (N,2) as (u, v)."""
    h, w = depth_m.shape
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu].astype(np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    ok = np.isfinite(z) & (z > z_min) & (z < z_max)
    z = z[ok]
    uu, vv = uu[ok], vv[ok]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)
    pix = np.stack([uu, vv], axis=1).astype(np.int32)
    return pts, pix


def transform_points(pts_cam: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return pts_cam @ r.T + t


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def opengl_pose_to_opencv(pose7: list[float]) -> np.ndarray:
    """Record3D pose [qx, qy, qz, qw, tx, ty, tz] is T_wc in OpenGL camera axes."""
    qx, qy, qz, qw, tx, ty, tz = pose7
    r_gl = quat_to_matrix(qx, qy, qz, qw)
    # p_gl = diag(1, -1, -1) @ p_cv  =>  R_cv = R_gl @ diag(1, -1, -1)
    r_cv = r_gl @ np.diag([1.0, -1.0, -1.0])
    t = np.eye(4)
    t[:3, :3] = r_cv
    t[:3, 3] = [tx, ty, tz]
    return t


def intrinsics_from_record3d(meta: dict, depth_wh: tuple[int, int] | None = None) -> np.ndarray:
    """Author's layout: K[0]=fx, K[1]=fy, K[6]=cx, K[7]=cy, in RGB pixels.

    A 3x3 list is also accepted. When depth resolution differs from RGB, K is
    scaled onto the depth grid because that is the grid we unproject.
    """
    kraw = meta["K"]
    if len(kraw) == 9:
        k = np.array(kraw, dtype=float).reshape(3, 3)
    else:
        fx, fy = float(kraw[0]), float(kraw[1])
        cx, cy = float(kraw[6]), float(kraw[7])
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    rgb_w, rgb_h = int(meta["w"]), int(meta["h"])
    if depth_wh is None:
        return k
    dw, dh = depth_wh
    if (dw, dh) != (rgb_w, rgb_h) and rgb_w > 0 and rgb_h > 0:
        k = k.copy()
        k[0, 0] *= dw / rgb_w
        k[0, 2] *= dw / rgb_w
        k[1, 1] *= dh / rgb_h
        k[1, 2] *= dh / rgb_h
    return k


def _maybe_lzfse_decompress(blob: bytes) -> bytes:
    try:
        import lzfse  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Reading Record3D depth needs the 'lzfse' package. Install it with "
            "pip install lzfse  (the photo and video tiers do not need it)."
        ) from exc
    return lzfse.decompress(blob)


def load_record3d(path: Path, frame_stride: int = 3) -> dict:
    """Load a Record3D .r3d (a zip) into the pipeline's in-memory capture."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("metadata"))
        names = set(zf.namelist())
        indices = []
        i = 0
        while f"rgbd/{i}.jpg" in names:
            indices.append(i)
            i += 1
        if not indices:
            raise ValueError(f"{path} has no rgbd/<n>.jpg frames")
        indices = indices[:: max(1, frame_stride)]
        dw, dh = int(meta["dw"]), int(meta["dh"])
        k = intrinsics_from_record3d(meta, (dw, dh))
        frames = []
        poses = meta.get("poses") or []
        for n, i in enumerate(indices):
            rgb = cv2.imdecode(np.frombuffer(zf.read(f"rgbd/{i}.jpg"), np.uint8), cv2.IMREAD_COLOR)
            depth_blob = _maybe_lzfse_decompress(zf.read(f"rgbd/{i}.depth"))
            depth = np.frombuffer(depth_blob, dtype=np.float32)
            if depth.size != dw * dh:
                raise ValueError(f"depth frame {i} has {depth.size} values, expected {dw*dh}")
            depth = depth.reshape(dh, dw)
            conf = None
            conf_name = f"rgbd/{i}.conf"
            if conf_name in names:
                conf_blob = _maybe_lzfse_decompress(zf.read(conf_name))
                conf = np.frombuffer(conf_blob, dtype=np.uint8).reshape(dh, dw)
            pose = np.eye(4)
            if i < len(poses):
                pose = opengl_pose_to_opencv(poses[i])
            frames.append({"depth_m": depth, "rgb": rgb, "confidence": conf, "T": pose})
    return {
        "tier": "lidar",
        "device": "Record3D",
        "intrinsics": k,
        "frames": frames,
        "source": str(path),
    }


def stray_pose_to_opencv_zup(qx: float, qy: float, qz: float, qw: float, x: float, y: float, z: float) -> np.ndarray:
    """Stray Scanner odometry: OpenCV camera (z forward) in a Y-up world.

    The pipeline's world is Z-up. Checked against this capture set: the phone
    stays about 1.45 m above a flat floor, and that floor is the low mode.
    """
    r_yup = quat_to_matrix(qx, qy, qz, qw)
    t_yup = np.array([x, y, z], dtype=float)
    # (x, y, z)_yup -> (x, -z, y)_zup
    s = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    out = np.eye(4)
    out[:3, :3] = s @ r_yup
    out[:3, 3] = s @ t_yup
    return out


def _stray_root(zf: zipfile.ZipFile) -> str:
    for name in zf.namelist():
        if name.endswith("odometry.csv"):
            return name[: -len("odometry.csv")]
    raise ValueError("Not a Stray Scanner archive: no odometry.csv")


def load_stray(path: Path, frame_stride: int | None = None, target_frames: int = 160) -> dict:
    """iPhone LiDAR export from Stray Scanner: depth pngs, confidence, odometry, rgb.mp4."""
    path = Path(path)
    zf = zipfile.ZipFile(path)
    root = _stray_root(zf)
    text = zf.read(root + "odometry.csv").decode("utf-8").splitlines()
    rows = [line.split(",") for line in text if line and not line.startswith("timestamp")]
    if frame_stride is None:
        frame_stride = max(1, len(rows) // target_frames)
    chosen = rows[::frame_stride]
    # Depth is 256x192. The intrinsics in the csv are for the 1920x1440 RGB stream.
    sample = cv2.imdecode(np.frombuffer(zf.read(root + f"depth/{chosen[0][1].strip()}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
    dh, dw = sample.shape[:2]
    rgb_w, rgb_h = 1920, 1440
    scale_x, scale_y = dw / rgb_w, dh / rgb_h
    fx, fy, cx, cy = (float(chosen[0][i]) for i in (9, 10, 11, 12))
    k = np.array(
        [[fx * scale_x, 0, cx * scale_x], [0, fy * scale_y, cy * scale_y], [0, 0, 1]],
        dtype=float,
    )
    video = _stray_video(zf, root, path)
    frames = []
    for row in chosen:
        stem = row[1].strip()
        depth = cv2.imdecode(np.frombuffer(zf.read(root + f"depth/{stem}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
        conf = cv2.imdecode(np.frombuffer(zf.read(root + f"confidence/{stem}.png"), np.uint8), cv2.IMREAD_UNCHANGED)
        depth_m = depth.astype(np.float32) / 1000.0
        # ARKit confidence: 0 low, 1 medium, 2 high. Medium is too noisy for a wall line.
        depth_m[conf < 2] = 0
        pose = stray_pose_to_opencv_zup(*(float(row[i]) for i in (5, 6, 7, 8, 2, 3, 4)))
        rgb = _video_frame(video, int(stem), (dw, dh))
        frames.append({"depth_m": depth_m, "rgb": rgb, "confidence": conf, "T": pose})
    if video is not None:
        video.release()
    zf.close()
    print(f"loaded {len(frames)} frames, stride {frame_stride}", flush=True)
    return {
        "tier": "lidar",
        "device": "iPhone LiDAR (Stray Scanner)",
        "intrinsics": k,
        "frames": frames,
        "source": str(path),
        "frame_stride": frame_stride,
    }


def _stray_video(zf: zipfile.ZipFile, root: str, archive: Path):
    name = root + "rgb.mp4"
    if name not in zf.namelist():
        return None
    dest = archive.parent / "_cache" / (archive.stem + ".mp4")
    if not dest.is_file() or dest.stat().st_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))
    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        return None
    return cap


def _video_frame(cap, index: int, size: tuple[int, int]):
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

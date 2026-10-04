"""Video tier. Each room is measured from its own letter sheet.

A walk that carries one pose across the building loses scale, because every
sheet is the same rectangle and a track that drops frames never gets the
missed step back. The sheet leaving the frame starts a new room. A later
visit whose walls match a room already measured is the same sheet: its last
look is the doorway into the next room, and the rooms are stitched on that ray.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from propertyscan.photos import _room_from_images, _stitch
from propertyscan.planar import image_is_dark, intrinsics_matrix, pose_from_sheet


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
    acc = _rooms_from_sheets(frames, k)
    acc["dark"] = dark
    acc["intrinsics_source"] = k_source
    return acc


def _rooms_from_sheets(frames: list[np.ndarray], k: np.ndarray) -> dict:
    segments = _sheet_segments(frames, k)
    if not segments:
        raise RuntimeError("Video never locked onto the letter sheet, so it has no scale.")
    local: dict[str, object] = {}
    views: dict[str, list] = {}
    shapes: list[tuple[str, object]] = []
    # The last frame of a visit is the doorway. A repeat of a room already
    # measured keeps that look and spends it on the next new room.
    pending: tuple[str, int] | None = None
    for seg in segments:
        images = [(f"{i:04d}", frames[i]) for i in seg]
        room, used = _room_from_images(images, k)
        if room is None or len(used) < 3:
            continue
        same = _matching_room(room, shapes)
        if same is not None:
            pending = (same, seg[-1])
            continue
        name = f"room-{len(shapes) + 1}"
        local[name] = room
        views[name] = list(used)
        # First frame looks back through the door we entered. Last frame looks
        # at the door we leave by. Both rays have to land or the room is spun
        # onto the wrong wall.
        if pending is not None:
            parent, frame_i = pending
            _link_door(views, parent, frames[frame_i], k, name)
            _link_door(views, name, frames[seg[0]], k, parent)
            pending = None
        elif shapes:
            parent = shapes[-1][0]
            _link_door(views, parent, frames[shapes[-1][2]], k, name)
            _link_door(views, name, frames[seg[0]], k, parent)
        shapes.append((name, room, seg[-1]))
    if not local:
        raise RuntimeError("Video never locked onto the letter sheet, so it has no scale.")
    placed, links = _stitch(local, views, k)
    return {
        "rooms": placed,
        "names": [r.label_name for r in placed],
        "xyz": np.zeros((0, 3)),
        "rgb": np.zeros((0, 3), np.uint8),
        "drift": {
            "method": "sheet_pnp_room_stitch",
            "loop_closures": int(links),
            "used_raw_poses_as_is": False,
            "sheet_fixes": int(sum(len(s) for s in segments)),
            "correction_applied": links > 0,
        },
    }


def _sheet_segments(frames: list[np.ndarray], k: np.ndarray) -> list[list[int]]:
    """Runs of frames that still see a sheet. Two misses in a row are a doorway."""
    posed = [pose_from_sheet(img, k) is not None for img in frames]
    segments: list[list[int]] = []
    current: list[int] = []
    gap = 0
    for i, ok in enumerate(posed):
        if ok:
            if gap >= 2 and current:
                segments.append(current)
                current = []
            gap = 0
            current.append(i)
        else:
            gap += 1
    if current:
        segments.append(current)
    return segments


def _link_door(views: dict, parent: str, img: np.ndarray, k: np.ndarray, child: str) -> None:
    """The doorway look is the last frame of the visit, even if it added no wall points."""
    posed = pose_from_sheet(img, k)
    if posed is None or parent not in views:
        return
    r_wc, t_wc, corners = posed
    views[parent].append(
        {"name": f"door-to-{child}", "R": r_wc, "t": t_wc, "image": img, "corners": corners}
    )


def _matching_room(room, shapes: list[tuple]) -> str | None:
    walls = sorted(room.wall_lengths())
    for name, other, _door in shapes:
        prev = sorted(other.wall_lengths())
        if len(prev) == len(walls) and all(abs(a - b) <= 0.22 for a, b in zip(prev, walls)):
            return name
    return None

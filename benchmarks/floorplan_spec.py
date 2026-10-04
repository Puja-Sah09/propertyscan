"""Ground-truth floor plan for the benchmark generator.

The estimation package does not import this module. Numbers here are the tape.
"""

from __future__ import annotations

# Letter / A4-short side is not used. US Letter, flat on the floor.
LETTER_W = 0.2794
LETTER_H = 0.2159

ROOMS = [
    {"name": "living", "x0": 0.0, "y0": 0.0, "x1": 5.0, "y1": 4.0, "ceiling": 2.440},
    {"name": "hall", "x0": 5.0, "y0": 1.2, "x1": 6.2, "y1": 5.2, "ceiling": 2.440},
    {"name": "kitchen", "x0": 6.2, "y0": 1.2, "x1": 9.7, "y1": 4.7, "ceiling": 2.440},
    {"name": "bedroom", "x0": 5.0, "y0": 5.2, "x1": 8.8, "y1": 9.4, "ceiling": 2.470},
]

# Doors are open gaps. Axis is the wall they sit on: "x" means the wall is x=const.
DOORS = [
    {"a": "living", "b": "hall", "axis": "x", "at": 5.0, "u0": 2.00, "u1": 2.80, "height": 2.03},
    {"a": "hall", "b": "kitchen", "axis": "x", "at": 6.2, "u0": 2.40, "u1": 3.30, "height": 2.03},
    {"a": "hall", "b": "bedroom", "axis": "y", "at": 5.2, "u0": 5.20, "u1": 6.00, "height": 2.03},
]

SOFA = {"x0": 0.80, "y0": 0.60, "x1": 2.20, "y1": 1.80, "z1": 0.75, "room": "living"}

# Staged damage on the living north wall (y = 4). Two classes.
STAIN = {"room": "living", "wall": "north", "x0": 3.30, "x1": 3.90, "z0": 1.70, "z1": 2.15, "klass": "water_stain"}
CRACK = {
    "room": "living",
    "wall": "north",
    "x0": 1.20,
    "x1": 1.23,
    "z0": 0.50,
    "z1": 1.40,
    "klass": "crack",
    "recess_m": 0.018,
}

ADJACENCY = [("living", "hall"), ("hall", "kitchen"), ("hall", "bedroom")]


def room_by_name(name: str) -> dict:
    for r in ROOMS:
        if r["name"] == name:
            return r
    raise KeyError(name)


def sheet_rect(room: dict) -> tuple[float, float, float, float]:
    cx = 0.5 * (room["x0"] + room["x1"])
    cy = 0.5 * (room["y0"] + room["y1"])
    return cx - LETTER_W / 2, cy - LETTER_H / 2, cx + LETTER_W / 2, cy + LETTER_H / 2


def wall_lengths(room: dict) -> list[dict]:
    dx = room["x1"] - room["x0"]
    dy = room["y1"] - room["y0"]
    walls = [
        {"id": "south", "length_m": dx, "angle_deg": 0.0},
        {"id": "east", "length_m": dy, "angle_deg": 90.0},
        {"id": "north", "length_m": dx, "angle_deg": 180.0},
        {"id": "west", "length_m": dy, "angle_deg": -90.0},
    ]
    for w in walls:
        w["openings"] = []
    for d in DOORS:
        if room["name"] not in (d["a"], d["b"]):
            continue
        width = round(d["u1"] - d["u0"], 4)
        if d["axis"] == "x":
            # Vertical wall. East of the west room, west of the east room.
            side = "east" if room["x1"] == d["at"] else "west"
        else:
            side = "north" if room["y1"] == d["at"] else "south"
        for w in walls:
            if w["id"] == side:
                w["openings"].append({"kind": "door", "width_m": width})
    return walls


def footprint_area() -> float:
    return sum((r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) for r in ROOMS)


def ground_truth() -> dict:
    rooms = []
    for r in ROOMS:
        dx = r["x1"] - r["x0"]
        dy = r["y1"] - r["y0"]
        rooms.append(
            {
                "name": r["name"],
                "ceiling_height_m": r["ceiling"],
                "floor_area_m2": round(dx * dy, 4),
                "centroid_m": [0.5 * (r["x0"] + r["x1"]), 0.5 * (r["y0"] + r["y1"])],
                "polygon_m": [
                    [r["x0"], r["y0"]],
                    [r["x1"], r["y0"]],
                    [r["x1"], r["y1"]],
                    [r["x0"], r["y1"]],
                ],
                "walls": wall_lengths(r),
            }
        )
    return {
        "rooms": rooms,
        "adjacency": [list(p) for p in ADJACENCY],
        "footprint_area_m2": round(footprint_area(), 4),
        "damages": [
            {
                "room": "living",
                "klass": "water_stain",
                "area_m2": round((STAIN["x1"] - STAIN["x0"]) * (STAIN["z1"] - STAIN["z0"]), 4),
            },
            {
                "room": "living",
                "klass": "crack",
                "length_m": round(CRACK["z1"] - CRACK["z0"], 4),
            },
        ],
    }

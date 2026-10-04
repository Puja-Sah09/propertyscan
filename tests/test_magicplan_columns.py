"""The Statistics header picks the interior area, not a wall or a volume."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propertyscan.score import _magicplan_columns


def test_interior_area_beats_wall_area():
    header = {
        "name": "Living",
        "area_with_walls": "21.0",
        "area_without_walls": "20.0",
        "volume": "48.0",
    }
    name, area = _magicplan_columns(header)
    assert name == "name"
    assert area == "area_without_walls"


def test_plain_area_column():
    header = {"Room": "Hall", "Area m2": "4.8"}
    name, area = _magicplan_columns(header)
    assert name == "Room"
    assert area == "Area m2"


if __name__ == "__main__":
    test_interior_area_beats_wall_area()
    test_plain_area_column()
    print("magicplan columns ok")

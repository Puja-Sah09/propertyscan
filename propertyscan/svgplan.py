"""One-page floor plan as SVG. Dimensions are the measured values."""

from __future__ import annotations

import html


def render_svg(plan: dict) -> str:
    rooms = plan["rooms"]
    polys = []
    for room in rooms:
        pts = room["polygon_m"]
        if len(pts) >= 3:
            polys.append((room, pts))
    if not polys:
        return "<svg xmlns='http://www.w3.org/2000/svg' width='400' height='200'><text x='20' y='40'>No rooms</text></svg>"
    xs = [p[0] for _, pts in polys for p in pts]
    ys = [p[1] for _, pts in polys for p in pts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1.0)
    scale = 720 / span
    pad = 70

    def xy(p):
        # Y up in the plan, Y down in SVG.
        return pad + (p[0] - minx) * scale, pad + (maxy - p[1]) * scale

    w = pad * 2 + (maxx - minx) * scale
    h = pad * 2 + (maxy - miny) * scale
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{w:.0f}' height='{h:.0f}' viewBox='0 0 {w:.1f} {h:.1f}'>",
        "<rect width='100%' height='100%' fill='#f7f5f2'/>",
        f"<text x='{pad}' y='32' font-family='Georgia, serif' font-size='20' fill='#1c1917'>{html.escape(plan.get('capture_id','plan'))} · {html.escape(plan['tier'])}</text>",
    ]
    for room, pts in polys:
        mapped = [xy(p) for p in pts]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in mapped) + " Z"
        parts.append(f"<path d='{d}' fill='#efeae2' stroke='#292524' stroke-width='3' stroke-linejoin='round'/>")
        cx = sum(p[0] for p in mapped) / len(mapped)
        cy = sum(p[1] for p in mapped) / len(mapped)
        label = room.get("name") or room["id"]
        ceil = room["ceiling_height_m"]["value"]
        area = room["floor_area_m2"]["value"]
        parts.append(
            f"<text x='{cx:.1f}' y='{cy:.1f}' text-anchor='middle' font-family='Calibri, sans-serif' font-size='14' fill='#44403c'>{html.escape(str(label))}</text>"
        )
        parts.append(
            f"<text x='{cx:.1f}' y='{cy+16:.1f}' text-anchor='middle' font-family='Calibri, sans-serif' font-size='11' fill='#78716c'>{area:.2f} m² · h {ceil:.3f} m</text>"
        )
        for i in range(len(pts)):
            a, b = mapped[i], mapped[(i + 1) % len(pts)]
            length = room["walls"][i]["length_m"]["value"] if i < len(room["walls"]) else 0
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            parts.append(
                f"<text x='{mx:.1f}' y='{my:.1f}' text-anchor='middle' font-family='Calibri, sans-serif' font-size='11' fill='#1c1917'>{length:.2f} m</text>"
            )
    drift = plan.get("drift", {})
    parts.append(
        f"<text x='{pad}' y='{h-24:.0f}' font-family='Calibri, sans-serif' font-size='12' fill='#57534e'>drift: {html.escape(str(drift.get('method','unknown')))} · loops {drift.get('loop_closures', 0)}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)

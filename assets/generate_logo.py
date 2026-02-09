"""Generate the searchat wordmark SVG.

Design: 1:1 square cells, right-angle construction, uniform stroke.
Slight curves ONLY where needed for letterform clarity:
  A — diagonal legs (triangular, not rectangular)
  R — curved right side of bowl
  S — curved direction-change corners
  C — curved open ends
  E, H, T — pure rectangles, no curves

Colors:
  "sear" = #3B82F6 (blue)
  "ch"   = linear gradient blue -> orange
  "at"   = #EA580C (orange)
"""

BLUE = "#3B82F6"
ORANGE = "#EA580C"

CELL = 48       # letter height = width (1:1)
S = 9           # stroke thickness
M = 19          # midline y = (CELL - S) // 2
MS = M + S      # 28, bottom of mid bar
B = CELL - S    # 39, top of bottom bar
R_X = CELL - S  # 39, left edge of right-side bars
GAP = 6
PADDING = 10
r = 4           # curve radius (used sparingly)

FILLS = [BLUE, BLUE, BLUE, BLUE, "url(#g)", "url(#g)", ORANGE, ORANGE]


def rects_to_d(rects: list[tuple]) -> str:
    """Convert (x, y, w, h) rectangles to SVG path d."""
    parts = []
    for x, y, w, h in rects:
        parts.append(f"M{x},{y}h{w}v{h}h{-w}Z")
    return " ".join(parts)


# ── S: outline with 4 slight curves at staircase corners ──

def letter_s_path() -> str:
    return (
        f"M0,0 L{CELL},0 L{CELL},{S - r} Q{CELL},{S},{CELL - r},{S}"
        f" L{S},{S} L{S},{M - r} Q{S},{M},{S + r},{M}"
        f" L{CELL},{M} L{CELL},{CELL} L0,{CELL} L0,{B + r} Q0,{B},{r},{B}"
        f" L{R_X},{B} L{R_X},{MS + r} Q{R_X},{MS},{R_X - r},{MS}"
        f" L0,{MS} Z"
    )


# ── E: pure rectangles ──

def letter_e_rects() -> list[tuple]:
    return [
        (0, 0, S, CELL),           # left full
        (0, 0, CELL, S),           # top bar
        (S, M, CELL - S * 2, S),   # mid bar (shorter)
        (0, B, CELL, S),           # bottom bar
    ]


# ── A: triangular with diagonal legs ──

def letter_a_path() -> str:
    # Outer triangle
    outer = f"M0,{CELL} L{CELL // 2},0 L{CELL},{CELL} Z"
    # Upper hole (small triangle above crossbar)
    upper_hole = f"M{M},{M + S + 2} L{CELL // 2},{M + 1} L{MS + 1},{M + S + 2} Z"
    # Lower hole (open bottom below crossbar)
    lower_hole = (
        f"M{15},{B} L{33},{B} L{38},{CELL} L{10},{CELL} Z"
    )
    return f"{outer} {upper_hole} {lower_hole}"


# ── R: rectangles with curved bowl counter ──

def letter_r_path() -> str:
    # Outer contour
    outer = (
        f"M0,0 L{CELL},0 L{CELL},{CELL} L{R_X},{CELL}"
        f" L{R_X},{MS} L{S},{MS} L{S},{CELL} L0,{CELL} Z"
    )
    # Inner hole (bowl counter) with curved right side
    hole = (
        f"M{S},{S} L{R_X - r},{S} Q{R_X},{S},{R_X},{S + r}"
        f" L{R_X},{M - r} Q{R_X},{M},{R_X - r},{M}"
        f" L{S},{M} Z"
    )
    return f"{outer} {hole}"


# ── C: outline with 2 slight curves at open ends ──

def letter_c_path() -> str:
    return (
        f"M0,0 L{CELL},0 L{CELL},{S - r} Q{CELL},{S},{CELL - r},{S}"
        f" L{S},{S} L{S},{B}"
        f" L{CELL - r},{B} Q{CELL},{B},{CELL},{B + r} L{CELL},{CELL}"
        f" L0,{CELL} Z"
    )


# ── H: pure rectangles ──

def letter_h_rects() -> list[tuple]:
    return [
        (0, 0, S, CELL),
        (R_X, 0, S, CELL),
        (S, M, CELL - S * 2, S),
    ]


# ── T: pure rectangles ──

def letter_t_rects() -> list[tuple]:
    cx = (CELL - S) // 2
    return [
        (0, 0, CELL, S),
        (cx, S, S, CELL - S),
    ]


WORD = "searchat"

LETTER_DEFS = {
    "s": ("path", letter_s_path),
    "e": ("rects", letter_e_rects),
    "a": ("path", letter_a_path),
    "r": ("path", letter_r_path),
    "c": ("path", letter_c_path),
    "h": ("rects", letter_h_rects),
    "t": ("rects", letter_t_rects),
}


def generate_svg() -> str:
    x = PADDING
    elements = []

    for i, ch in enumerate(WORD):
        kind, fn = LETTER_DEFS[ch]
        fill = FILLS[i]
        fill_rule = ""

        if kind == "rects":
            rects = [(rx + x, ry + PADDING, rw, rh) for rx, ry, rw, rh in fn()]
            d = rects_to_d(rects)
        else:
            raw_d = fn()
            # Offset all coordinates by (x, PADDING)
            d = offset_path(raw_d, x, PADDING)
            fill_rule = ' fill-rule="evenodd"'

        elements.append(f'  <path d="{d}" fill="{fill}"{fill_rule}/>')
        x += CELL + GAP

    w = x - GAP + PADDING
    h = CELL + PADDING * 2

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="searchat">',
        '  <defs>',
        '    <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="0%">',
        f'      <stop offset="0%" stop-color="{BLUE}"/>',
        f'      <stop offset="100%" stop-color="{ORANGE}"/>',
        '    </linearGradient>',
        '  </defs>',
        *elements,
        '</svg>',
    ]
    return "\n".join(lines)


def offset_path(d: str, dx: float, dy: float) -> str:
    """Offset all absolute coordinates in an SVG path by (dx, dy).

    Handles M, L, Q commands with absolute coordinates.
    h and v (relative) are left unchanged. Z is left unchanged.
    """
    import re
    result = []
    i = 0
    while i < len(d):
        ch = d[i]
        if ch in "MLQ":
            result.append(ch)
            i += 1
            # Read coordinate pairs
            nums_needed = 2 if ch in "ML" else 4  # Q has 2 pairs
            coords = []
            while len(coords) < nums_needed:
                # Skip whitespace and commas
                while i < len(d) and d[i] in " ,":
                    i += 1
                # Read number
                num_start = i
                if i < len(d) and d[i] == '-':
                    i += 1
                while i < len(d) and (d[i].isdigit() or d[i] == '.'):
                    i += 1
                if i > num_start:
                    coords.append(float(d[num_start:i]))
            # Offset pairs
            for j in range(0, len(coords), 2):
                coords[j] += dx
                coords[j + 1] += dy
            pairs = []
            for j in range(0, len(coords), 2):
                x_val = coords[j]
                y_val = coords[j + 1]
                x_str = f"{x_val:.0f}" if x_val == int(x_val) else f"{x_val}"
                y_str = f"{y_val:.0f}" if y_val == int(y_val) else f"{y_val}"
                pairs.append(f"{x_str},{y_str}")
            result.append(",".join(pairs))
        elif ch in "hvZ":
            result.append(ch)
            i += 1
            if ch in "hv":
                # Read one number (relative, don't offset)
                while i < len(d) and d[i] in " ,":
                    i += 1
                num_start = i
                if i < len(d) and d[i] == '-':
                    i += 1
                while i < len(d) and (d[i].isdigit() or d[i] == '.'):
                    i += 1
                result.append(d[num_start:i])
        elif ch in " ,":
            result.append(ch)
            i += 1
        else:
            i += 1
    return "".join(result)


if __name__ == "__main__":
    from pathlib import Path

    svg = generate_svg()
    out = Path(__file__).parent / "logo.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"Written {len(svg)} bytes to {out}")

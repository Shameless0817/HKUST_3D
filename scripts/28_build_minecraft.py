#!/usr/bin/env python3
"""
Build clean Minecraft-style voxel structures from geometric primitives.

Since the photogrammetry mesh lacks ground geometry and has no usable colors,
we construct landmarks as clean geometric shapes with a curated color palette.

Usage:
  python3 scripts/28_build_minecraft.py --landmark track
  python3 scripts/28_build_minecraft.py --landmark plaza
  python3 scripts/28_build_minecraft.py --landmark academic
  python3 scripts/28_build_minecraft.py --landmark atrium
  python3 scripts/28_build_minecraft.py --landmark all
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "output/demo"

# ═══════════════════════════════════════════════════════════════
#  Color Palette (real-world HKUST material colors)
# ═══════════════════════════════════════════════════════════════

COLORS = {
    'track_red':       (196, 75, 60),    # #C44B3C - running track terracotta
    'track_white':     (240, 240, 235),   # lane markings
    'field_green':     (90, 143, 74),     # #5A8F4A - soccer field grass
    'field_alt_green': (75, 130, 60),     # slightly different green for mowing pattern
    'plaza_stone':     (205, 200, 195),   # #CDC8C3 - light grey plaza stone
    'plaza_stone_mid': (190, 185, 180),   # #BEB9B4 - medium grey stone rings
    'plaza_stone_dark': (170, 165, 160),  # #AAA5A0 - dark grey stone accents
    'plaza_accent':    (215, 210, 205),   # lighter stone border
    'building_wall':   (235, 232, 228),   # #EBE8E4 - white concrete (HKUST main color)
    'building_wall_dark': (218, 214, 210), # #DAD6D2 - shadowed white concrete
    'building_wall_rib': (225, 222, 218),  # #E1DEDA - rib texture on white
    'building_glass':  (123, 164, 181),   # #7BA4B5 - blue-tinted glass
    'building_glass_dark': (90, 140, 165), # darker glass
    'building_glass_bright': (145, 185, 200), # brighter glass highlights
    'building_roof':   (140, 130, 120),   # roof gray
    'building_roof_dark': (120, 110, 100), # darker roof
    'ground_gray':     (160, 155, 150),   # surrounding ground
    'ground_brown':    (140, 120, 100),   # dirt/earth
    'water_blue':      (68, 136, 170),     # #4488AA - ocean
    'water_light':     (90, 160, 195),     # lighter water
    'water_dark':      (50, 115, 150),     # deeper water
    'sand_beige':      (210, 195, 170),   # beach/sand
    'sundial_gold':    (229, 192, 123),   # #E5C07B - sundial gold (kept for accents)
    'sundial_dark':    (200, 165, 100),   # dark gold
    'path_gray':       (180, 175, 170),   # walkways
    'path_dark':       (150, 145, 140),   # dark path
    'white':           (245, 243, 240),   # pure white
    'tree_green':      (65, 110, 55),     # #416E37 - tree leaves
    'tree_dark':       (50, 85, 40),      # dark leaves
    'tree_bright':     (85, 135, 70),     # bright leaf highlights
    'trunk_brown':     (120, 90, 60),     # #785A3C - tree trunk
    'trunk_dark':      (95, 70, 45),      # dark trunk
    'hillside_grass':  (100, 140, 80),    # #648C50 - hillside vegetation
    'hillside_dirt':   (150, 135, 115),   # #968773 - exposed earth
    'hillside_rock':   (175, 170, 165),   # #AFAAA5 - rocky outcrop
    'coast_rock':      (145, 140, 135),   # #918C87 - coastal rocks
    'beach_sand':      (225, 215, 195),   # #E1D7C3 - beach sand
    'deep_ocean':      (35, 100, 140),    # #23648C - deep water
    'red_bird':        (210, 60, 45),     # #D23C2D - Red Bird sculpture main
    'red_bird_bright': (235, 70, 50),     # bright red highlights
    'red_bird_dark':   (165, 35, 25),     # dark red shadows
    'concrete_pillar': (195, 190, 185),   # pillar concrete
    'metal_gray':      (130, 128, 125),   # metal fixtures / flagpoles
    'bench_brown':     (150, 115, 80),    # wooden bench
    # Shaw Auditorium specific
    'shaw_white':       (248, 246, 242),   # pure white mineral paint — ring bands
    'champagne_bronze': (205, 175, 135),   # anodized aluminium edge trim
    'glass_facetted':   (135, 178, 195),   # facetted curtain wall glass
    'bamboo_clad':      (185, 155, 115),   # renewable bamboo cladding
    'pv_panel':         (45, 55, 75),      # photovoltaic panels — dark blue-gray
    'skylight_glass':   (160, 200, 215),   # skylight glass — bright blue
    # North Gate specific
    'stone_pillar':   (215, 208, 195),  # natural stone — beige/limestone
    'road_asphalt':   (100, 98, 95),    # asphalt road surface
    'gate_gold':      (218, 180, 110),  # gold lettering on stone pillars
    'canopy_white':   (240, 238, 233),  # bus shelter / covered walkway white
}

PITCH = 0.5  # 0.5m blocks


# ═══════════════════════════════════════════════════════════════
#  Geometric Primitives
# ═══════════════════════════════════════════════════════════════

def snap_grid(wx, wy, wz=None):
    """Snap world coordinates to the voxel grid."""
    gx = round(wx / PITCH) * PITCH
    gy = round(wy / PITCH) * PITCH
    if wz is not None:
        gz = round(wz / PITCH) * PITCH
        return (round(gx, 1), round(gy, 1), round(gz, 1))
    return (round(gx, 1), round(gy, 1))


def ellipse_ring(center, a, b, ring_width, height=1, angle_deg=0, z_offset=0):
    """Generate voxel positions for an elliptical ring (the track)."""
    cx, cy = center
    angle_rad = math.radians(angle_deg)
    circumference = math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))
    n_samples = int(circumference / (PITCH * 0.3))
    positions = set()
    for h in range(height):
        z = z_offset + h * PITCH
        for i in range(n_samples):
            theta = 2 * math.pi * i / n_samples
            ex = a * math.cos(theta); ey = b * math.sin(theta)
            rx = ex * math.cos(angle_rad) - ey * math.sin(angle_rad)
            ry = ex * math.sin(angle_rad) + ey * math.cos(angle_rad)
            nx = b * math.cos(theta); ny = a * math.sin(theta)
            n_len = math.sqrt(nx*nx + ny*ny)
            if n_len > 0: nx /= n_len; ny /= n_len
            rnx = nx * math.cos(angle_rad) - ny * math.sin(angle_rad)
            rny = nx * math.sin(angle_rad) + ny * math.cos(angle_rad)
            for w in np.arange(-ring_width/2, ring_width/2, PITCH * 0.8):
                positions.add(snap_grid(cx + rx + rnx * w, cy + ry + rny * w, z))
    return list(positions)


def filled_ellipse(center, a, b, angle_deg=0, z_offset=0):
    """Generate voxel positions filling an ellipse area at given z."""
    cx, cy = center
    angle_rad = math.radians(angle_deg)
    positions = set()
    step = PITCH * 0.8
    for x in np.arange(cx - a, cx + a + step, step):
        for y in np.arange(cy - b, cy + b + step, step):
            dx, dy = x - cx, y - cy
            rx = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
            ry = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
            if (rx/a)**2 + (ry/b)**2 <= 1.0:
                positions.add(snap_grid(x, y, z_offset))
    return list(positions)


def filled_rect(x1, y1, x2, y2, height=1, z_offset=0):
    """Generate voxel positions for a filled rectangle."""
    positions = set()
    for h in range(height):
        z = z_offset + h * PITCH
        for x in np.arange(min(x1,x2), max(x1,x2), PITCH * 0.8):
            for y in np.arange(min(y1,y2), max(y1,y2), PITCH * 0.8):
                positions.add(snap_grid(x, y, z))
    return list(positions)


def circle_ring(center, radius, ring_width, height=1, z_offset=0):
    """Generate voxel positions for a circular ring."""
    cx, cy = center
    positions = set()
    circumference = 2 * math.pi * radius
    n_samples = int(circumference / (PITCH * 0.3))
    for h in range(height):
        z = z_offset + h * PITCH
        for i in range(n_samples):
            theta = 2 * math.pi * i / n_samples
            nx, ny = math.cos(theta), math.sin(theta)
            for w in np.arange(-ring_width/2, ring_width/2, PITCH * 0.8):
                positions.add(snap_grid(cx + (radius + w) * nx, cy + (radius + w) * ny, z))
    return list(positions)


def filled_circle(center, radius, z=0.0):
    """Generate voxel positions filling a circle at given z."""
    cx, cy = center
    positions = set()
    for x in np.arange(cx - radius, cx + radius, PITCH * 0.8):
        for y in np.arange(cy - radius, cy + radius, PITCH * 0.8):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                positions.add(snap_grid(x, y, z))
    return list(positions)


def filled_arc_sector(center, inner_r, outer_r, start_angle, end_angle, z=0.0):
    """Fill an arc sector (annular wedge) at given z."""
    cx, cy = center
    positions = set()
    step = PITCH * 0.8
    for x in np.arange(cx - outer_r, cx + outer_r + step, step):
        for y in np.arange(cy - outer_r, cy + outer_r + step, step):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < inner_r - PITCH or dist > outer_r + PITCH:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            sa, ea = start_angle % 360, end_angle % 360
            a = angle % 360
            in_arc = (sa <= a <= ea) if sa <= ea else (a >= sa or a <= ea)
            if in_arc:
                positions.add(snap_grid(x, y, z))
    return list(positions)


def wall_arc(center, radius, start_angle, end_angle, height, thickness=1.0, z_offset=0):
    """Generate a curved wall along an arc."""
    cx, cy = center
    positions = set()
    arc_len = abs(end_angle - start_angle) / 360 * 2 * math.pi * radius
    n_samples = int(arc_len / (PITCH * 0.3))
    sa_rad, ea_rad = math.radians(start_angle), math.radians(end_angle)
    for h in range(height):
        z = z_offset + h * PITCH
        for i in range(n_samples):
            theta = sa_rad + (ea_rad - sa_rad) * i / n_samples
            nx, ny = math.cos(theta), math.sin(theta)
            for w in np.arange(-thickness/2, thickness/2, PITCH * 0.8):
                positions.add(snap_grid(cx + (radius + w) * nx, cy + (radius + w) * ny, z))
    return list(positions)


# ═══════════════════════════════════════════════════════════════
#  Detail Elements
# ═══════════════════════════════════════════════════════════════

def build_flame_sculpture(cx, cy, base_z):
    """Build the Red Bird flame/bird abstract sundial sculpture.

    The real sculpture is a bright RED steel abstract form:
    - Flame/bird shape with wings spread, ~8.5m tall
    - Broad at base, tapers to a pointed tip
    - Two wing-like extensions curve outward at mid-height (~4-7m)
    - Slight forward lean
    - Sits on a stepped podium

    Returns list of (position, color) tuples.
    """
    positions = []
    total_h = 8.5  # meters
    # Forward lean: the sculpture leans slightly south (+Y)
    lean_amount = 1.2  # max lean at top

    for h in np.arange(0, total_h, PITCH):
        t = h / total_h  # 0 at base, 1 at tip
        gz = round((base_z + h) / PITCH) * PITCH

        # Center shifts forward with height (forward lean)
        lean = t * lean_amount
        scx = cx
        scy = cy + lean

        # Body cross-section: roughly elliptical, widens then tapers
        if t < 0.12:
            # Base widening (podium transition)
            body_rx = 0.6 + t * 4.0  # widens to ~1.1m
            body_ry = 0.6 + t * 2.5
        elif t < 0.45:
            # Main body, slowly tapering
            body_rx = 1.1 - (t - 0.12) * 0.5
            body_ry = 0.9 - (t - 0.12) * 0.3
        else:
            # Upper body, rapid taper to point
            body_rx = 0.93 * (1.0 - (t - 0.45) / 0.55)
            body_ry = 0.83 * (1.0 - (t - 0.45) / 0.55)

        # Wing extensions (at mid-height, t ~ 0.35-0.75)
        wing_ext = 0.0
        wing_thick = 0.4
        if 0.35 < t < 0.75:
            phase = (t - 0.35) / 0.40  # 0→1→0
            wing_ext = 2.8 * math.sin(phase * math.pi)
            wing_thick = 0.4 + 0.2 * math.sin(phase * math.pi)

        # Twist: body rotates slightly as we go up
        twist_angle = t * math.radians(30)  # 30 degree twist over full height

        # Sample cross-section
        max_r = max(body_rx, wing_ext) + 0.5
        for dx in np.arange(-max_r, max_r + PITCH, PITCH * 0.7):
            for dy in np.arange(-body_ry - 0.5, body_ry + 0.5, PITCH * 0.7):
                # Apply twist rotation
                cos_t = math.cos(-twist_angle)
                sin_t = math.sin(-twist_angle)
                rdx = dx * cos_t - dy * sin_t
                rdy = dx * sin_t + dy * cos_t

                # Check if this point is in the body
                body_dist = (rdx / body_rx)**2 + (rdy / body_ry)**2 if body_rx > 0 and body_ry > 0 else 999
                in_body = body_dist <= 1.0

                # Check if this point is in a wing
                in_wing = False
                if wing_ext > 0.1:
                    # Wings extend in ±X direction
                    wing_tip_dist = abs(abs(rdx) - wing_ext)
                    if wing_tip_dist < 0.5 and abs(rdy) < wing_thick:
                        in_wing = True

                if in_body or in_wing:
                    gx = round((scx + dx) / PITCH) * PITCH
                    gy = round((scy + dy) / PITCH) * PITCH

                    # Color variation for visual interest
                    if in_wing and not in_body:
                        color = COLORS['red_bird_bright']  # wing tips bright
                    elif in_body and body_dist > 0.5:
                        color = COLORS['red_bird_dark']  # edges dark
                    elif in_body and body_dist < 0.15:
                        color = COLORS['red_bird_bright']  # center bright
                    else:
                        color = COLORS['red_bird']  # main body

                    positions.append(((round(gx, 1), round(gy, 1), round(gz, 1)), color))

    # Deduplicate
    seen = {}
    for pos, color in positions:
        if pos not in seen:
            seen[pos] = color
    return list(seen.items())


def build_podium(cx, cy, base_z):
    """Build the 3-step circular podium under the sundial.

    Returns list of (position, color) tuples.
    """
    positions = []
    podium_radii = [3.5, 2.8, 2.1]  # bottom to top step radii
    podium_colors = [COLORS['path_dark'], COLORS['path_gray'], COLORS['plaza_accent']]

    for step, (radius, color) in enumerate(zip(podium_radii, podium_colors)):
        step_z = base_z + step * PITCH
        pts = filled_circle((cx, cy), radius, z=step_z)
        for pos in pts:
            positions.append((pos, color))
        # Raised ring at each step edge
        if step < 2:
            ring = circle_ring((cx, cy), radius, 0.5, height=1, z_offset=step_z + PITCH)
            for pos in ring:
                positions.append((pos, COLORS['path_dark']))

    return positions


def build_reflecting_pool(cx, cy, base_z):
    """Build a circular reflecting pool around the sundial podium.

    Inner radius ~3.8m (around podium), outer radius ~5.5m, shallow (1 voxel deep).
    """
    positions = []
    inner_r = 3.8
    outer_r = 5.5
    pool_z = base_z - PITCH  # recessed into ground

    # Water surface ring
    for x in np.arange(cx - outer_r, cx + outer_r + PITCH, PITCH * 0.8):
        for y in np.arange(cy - outer_r, cy + outer_r + PITCH, PITCH * 0.8):
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if inner_r - 0.3 <= dist <= outer_r + 0.3:
                gx, gy = snap_grid(x, y)[0], snap_grid(x, y)[1]
                wdist = math.sqrt((gx - cx)**2 + (gy - cy)**2)
                if inner_r <= wdist <= outer_r:
                    # Water with varying tones
                    ripple = math.sin(wdist * 3.0) * math.cos(wdist * 2.5)
                    if ripple > 0.3:
                        color = COLORS['water_light']
                    elif ripple < -0.3:
                        color = COLORS['water_dark']
                    else:
                        color = COLORS['water_blue']
                    positions.append(((gx, gy, pool_z), color))

    # Pool border ring (raised edge)
    border = circle_ring((cx, cy), outer_r, 0.5, height=1, z_offset=pool_z + PITCH)
    for pos in border:
        positions.append((pos, COLORS['plaza_accent']))

    return positions


def build_trees(cx, cy, ground_z, positions_list):
    """Add Minecraft-style trees around the plaza.

    Each tree: 2-3m trunk + rounded canopy (3 layers).
    """
    # Tree positions around the plaza perimeter
    tree_angles = [15, 50, 85, 120, 160, 200, 240, 280, 320, 350]
    tree_dist = 26.0  # meters from plaza center (closer to perimeter)

    for angle_deg in tree_angles:
        rad = math.radians(angle_deg)
        tx = cx + tree_dist * math.cos(rad)
        ty = cy + tree_dist * math.sin(rad)

        trunk_h = np.random.default_rng(angle_deg).integers(4, 7)  # 2-3m
        # Trunk
        for h in range(trunk_h):
            for dx in [-0.5, 0.0]:
                for dy in [-0.5, 0.0]:
                    tz = ground_z + h * PITCH
                    color = COLORS['trunk_brown'] if h % 2 == 0 else COLORS['trunk_dark']
                    positions_list.append((snap_grid(tx + dx, ty + dy, tz), color))

        # Canopy layers
        canopy_base = ground_z + trunk_h * PITCH
        layers = [
            (2.0, COLORS['tree_dark']),     # bottom, widest
            (1.5, COLORS['tree_green']),     # middle
            (1.0, COLORS['tree_bright']),    # top, smallest
        ]
        for li, (radius, base_color) in enumerate(layers):
            cz = canopy_base + li * PITCH
            for cx_ in np.arange(tx - radius, tx + radius + PITCH, PITCH * 0.8):
                for cy_ in np.arange(ty - radius, ty + radius + PITCH, PITCH * 0.8):
                    if (cx_ - tx)**2 + (cy_ - ty)**2 <= radius**2:
                        # Mix colors for natural look
                        dist = math.sqrt((cx_ - tx)**2 + (cy_ - ty)**2)
                        if dist < radius * 0.3:
                            color = COLORS['tree_bright']
                        elif dist > radius * 0.7:
                            color = COLORS['tree_dark']
                        else:
                            color = COLORS['tree_green']
                        positions_list.append((snap_grid(cx_, cy_, cz), color))


def build_benches(cx, cy, ground_z):
    """Add benches around the plaza perimeter.

    Simple 2m × 0.5m × 0.5m bench structures.
    """
    positions = []
    bench_angles = [30, 75, 105, 150, 210, 255, 300, 345]
    bench_dist = 21.0  # near outer edge of plaza

    for angle_deg in bench_angles:
        rad = math.radians(angle_deg)
        bx = cx + bench_dist * math.cos(rad)
        by = cy + bench_dist * math.sin(rad)

        # Bench seat (2m long, oriented tangentially to the circle)
        tangent_rad = rad + math.pi / 2
        for length in np.arange(-1.0, 1.1, PITCH):
            lx = bx + length * math.cos(tangent_rad)
            ly = by + length * math.sin(tangent_rad)
            # Seat
            positions.append((snap_grid(lx, ly, ground_z + PITCH), COLORS['bench_brown']))
            # Legs at ends
            if abs(length) > 0.7:
                positions.append((snap_grid(lx, ly, ground_z), COLORS['bench_brown']))
            # Backrest (offset outward)
            back_lx = bx + length * math.cos(tangent_rad) + 0.5 * math.cos(rad)
            back_ly = by + length * math.sin(tangent_rad) + 0.5 * math.sin(rad)
            for bh in [PITCH, PITCH * 2]:
                positions.append((snap_grid(back_lx, back_ly, ground_z + bh), COLORS['bench_brown']))

    return positions


def build_flagpoles(cx, cy, ground_z):
    """Three flagpoles at the south side of the plaza."""
    positions = []
    pole_base_y = cy - 24.0  # south edge
    pole_heights = [10.0, 11.0, 10.0]
    pole_xs = [-3.0, 0.0, 3.0]

    for px, ph in zip(pole_xs, pole_heights):
        pole_height_v = int(ph / PITCH)
        for h in range(pole_height_v):
            pz = ground_z + h * PITCH
            # 0.5m × 0.5m pole
            positions.append((snap_grid(cx + px, pole_base_y, pz), COLORS['metal_gray']))
            positions.append((snap_grid(cx + px + 0.5, pole_base_y, pz), COLORS['metal_gray']))
        # Flag at top (small colored block)
        positions.append((snap_grid(cx + px + 1.0, pole_base_y, ground_z + pole_height_v * PITCH),
                          COLORS['red_bird']))
        positions.append((snap_grid(cx + px + 1.5, pole_base_y, ground_z + pole_height_v * PITCH),
                          COLORS['red_bird_bright']))

    return positions


# ═══════════════════════════════════════════════════════════════
#  Landmark Builders
# ═══════════════════════════════════════════════════════════════

def build_track():
    """Build a standard 400m running track with soccer field, bleachers, field markings, lights."""
    print("\n--- Building Track ---")
    straight_len = 84.39; curve_radius = 36.5; track_width = 9.76
    a = curve_radius + straight_len / 2; b = curve_radius + track_width / 2
    center_x, center_y = 85.0, -155.0
    track_base_z = 2.0
    print(f"  Center: ({center_x}, {center_y}), Track: ~{2*a:.0f}m x ~{2*b:.0f}m")

    all_positions, all_colors = [], []
    SHORELINE_Y = -185.0

    # ═══ 3D TRACK RING (2-voxel thick) ═══
    ring = ellipse_ring((center_x, center_y), a, b, track_width, height=2, z_offset=track_base_z)
    all_positions.extend(ring); all_colors.extend([COLORS['track_red']] * len(ring))
    print(f"  Track ring: {len(ring)} voxels")

    # ═══ INNER FIELD (with mowing stripes) ═══
    inner_a = a - track_width / 2; inner_b = b - track_width / 2
    field = filled_ellipse((center_x, center_y), inner_a, inner_b, z_offset=track_base_z)
    for pos in field:
        stripe = int((pos[1] - center_y) / 5) % 2
        color = COLORS['field_green'] if stripe == 0 else COLORS['field_alt_green']
        all_positions.append(pos); all_colors.append(color)
    print(f"  Inner field: {len(field)} voxels")

    # ═══ LANE MARKINGS ═══
    marking_set = set()
    for lane in range(1, 8):
        inner_r = -track_width/2 + lane * 1.22
        la, lb = a + inner_r, b + inner_r
        circ = math.pi * (3*(la+lb) - math.sqrt((3*la+lb)*(la+3*lb)))
        for i in range(int(circ / (PITCH * 0.5))):
            theta = 2 * math.pi * i / int(circ / (PITCH * 0.5))
            marking_set.add(snap_grid(center_x + la * math.cos(theta), center_y + lb * math.sin(theta), track_base_z + 0.5))
    marking_list = list(marking_set)
    all_positions.extend(marking_list); all_colors.extend([COLORS['track_white']] * len(marking_list))
    print(f"  Lane markings: {len(marking_list)} voxels")

    # ═══ FIELD MARKINGS ═══
    # Center circle (R=9.15m)
    center_circle = circle_ring((center_x, center_y), 9.15, 0.5, z_offset=track_base_z + 0.2)
    all_positions.extend(center_circle); all_colors.extend([COLORS['track_white']] * len(center_circle))
    # Center dot
    cdot = filled_circle((center_x, center_y), 0.5, z=track_base_z + 0.2)
    for pos in cdot:
        all_positions.append(pos); all_colors.append(COLORS['track_white'])

    # Penalty areas (16.5m x 40.3m)
    for y_dir in [-1, 1]:
        py = center_y + y_dir * (inner_b - 16.5)
        for px in np.arange(center_x - 20.15, center_x + 20.15, PITCH * 0.4):
            for side_y in [py, py + y_dir * 16.5]:
                gx, gy = snap_grid(px, side_y)
                all_positions.append((gx, gy, track_base_z + 0.2)); all_colors.append(COLORS['track_white'])
        for pl_y in np.arange(py, py + y_dir * 16.5, PITCH * 0.4):
            for wx_sign in [-1, 1]:
                gx, gy = snap_grid(center_x + wx_sign * 20.15, pl_y)
                all_positions.append((gx, gy, track_base_z + 0.2)); all_colors.append(COLORS['track_white'])
        # Goal area (5.5m x 18.3m)
        gy2 = center_y + y_dir * (inner_b - 5.5)
        for gax in np.arange(center_x - 9.15, center_x + 9.15, PITCH * 0.4):
            gx, gy = snap_grid(gax, gy2)
            all_positions.append((gx, gy, track_base_z + 0.2)); all_colors.append(COLORS['track_white'])
        # Penalty spot
        spot_y = center_y + y_dir * (inner_b - 11.0)
        spot = filled_circle((center_x, spot_y), 0.3, z=track_base_z + 0.2)
        for pos in spot:
            all_positions.append(pos); all_colors.append(COLORS['track_white'])

    # Corner arcs (4 corners of field)
    for cx_sign in [-1, 1]:
        for cy_sign in [-1, 1]:
            ca_x = center_x + cx_sign * inner_a
            ca_y = center_y + cy_sign * inner_b
            for ang in np.arange(0, 90, 5):
                rad = math.radians(ang + (45 if cx_sign > 0 else 135) + (0 if cy_sign > 0 else 180))
                gx, gy = snap_grid(ca_x + 0.5 * math.cos(rad), ca_y + 0.5 * math.sin(rad))
                all_positions.append((gx, gy, track_base_z + 0.2)); all_colors.append(COLORS['track_white'])

    # ═══ SOCCER GOALS (at each end) ═══
    for gy_sign in [-1, 1]:
        goal_y = center_y + gy_sign * (inner_b - 0.5)
        # Posts (7.32m apart, 2.44m tall)
        for gx_sign in [-1, 1]:
            gx = center_x + gx_sign * 3.66
            for gz in np.arange(track_base_z, track_base_z + 2.44, PITCH):
                all_positions.append((gx, goal_y, round(gz, 1))); all_colors.append(COLORS['track_white'])
        # Crossbar
        for gx in np.arange(center_x - 3.66, center_x + 3.66, PITCH * 0.4):
            all_positions.append((round(gx/PITCH)*PITCH, goal_y, track_base_z + 2.44)); all_colors.append(COLORS['track_white'])
        # Net (dark glass behind goal)
        net_y = goal_y - gy_sign * 1.5
        for nx in np.arange(center_x - 4.0, center_x + 4.0, PITCH * 0.4):
            for nz in np.arange(track_base_z, track_base_z + 2.5, PITCH):
                all_positions.append((round(nx/PITCH)*PITCH, net_y, round(nz, 1))); all_colors.append(COLORS['building_glass_dark'])

    # ═══ BLEACHERS (west straight section) ═══
    stand_cx = center_x - 40.0; stand_cy = center_y
    stand_l = 35.0
    for tier in range(6):
        tz = track_base_z + 0.5 + tier * 1.0
        seat_w = 1.5 - tier * 0.15
        sx1 = stand_cx - seat_w; sx2 = stand_cx + seat_w
        sy1, sy2 = stand_cy - stand_l/2, stand_cy + stand_l/2
        for sx in np.arange(sx1, sx2, PITCH * 0.6):
            for sy in np.arange(sy1, sy2, PITCH * 0.6):
                gx, gy = snap_grid(sx, sy)
                all_positions.append((gx, gy, round(tz, 1))); all_colors.append(COLORS['path_gray'])
                all_positions.append((gx, gy, round(tz + 0.5, 1))); all_colors.append(COLORS['path_gray'])
        # Riser (vertical back)
        for ry in np.arange(sy1, sy2, PITCH * 0.6):
            all_positions.append((round(sx2/PITCH)*PITCH, round(ry/PITCH)*PITCH, round(tz + 1.0, 1))); all_colors.append(COLORS['building_wall'])
    # Stand canopy roof
    roof_z = track_base_z + 7.0
    for rx in np.arange(sx1 - 3, sx2 + 3, PITCH * 0.6):
        for ry in np.arange(sy1 - 1, sy2 + 1, PITCH * 0.6):
            all_positions.append((round(rx/PITCH)*PITCH, round(ry/PITCH)*PITCH, roof_z)); all_colors.append(COLORS['building_roof'])

    # ═══ LIGHTING TOWERS (4 corners) ═══
    lt_positions = [
        (center_x + inner_a + 10, center_y + inner_b + 5),
        (center_x + inner_a + 10, center_y - inner_b - 5),
        (center_x - inner_a - 10, center_y + inner_b + 5),
        (center_x - inner_a - 10, center_y - inner_b - 5),
    ]
    for ltx, lty in lt_positions:
        for lz in np.arange(track_base_z, track_base_z + 18.0, PITCH):
            for ldx in [-0.5, 0.0, 0.5]:
                for ldy in [-0.5, 0.0, 0.5]:
                    if abs(ldx) + abs(ldy) <= 1.0:
                        gx, gy = snap_grid(ltx + ldx, lty + ldy)
                        all_positions.append((gx, gy, round(lz, 1))); all_colors.append(COLORS['metal_gray'])
        # Horizontal arm extending toward field
        arm_dir_x = center_x - ltx; arm_dir_y = center_y - lty
        arm_len = math.sqrt(arm_dir_x**2 + arm_dir_y**2)
        arm_dx = arm_dir_x / arm_len; arm_dy = arm_dir_y / arm_len
        for ad in np.arange(0, 6.0, PITCH * 0.4):
            gx, gy = snap_grid(ltx + arm_dx * ad, lty + arm_dy * ad)
            all_positions.append((gx, gy, track_base_z + 18.0)); all_colors.append(COLORS['metal_gray'])
            all_positions.append((gx, gy, track_base_z + 17.5)); all_colors.append(COLORS['metal_gray'])
        # Light clusters at arm tip
        for ld in range(4):
            gx, gy = snap_grid(ltx + arm_dx * (5 + ld * 0.5), lty + arm_dy * (5 + ld * 0.5))
            all_positions.append((gx, gy, track_base_z + 17.5)); all_colors.append(COLORS['sundial_gold'])

    # ═══ LONG JUMP PIT (east side) ═══
    lj_x = center_x + inner_a + 8; lj_y = center_y - 30.0
    # Sand pit
    for px in np.arange(lj_x - 1.5, lj_x + 1.5, PITCH * 0.4):
        for py in np.arange(lj_y, lj_y + 9.0, PITCH * 0.4):
            gx, gy = snap_grid(px, py)
            all_positions.append((gx, gy, track_base_z - 0.3)); all_colors.append(COLORS['sand_beige'])
    # Takeoff board
    for bx in np.arange(lj_x - 1.5, lj_x + 1.5, PITCH * 0.4):
        gx = round(bx / PITCH) * PITCH
        all_positions.append((gx, lj_y, track_base_z + 0.1)); all_colors.append(COLORS['track_white'])
    # Runway approach
    for rx in np.arange(lj_x - 1.0, lj_x + 1.0, PITCH * 0.4):
        for ry in np.arange(lj_y - 40.0, lj_y, PITCH * 0.4):
            gx, gy = snap_grid(rx, ry)
            all_positions.append((gx, gy, track_base_z)); all_colors.append(COLORS['track_red'])

    # ═══ PERIMETER FENCE ═══
    for pa in np.arange(0, 360, 3):
        rad = math.radians(pa)
        fx = center_x + (a + track_width/2 + 3) * math.cos(rad)
        fy = center_y + (b + track_width/2 + 3) * math.sin(rad)
        gx, gy = snap_grid(fx, fy)
        for fz in np.arange(track_base_z, track_base_z + 2.0, PITCH):
            all_positions.append((gx, gy, round(fz, 1))); all_colors.append(COLORS['metal_gray'])

    # ═══ GROUND SURROUND ═══
    margin = 15; outer_a = a + track_width/2 + 1; outer_b = b + track_width/2 + 1
    ground = []
    for x in np.arange(center_x - outer_a - margin, center_x + outer_a + margin, PITCH):
        for y in np.arange(center_y - outer_b - margin, center_y + outer_b + margin, PITCH):
            gx, gy = snap_grid(x, y)
            if gy < SHORELINE_Y - 2: continue
            if ((gx-center_x)/outer_a)**2 + ((gy-center_y)/outer_b)**2 <= 1.0: continue
            ground.append((gx, gy, track_base_z))
    all_positions.extend(ground); all_colors.extend([COLORS['ground_gray']] * len(ground))
    print(f"  Ground: {len(ground)} voxels")
    print(f"  TOTAL: {len(all_positions)} voxels")
    return all_positions, all_colors


def build_plaza():
    """Build the Red Bird Plaza with the RED flame/bird sundial sculpture.

    Key features:
    - Circular brick plaza (~22m radius) with concentric rings
    - Central RED flame/bird sundial sculpture (NOT gold!)
    - Circular reflecting pool around sundial base
    - Stepped podium under the sculpture
    - Trees, benches, flagpoles
    - Ground level at Z ≈ 38m
    """
    print("\n--- Building Red Bird Plaza ---")
    cx, cy = 0.0, -88.0
    ground_z = 38.0
    plaza_radius = 22.0

    # Use dict for dedup
    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    # 1. Plaza base — filled circle with concentric rings in grey stone
    plaza_base = filled_circle((cx, cy), plaza_radius, z=ground_z)
    ring_count = 6
    for pos in plaza_base:
        x, y, z = pos
        dist = math.sqrt((x - cx)**2 + (y - cy)**2)
        ring_idx = int(dist / (plaza_radius / ring_count))
        if ring_idx % 3 == 0: color = COLORS['plaza_stone_dark']
        elif ring_idx % 3 == 1: color = COLORS['plaza_stone']
        else: color = COLORS['plaza_stone_mid']
        add(pos, color)
    print(f"  Plaza base: {len(plaza_base)} voxels")

    # 2. Plaza raised border (2 voxels tall)
    border = circle_ring((cx, cy), plaza_radius, 1.5, height=2, z_offset=ground_z)
    for pos in border: add(pos, COLORS['plaza_stone_dark'])
    print(f"  Plaza border: {len(border)} voxels")

    # 3. Inner decorative ring
    inner_r = plaza_radius * 0.55
    inner_ring = circle_ring((cx, cy), inner_r, 0.5, height=1, z_offset=ground_z)
    for pos in inner_ring: add(pos, COLORS['plaza_stone_mid'])
    print(f"  Inner ring: {len(inner_ring)} voxels")

    # 4. Reflecting pool around sundial (built BEFORE sundial so it's underneath)
    pool_items = build_reflecting_pool(cx, cy, ground_z)
    for pos, color in pool_items: add(pos, color)
    print(f"  Reflecting pool: {len(pool_items)} items")

    # 5. Stepped podium
    podium_items = build_podium(cx, cy, ground_z)
    for pos, color in podium_items: add(pos, color)
    print(f"  Podium: {len(podium_items)} items")

    # 6. RED FLAME/BIRD SUNDIAL — the iconic sculpture
    sundial_items = build_flame_sculpture(cx, cy, ground_z + 1.5)  # starts above podium
    for pos, color in sundial_items: add(pos, color)
    print(f"  Red Bird sundial (RED flame sculpture): {len(sundial_items)} voxels")

    # 7. Radial paths
    for angle in [45, 135, 225, 315]:
        rad = math.radians(angle)
        for dist in np.arange(4.5, plaza_radius - 2, PITCH):
            px = cx + dist * math.cos(rad); py = cy + dist * math.sin(rad)
            for w in [-0.5, 0.0, 0.5]:
                wx = px + w * math.cos(rad + math.pi/2)
                wy = py + w * math.sin(rad + math.pi/2)
                if math.sqrt((wx-cx)**2 + (wy-cy)**2) < plaza_radius - 1:
                    add(snap_grid(wx, wy, ground_z), COLORS['path_gray'])
    print(f"  Radial paths added")

    # 7b. Central compass rose (radial star pattern)
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        rad = math.radians(angle)
        for dist in np.arange(1.5, 5.0, PITCH * 0.6):
            gx, gy = snap_grid(cx + dist * math.cos(rad), cy + dist * math.sin(rad))
            add((gx, gy, ground_z + 0.5), COLORS['plaza_stone_dark'])
    # North arrow (longer, gold)
    for dist in np.arange(1.5, 7.0, PITCH * 0.6):
        gx, gy = snap_grid(cx + dist * math.cos(0), cy + dist * math.sin(0))
        add((gx, gy, ground_z + 0.5), COLORS['sundial_gold'])
    print(f"  Compass rose added")

    # 7c. Lighting bollards around perimeter
    for bl_ang in np.arange(0, 360, 8):
        rad = math.radians(bl_ang)
        bx = cx + (plaza_radius - 1.0) * math.cos(rad)
        by = cy + (plaza_radius - 1.0) * math.sin(rad)
        gx, gy = snap_grid(bx, by)
        add((gx, gy, ground_z + 0.5), COLORS['concrete_pillar'])
        add((gx, gy, ground_z + 1.0), COLORS['sundial_gold'])
    print(f"  Bollard lights added")

    # 7d. Underground vent structures (4 boxes)
    vent_positions = [(-8, -78), (8, -78), (-8, -98), (8, -98)]
    for vx, vy in vent_positions:
        for vdx in np.arange(-1.0, 1.0, PITCH * 0.6):
            for vdy in np.arange(-0.5, 0.5, PITCH * 0.6):
                add(snap_grid(vx + vdx, vy + vdy, ground_z + 0.5), COLORS['metal_gray'])
    print(f"  Vent structures added")

    # 7e. Inward-facing U-shape seating groups (4 positions)
    for seat_ang in [45, 135, 225, 315]:
        rad = math.radians(seat_ang)
        scx = cx + 12.0 * math.cos(rad)
        scy = cy + 12.0 * math.sin(rad)
        # Small tree at center of seating group
        for lr, lc in [(1.0, 'tree_dark'), (0.8, 'tree_green'), (0.5, 'tree_bright')]:
            layer = filled_circle((scx, scy), lr, z=ground_z + 3.0)
            for pos in layer: add(pos, COLORS[lc])
        # Trunk
        for th in range(4):
            add(snap_grid(scx, scy, ground_z + th * PITCH), COLORS['trunk_brown'])
        # 3 benches in U-shape around tree
        for bench_off in [-1, 0, 1]:
            bang = rad + bench_off * 0.35
            brad = math.radians(bang) if isinstance(bang, float) else bang
            bx = scx + 2.5 * math.cos(rad + bench_off * 0.3)
            by = scy + 2.5 * math.sin(rad + bench_off * 0.3)
            for bx2 in np.arange(bx - 1.0, bx + 1.0, PITCH * 0.6):
                add(snap_grid(bx2, by, ground_z + 0.5), COLORS['bench_brown'])
    print(f"  U-shape seating groups added")

    # 8. Trees around the plaza
    tree_items = []
    build_trees(cx, cy, ground_z, tree_items)
    for pos, color in tree_items: add(pos, color)
    print(f"  Trees: {len(tree_items)} voxels")

    # 9. Benches
    bench_items = build_benches(cx, cy, ground_z)
    for pos, color in bench_items: add(pos, color)
    print(f"  Benches: {len(bench_items)} voxels")

    # 10. Flagpoles (south side)
    flag_items = build_flagpoles(cx, cy, ground_z)
    for pos, color in flag_items: add(pos, color)
    print(f"  Flagpoles: {len(flag_items)} voxels")

    # 11. Ground fill around plaza
    ground_margin = 10
    for x in np.arange(cx - plaza_radius - ground_margin, cx + plaza_radius + ground_margin, PITCH):
        for y in np.arange(cy - plaza_radius - ground_margin, cy + plaza_radius + ground_margin, PITCH):
            gx, gy = snap_grid(x, y)
            if math.sqrt((gx-cx)**2 + (gy-cy)**2) <= plaza_radius + 0.5: continue
            add((gx, gy, ground_z), COLORS['ground_gray'])
    print(f"  Surrounding ground added")

    # ═══ SOUTH BALCONY BALUSTRADE (sea-facing edge 135°-225°) ═══
    for ba in np.arange(135, 225, 3):
        rad = math.radians(ba)
        bx = cx + (plaza_radius + 1.0) * math.cos(rad)
        by = cy + (plaza_radius + 1.0) * math.sin(rad)
        gx, gy = snap_grid(bx, by)
        for bz in np.arange(ground_z + 0.5, ground_z + 2.0, PITCH):
            add((gx, gy, round(bz, 1)), COLORS['building_glass'] if bz < ground_z + 1.5 else COLORS['concrete_pillar'])
    print(f"  South balustrade added")

    # ═══ GROUND TEXTURING (grass + flower beds) ═══
    for gx in np.arange(cx - plaza_radius - 5, cx + plaza_radius + 5, PITCH * 3):
        for gy in np.arange(cy + plaza_radius - 5, cy + plaza_radius + 5, PITCH * 3):
            ggx, ggy = snap_grid(gx, gy)
            if math.sqrt((ggx-cx)**2 + (ggy-cy)**2) <= plaza_radius + 1.0: continue
            add((ggx, ggy, ground_z + 0.5), COLORS['hillside_grass'])
    for (fbx, fby) in [(-15, -75), (15, -75)]:
        for fdx in np.arange(-1.5, 1.5, PITCH * 0.6):
            for fdy in np.arange(-0.8, 0.8, PITCH * 0.6):
                add(snap_grid(fbx + fdx, fby + fdy, ground_z + 0.5), COLORS['tree_bright'])
                if abs(fdx) < 0.5 and abs(fdy) < 0.5:
                    add(snap_grid(fbx + fdx, fby + fdy, ground_z + 1.0), COLORS['red_bird_bright'])
    print(f"  Ground texturing added")

    all_positions, all_colors = [], []
    for pos, color in pos_color.items():
        all_positions.append(pos); all_colors.append(color)

    print(f"  TOTAL: {len(all_positions)} voxels")
    return all_positions, all_colors


def build_academic():
    """Build the curved Academic Building arc with enhanced details.

    Enhancements over v1:
    - Ribbed concrete facade texture (vertical ribs every 2m)
    - Thicker window mullions (vertical frames)
    - Entrance openings on the inner arc (facing plaza)
    - Canopies above entrances
    - Roof mechanical equipment
    - End stairwell tower blocks
    """
    print("\n--- Building Academic Building Arc (Enhanced) ---")

    cx, cy = 0.0, 0.0; ground_z = 38.0
    inner_r = 32.0; outer_r = 48.0; wall_thick = 1.0
    arc_start = -30; arc_end = 210
    num_floors = 7; floor_height = 5.0
    building_height = num_floors * floor_height

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    # Entrance definitions: (angle_deg, width_deg)
    entrances = [(15, 7), (70, 7), (130, 7), (170, 7), (190, 7)]
    canopy_styles = ['flat', 'curved', 'flat', 'curved', 'flat']  # varied designs

    # Build each floor
    for floor in range(num_floors):
        fb_z = ground_z + floor * floor_height
        print(f"  Floor {floor+1}/{num_floors}: z={fb_z:.0f}-{fb_z+floor_height:.0f}m")

        # --- Floor slab ---
        slab = filled_arc_sector((cx, cy), inner_r - 1, outer_r + 1, arc_start, arc_end, z=fb_z)
        for pos in slab: add(pos, COLORS['building_wall'] if floor % 2 == 0 else COLORS['building_wall_dark'])
        slab2 = filled_arc_sector((cx, cy), inner_r - 1, outer_r + 1, arc_start, arc_end, z=fb_z + PITCH)
        for pos in slab2: add(pos, COLORS['building_wall_dark'])

        # --- Window band (2m = 4 voxels) ---
        win_z_start = fb_z + PITCH * 2
        for wz in np.arange(win_z_start, win_z_start + 2.0, PITCH):
            for (wall_r, is_outer) in [(outer_r, True), (inner_r, False)]:
                glass_voxels = wall_arc((cx, cy), wall_r, arc_start, arc_end, 1, thickness=wall_thick, z_offset=wz)
                for pos in glass_voxels:
                    gx, gy, gz = pos
                    # Check if this position falls in an entrance (only for inner wall, ground floor)
                    in_entrance = False
                    if not is_outer and floor == 0 and gz < ground_z + 4.0:
                        ang = math.degrees(math.atan2(gy - cy, gx - cx)) % 360
                        for e_ang, e_wid in entrances:
                            if abs(ang - e_ang) < e_wid / 2 or abs(ang - e_ang - 360) < e_wid / 2:
                                in_entrance = True; break
                    if not in_entrance:
                        c = COLORS['building_glass_bright'] if (is_outer and wz < win_z_start + 1.0) else \
                            COLORS['building_glass'] if is_outer else COLORS['building_glass_dark']
                        add(pos, c)

        # --- Vertical window mullions (frames every 2.5m) ---
        mullion_spacing = 5.0  # degrees
        ang = arc_start
        while ang <= arc_end:
            in_entrance = False
            if floor == 0:
                for e_ang, e_wid in entrances:
                    if abs(ang - e_ang) < e_wid / 2: in_entrance = True; break
            if not in_entrance:
                rad = math.radians(ang)
                for wall_r in [inner_r, outer_r]:
                    mx = cx + wall_r * math.cos(rad)
                    my = cy + wall_r * math.sin(rad)
                    for mz in np.arange(win_z_start, win_z_start + 2.0, PITCH):
                        add(snap_grid(mx, my, mz), COLORS['concrete_pillar'])
            ang += mullion_spacing

        # --- Concrete wall band (1m = 2 voxels) ---
        wall_z_start = win_z_start + 2.0
        for wz in np.arange(wall_z_start, wall_z_start + 1.0, PITCH):
            outer_wall = wall_arc((cx, cy), outer_r, arc_start, arc_end, 1, thickness=wall_thick, z_offset=wz)
            for pos in outer_wall: add(pos, COLORS['building_wall_dark'] if floor % 2 == 0 else COLORS['building_wall'])
            inner_wall = wall_arc((cx, cy), inner_r, arc_start, arc_end, 1, thickness=wall_thick, z_offset=wz)
            for pos in inner_wall: add(pos, COLORS['building_wall'])

        # --- Concrete rib texture on outer wall (vertical ribs every 2m) ---
        rib_spacing = 4.0  # degrees (~2m at outer radius)
        ang = arc_start
        while ang <= arc_end:
            rad = math.radians(ang)
            rx = cx + outer_r * math.cos(rad)
            ry = cy + outer_r * math.sin(rad)
            # Rib spans full floor height, just outside the outer wall
            for rz in np.arange(fb_z, fb_z + floor_height, PITCH):
                add(snap_grid(rx, ry, rz), COLORS['building_wall_rib'])
            ang += rib_spacing

        # --- Vertical structural pillars ---
        pillar_spacing = 8  # degrees
        ang = arc_start
        while ang <= arc_end:
            in_entrance = False
            if floor == 0:
                for e_ang, e_wid in entrances:
                    if abs(ang - e_ang) < e_wid / 2 + 2: in_entrance = True; break
            if not in_entrance:
                rad = math.radians(ang)
                for wall_r in [inner_r, outer_r]:
                    ppx = cx + wall_r * math.cos(rad); ppy = cy + wall_r * math.sin(rad)
                    for pz in np.arange(fb_z, fb_z + floor_height, PITCH):
                        add(snap_grid(ppx, ppy, pz), COLORS['concrete_pillar'])
            ang += pillar_spacing

    # --- Roof slab ---
    roof_z = ground_z + building_height
    roof_slab = filled_arc_sector((cx, cy), inner_r - 1, outer_r + 1, arc_start, arc_end, z=roof_z)
    for pos in roof_slab: add(pos, COLORS['building_roof'])

    # Roof parapet
    for parapet_z in np.arange(roof_z + PITCH, roof_z + 1.5, PITCH):
        for wall_r in [inner_r, outer_r]:
            parapet = wall_arc((cx, cy), wall_r, arc_start, arc_end, 1, thickness=0.5, z_offset=parapet_z)
            for pos in parapet: add(pos, COLORS['building_roof_dark'])

    # --- Roof mechanical equipment ---
    equip_positions = [(-10, 30), (40, 25), (90, 10), (140, -15), (190, 10)]
    for eq_angle, eq_offset in equip_positions:
        rad = math.radians(eq_angle)
        eq_r = inner_r + (outer_r - inner_r) * 0.5
        eqx = cx + eq_r * math.cos(rad)
        eqy = cy + eq_r * math.sin(rad)
        eq_size = 2.5  # 2.5m square
        for eh in np.arange(0, 3.0, PITCH):  # 3m tall
            for edx in np.arange(-eq_size/2, eq_size/2, PITCH):
                for edy in np.arange(-eq_size/2, eq_size/2, PITCH):
                    add(snap_grid(eqx + edx, eqy + edy, roof_z + 1.5 + eh), COLORS['metal_gray'])
    print(f"  Roof equipment added")

    # --- End wall stairwell towers ---
    for end_angle in [arc_start, arc_end]:
        rad = math.radians(end_angle)
        mid_r = (inner_r + outer_r) / 2
        tx = cx + mid_r * math.cos(rad)
        ty = cy + mid_r * math.sin(rad)
        tower_half = 2.0  # 4m × 4m tower
        for th in np.arange(ground_z, roof_z + 4.0, PITCH):  # extends above roof
            for tdx in np.arange(-tower_half, tower_half + PITCH, PITCH):
                for tdy in np.arange(-tower_half, tower_half + PITCH, PITCH):
                    add(snap_grid(tx + tdx, ty + tdy, th), COLORS['building_wall_dark'])
    print(f"  End stairwell towers added")

    # --- Entrance canopies (on inner wall at ground floor) ---
    for ei, (e_ang, e_wid) in enumerate(entrances):
        rad = math.radians(e_ang)
        canopy_r = inner_r - 1.5
        canopy_z = ground_z + 4.0
        style = canopy_styles[ei % len(canopy_styles)]
        # Canopy slab
        for cdx in np.arange(-e_wid/2 * 0.5, e_wid/2 * 0.5, PITCH * 0.5):
            dist_along = cdx * (math.pi / 180) * canopy_r
            cx_ = cx + canopy_r * math.cos(rad) + dist_along * math.cos(rad + math.pi/2)
            cy_ = cy + canopy_r * math.sin(rad) + dist_along * math.sin(rad + math.pi/2)
            for cw in np.arange(-1.5, 1.5, PITCH * 0.8):
                cpx = cx_ + cw * math.cos(rad)
                cpy = cy_ + cw * math.sin(rad)
                add(snap_grid(cpx, cpy, canopy_z), COLORS['building_roof_dark'])
                if style == 'curved':
                    add(snap_grid(cpx, cpy, canopy_z + 0.5), COLORS['building_roof_dark'])
        # Gold signage above entrance
        for sx in np.arange(-e_wid/4, e_wid/4, PITCH * 0.5):
            dist_s = sx * (math.pi / 180) * (inner_r - 0.5)
            sx_ = cx + (inner_r - 0.5) * math.cos(rad) + dist_s * math.cos(rad + math.pi/2)
            sy_ = cy + (inner_r - 0.5) * math.sin(rad) + dist_s * math.sin(rad + math.pi/2)
            add(snap_grid(sx_, sy_, canopy_z + 0.5), COLORS['sundial_gold'])
    print(f"  Entrance canopies added")

    # ═══ EAST/WEST WING TERMINATIONS ═══
    # West wing (-30°): projecting glass-faced stair tower
    w_end_ang = arc_start
    w_rad = math.radians(w_end_ang)
    w_mid_r = (inner_r + outer_r) / 2
    wt_x = cx + w_mid_r * math.cos(w_rad)
    wt_y = cy + w_mid_r * math.sin(w_rad)
    wt_size = 4.0
    for th in np.arange(ground_z, roof_z + 6.0, PITCH):
        for tdx in np.arange(-wt_size/2, wt_size/2, PITCH * 0.6):
            for tdy in np.arange(-wt_size/2, wt_size/2, PITCH * 0.6):
                gx, gy = snap_grid(wt_x + tdx, wt_y + tdy)
                # Glass walls for stair visibility
                if abs(tdx) > wt_size/2 - 0.5 or abs(tdy) > wt_size/2 - 0.5:
                    add((gx, gy, round(th, 1)), COLORS['building_glass_bright'])
                else:
                    add((gx, gy, round(th, 1)), COLORS['concrete_pillar'])

    # East wing (210°): entrance pavilion with gold roof accent
    e_end_ang = arc_end
    e_rad = math.radians(e_end_ang)
    e_mid_r = (inner_r + outer_r) / 2
    ep_cx = cx + e_mid_r * math.cos(e_rad)
    ep_cy = cy + e_mid_r * math.sin(e_rad)
    ep_w, ep_d, ep_h = 8.0, 6.0, 8.0
    ep_x1, ep_x2 = ep_cx - ep_w/2, ep_cx + ep_w/2
    ep_y1, ep_y2 = ep_cy - ep_d/2, ep_cy + ep_d/2
    for eh in np.arange(ground_z, ground_z + ep_h, PITCH):
        for ex in np.arange(ep_x1, ep_x2 + PITCH, PITCH * 0.6):
            add(snap_grid(ex, ep_y1, eh), COLORS['building_wall'])
            add(snap_grid(ex, ep_y2, eh), COLORS['building_wall_dark'])
        for ey in np.arange(ep_y1, ep_y2 + PITCH, PITCH * 0.6):
            add(snap_grid(ep_x1, ey, eh), COLORS['building_glass'])
            add(snap_grid(ep_x2, ey, eh), COLORS['building_glass_bright'])
    ep_roof = filled_rect(ep_x1, ep_y1, ep_x2, ep_y2, height=2, z_offset=ground_z + ep_h)
    for pos in ep_roof: add(pos, COLORS['building_roof'])
    # Gold accent on roof
    for erx in np.arange(ep_x1, ep_x2 + PITCH, PITCH * 0.6):
        add(snap_grid(erx, ep_cy, ground_z + ep_h + 2.0), COLORS['sundial_gold'])
    print(f"  Wing terminations added")

    # ═══ ROOF PHOTOVOLTAIC PANELS ═══
    pv_positions = [(0, 25), (60, 10), (100, 0), (150, -10), (195, 10)]
    for pv_ang, pv_off in pv_positions:
        pv_rad = math.radians(pv_ang)
        pv_r = inner_r + (outer_r - inner_r) * 0.6
        pvx = cx + pv_r * math.cos(pv_rad)
        pvy = cy + pv_r * math.sin(pv_rad)
        pv_size = 4.0
        for pdx in np.arange(-pv_size/2, pv_size/2, PITCH * 0.6):
            for pdy in np.arange(-pv_size/2, pv_size/2, PITCH * 0.6):
                add(snap_grid(pvx + pdx, pvy + pdy, roof_z + 2.0), COLORS['building_glass_bright'])
        # Metal frame edge
        for pe in np.arange(-pv_size/2, pv_size/2, PITCH * 0.6):
            add(snap_grid(pvx + pe, pvy - pv_size/2, roof_z + 2.5), COLORS['metal_gray'])
            add(snap_grid(pvx + pe, pvy + pv_size/2, roof_z + 2.5), COLORS['metal_gray'])
    print(f"  Photovoltaic panels added")

    # --- Academic Concourse floor (inner arc platform at podium level) ---
    # Real HKUST: the Academic Concourse is the main east-west indoor
    # pedestrian spine running through the arc's interior at G/F level.
    for x in np.arange(-inner_r, inner_r + PITCH, PITCH * 0.8):
        for y in np.arange(-inner_r, inner_r + PITCH, PITCH * 0.8):
            gx, gy = snap_grid(x, y)
            dist = math.sqrt(gx**2 + gy**2)
            if dist >= inner_r - 1.0: continue
            angle = math.degrees(math.atan2(gy, gx))
            if angle < 0: angle += 360
            # Arc gap: 210 to 330 degrees. Skip the wide-open south-facing area.
            in_gap = (210 <= angle <= 330)
            if in_gap and dist > inner_r * 0.25: continue
            add((gx, gy, round(ground_z, 1)), COLORS['path_gray'])
            add((gx, gy, round(ground_z + PITCH, 1)), COLORS['path_gray'])
    print(f"  Concourse platform added")

    # ═══ CONCOURSE SHOPS/CAFES (6-8 units along inner arc) ═══
    shop_positions = [(30, 0), (55, 1), (85, 2), (110, 3), (145, 4), (175, 5)]
    for shop_ang, shop_idx in shop_positions:
        shop_rad = math.radians(shop_ang)
        sx = cx + (inner_r - 1.0) * math.cos(shop_rad)
        sy = cy + (inner_r - 1.0) * math.sin(shop_rad)
        # Shopfront (glass + sign band)
        for sz in np.arange(ground_z + 0.5, ground_z + 3.0, PITCH):
            # Glass shopfront
            for sw in np.arange(-1.5, 1.5, PITCH * 0.6):
                perp_x = -math.sin(shop_rad); perp_y = math.cos(shop_rad)
                add(snap_grid(sx + perp_x * sw, sy + perp_y * sw, sz), COLORS['building_glass_bright'])
        # Sign band
        for sw in np.arange(-1.8, 1.8, PITCH * 0.6):
            perp_x = -math.sin(shop_rad); perp_y = math.cos(shop_rad)
            add(snap_grid(sx + perp_x * sw, sy + perp_y * sw, ground_z + 3.0), COLORS['building_roof_dark'])
        # Gold signage above shop
        add(snap_grid(sx, sy, ground_z + 3.5), COLORS['sundial_gold'])

    # Outdoor cafe seating (in front of 2 shops)
    for cafe_ang in [55, 110]:
        cafe_rad = math.radians(cafe_ang)
        cx_pos = cx + (inner_r - 3.0) * math.cos(cafe_rad)
        cy_pos = cy + (inner_r - 3.0) * math.sin(cafe_rad)
        # Tables
        for td in [(0, 0), (2, 0), (0, 2)]:
            gx, gy = snap_grid(cx_pos + td[0], cy_pos + td[1])
            add((gx, gy, ground_z + 0.5), COLORS['bench_brown'])
    print(f"  Concourse shops + cafes added")

    # ═══ LECTURE THEATRE PODS (3 semi-cylindrical, projecting inward) ═══
    pod_angles = [50, 110, 170]  # degrees along inner arc
    for pod_ang in pod_angles:
        pod_rad = math.radians(pod_ang)
        pod_r = 5.0; pod_w = 8.0
        pod_cx = cx + (inner_r - pod_r) * math.cos(pod_rad)
        pod_cy = cy + (inner_r - pod_r) * math.sin(pod_rad)
        # Semi-cylindrical wall (180 degrees, facing inward)
        for pa in np.arange(-90, 90, 3):
            prad = math.radians(pa)
            px = pod_cx + pod_r * math.cos(prad)
            py = pod_cy + pod_r * math.sin(prad)
            for dw in np.arange(-pod_w/2, pod_w/2, PITCH * 0.6):
                gx, gy = snap_grid(px, py + dw * math.cos(prad))
                for pz in np.arange(ground_z, ground_z + 5.0, PITCH):
                    add((gx, gy, round(pz, 1)), COLORS['building_wall_dark'])
            # Clerestory window at top
            for pz in np.arange(ground_z + 3.5, ground_z + 5.0, PITCH):
                gx, gy = snap_grid(px, py)
                add((gx, gy, round(pz, 1)), COLORS['building_glass_bright'])
        # Pod roof
        for prx in np.arange(pod_cx - pod_r, pod_cx + pod_r, PITCH * 0.6):
            for pry in np.arange(pod_cy - pod_w/2, pod_cy + pod_w/2, PITCH * 0.6):
                gx, gy = snap_grid(prx, pry)
                if math.sqrt((gx-pod_cx)**2 + (gy-pod_cy)**2) < pod_r + 1.0:
                    add((gx, gy, ground_z + 5.0), COLORS['building_roof'])
    print(f"  Lecture theatre pods added")

    # ═══ CONCOURSE PAVING GRID ═══
    for grid_x in np.arange(-inner_r + 2, inner_r - 1, 5.0):
        for grid_y in np.arange(-inner_r + 2, inner_r - 1, 5.0):
            gx, gy = snap_grid(grid_x, grid_y)
            dist = math.sqrt(gx**2 + gy**2)
            if dist >= inner_r - 1.0: continue
            angle = math.degrees(math.atan2(gy, gx))
            if angle < 0: angle += 360
            in_gap = (210 <= angle <= 330)
            if in_gap and dist > inner_r * 0.25: continue
            add((gx, gy, ground_z + 0.5), COLORS['plaza_stone_dark'])
    print(f"  Concourse paving grid added")

    # --- End walls ---
    for end_angle in [arc_start, arc_end]:
        rad = math.radians(end_angle)
        for r in np.arange(inner_r, outer_r + PITCH, PITCH * 0.8):
            ex = cx + r * math.cos(rad); ey = cy + r * math.sin(rad)
            for hz in np.arange(ground_z, ground_z + building_height + 1.5, PITCH):
                add(snap_grid(ex, ey, hz), COLORS['building_wall_dark'])

    all_positions, all_colors = [], []
    for pos, color in pos_color.items():
        all_positions.append(pos); all_colors.append(color)

    print(f"  TOTAL: {len(all_positions)} voxels")
    return all_positions, all_colors


def build_atrium():
    """Build the Atrium (賽馬會大堂) — polished v2 with realistic features.

    Architectural features:
    - Glass skylight opening over central void (signature HKUST feature)
    - Open entrance portals on north (to Academic Arc) and south (to Plaza)
    - Widened portico (16m) with grand entry steps
    - Cross-crossing escalators within the central void
    - Balustrades/railings around void edges on each platform level
    - North connecting bridge to Academic Concourse (glass-roofed)
    - Cantilevered observation deck ("mushroom" viewing platform)
    - Interior furnishings: info desk, seating, LED screen
    - Glass-roofed south walkway to Red Bird Plaza
    - Arched roof with central skylight, glass curtain walls, stepped terraces
    """
    print("\n--- Building Atrium — 賽馬會大堂 (Polished v2) ---")

    atrium_cx = 0.0; atrium_cy = -38.0; ground_z = 38.0
    atrium_width = 22.0; atrium_depth = 18.0; atrium_height = 20.0

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    hx1 = atrium_cx - atrium_width / 2; hx2 = atrium_cx + atrium_width / 2
    hy1 = atrium_cy - atrium_depth / 2; hy2 = atrium_cy + atrium_depth / 2

    print(f"  Footprint: {atrium_width:.0f}m x {atrium_depth:.0f}m, Height: {atrium_height:.0f}m")

    # --- Central atrium void definition ---
    void_cx = atrium_cx; void_cy = atrium_cy
    void_hx = 4.0; void_hy = 6.0  # 8m x 12m open void
    void_x1 = void_cx - void_hx; void_x2 = void_cx + void_hx
    void_y1 = void_cy - void_hy; void_y2 = void_cy + void_hy

    # --- Entrance portal definitions ---
    # South entrance: X[-4, 4] at Y=hy1(-47), Z[39, 49] — 8m wide x 10m high
    door_sx1 = -4.0; door_sx2 = 4.0
    door_z_low = ground_z + 1.0; door_z_high = ground_z + 11.0  # 10m tall portal
    # North entrance: same width, at Y=hy2(-29)
    door_nx1 = -4.0; door_nx2 = 4.0

    def is_in_south_door(x, z):
        """Check if position is within the south entrance opening."""
        if not (door_sx1 - 0.5 < x < door_sx2 + 0.5): return False
        if not (door_z_low - 0.5 < z < door_z_high + 0.5): return False
        return True

    def is_in_north_door(x, z):
        """Check if position is within the north entrance opening."""
        if not (door_nx1 - 0.5 < x < door_nx2 + 0.5): return False
        if not (door_z_low - 0.5 < z < door_z_high + 0.5): return False
        return True

    # --- Skylight opening definition ---
    # Skylight: void area inset by 1m, X[-3, 3], Y[-43, -33]
    sky_x1 = void_x1 + 1.0; sky_x2 = void_x2 - 1.0
    sky_y1 = void_y1 + 1.0; sky_y2 = void_y2 - 1.0

    def is_in_skylight(x, y):
        """Check if a roof position falls within the skylight opening."""
        return sky_x1 - 0.5 < x < sky_x2 + 0.5 and sky_y1 - 0.5 < y < sky_y2 + 0.5

    # ================================================================
    #  1. FLOOR SLAB
    # ================================================================
    floor = filled_rect(hx1, hy1, hx2, hy2, height=3, z_offset=ground_z)
    for pos in floor: add(pos, COLORS['path_gray'])
    # Void floor — polished stone with concentric pattern
    void_floor = filled_rect(void_x1, void_y1, void_x2, void_y2, height=3, z_offset=ground_z)
    for pos in void_floor: add(pos, COLORS['plaza_accent'])
    # Inner void accent ring
    for x in np.arange(void_x1 + 1.0, void_x2 - 1.0 + PITCH, PITCH * 0.8):
        for y in np.arange(void_y1 + 1.0, void_y2 - 1.0 + PITCH, PITCH * 0.8):
            gx, gy = snap_grid(x, y)
            if void_x1 + 1.5 <= gx <= void_x2 - 1.5 and void_y1 + 1.5 <= gy <= void_y2 - 1.5:
                add((gx, gy, round(ground_z + PITCH, 1)), COLORS['path_gray'])
    print(f"  Floor slab added")

    # ================================================================
    #  2. LONG WALLS — Glass curtain walls with entrance openings
    # ================================================================
    wall_thick = 1.0
    for y_wall in [hy1, hy2]:
        for hz in np.arange(ground_z + 1.0, ground_z + atrium_height, PITCH):
            for x in np.arange(hx1, hx2 + PITCH, PITCH * 0.8):
                for wx in np.arange(-wall_thick/2, wall_thick/2 + PITCH, PITCH * 0.8):
                    gx = round((x + wx) / PITCH) * PITCH
                    # Skip door openings
                    if y_wall == hy1 and is_in_south_door(gx, hz): continue
                    if y_wall == hy2 and is_in_north_door(gx, hz): continue

                    rel_z = hz - ground_z
                    snap_x = round(x / PITCH) * PITCH
                    horiz_frame = (rel_z % 4.0 < PITCH * 1.5)
                    vert_frame = ((snap_x - hx1) % 3.0 < PITCH * 1.5)
                    if horiz_frame or vert_frame:
                        color = COLORS['concrete_pillar']
                    else:
                        color = COLORS['building_glass_bright'] if rel_z % 3.0 < 1.0 else COLORS['building_glass']
                    add((gx, y_wall, round(hz, 1)), color)

    # Door frame columns (reinforced pillars flanking each entrance)
    for y_wall, door_fn in [(hy1, is_in_south_door), (hy2, is_in_north_door)]:
        for door_edge_x in [door_sx1 - 0.5, door_sx2 + 0.5]:
            for hz in np.arange(door_z_low, door_z_high + PITCH, PITCH):
                for wx in np.arange(-wall_thick/2, wall_thick/2 + PITCH, PITCH * 0.8):
                    gx = round((door_edge_x + wx) / PITCH) * PITCH
                    if door_fn(gx, hz):
                        add((gx, y_wall, round(hz, 1)), COLORS['concrete_pillar'])
    # Door header beam (horizontal lintel above each entrance)
    for y_wall in [hy1, hy2]:
        for x in np.arange(door_sx1 - 1.0, door_sx2 + 1.5, PITCH * 0.8):
            for wx in np.arange(-wall_thick/2, wall_thick/2 + PITCH, PITCH * 0.8):
                gx = round((x + wx) / PITCH) * PITCH
                for dz in [0, PITCH]:
                    add((gx, y_wall, round(door_z_high + dz, 1)), COLORS['concrete_pillar'])
    print(f"  Glass curtain walls with entrance portals added")

    # ================================================================
    #  3. SHORT WALLS — Concrete end walls
    # ================================================================
    for x_wall in [hx1, hx2]:
        for hz in np.arange(ground_z + 1.0, ground_z + atrium_height, PITCH):
            for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
                for wy in np.arange(-wall_thick/2, wall_thick/2 + PITCH, PITCH * 0.8):
                    add(snap_grid(x_wall, y + wy, hz), COLORS['building_wall'])
    print(f"  Concrete end walls added")

    # ================================================================
    #  4. ARCHED ROOF with GLASS SKYLIGHT opening
    # ================================================================
    roof_peak_z = ground_z + atrium_height + 3.0  # peak at center Z=61.0
    roof_edge_z = ground_z + atrium_height  # Z=58.0

    for x in np.arange(hx1 - 1, hx2 + 1 + PITCH, PITCH * 0.8):
        for y in np.arange(hy1 - 1, hy2 + 1 + PITCH, PITCH * 0.8):
            gx, gy = snap_grid(x, y)
            # Skip skylight opening
            if is_in_skylight(gx, gy): continue

            x_frac = abs(gx - atrium_cx) / (atrium_width / 2 + 1)
            y_frac = abs(gy - atrium_cy) / (atrium_depth / 2 + 1)
            max_frac = max(x_frac, y_frac)
            roof_h = roof_peak_z - max_frac * 3.0
            gz = round(roof_h / PITCH) * PITCH
            add((gx, gy, round(gz, 1)), COLORS['building_roof'])
            add((gx, gy, round(gz - PITCH, 1)), COLORS['building_roof_dark'])

    # Skylight frame — raised curb around the opening
    curb_z = roof_edge_z  # Z=58.0
    for x in np.arange(sky_x1 - 0.5, sky_x2 + 1.0, PITCH * 0.6):
        for y in np.arange(sky_y1 - 0.5, sky_y2 + 1.0, PITCH * 0.6):
            gx, gy = snap_grid(x, y)
            # Only the perimeter (not inside the opening)
            on_edge = (abs(gx - sky_x1) < 1.0 or abs(gx - sky_x2) < 1.0 or
                       abs(gy - sky_y1) < 1.0 or abs(gy - sky_y2) < 1.0)
            in_border = (sky_x1 - 1.0 <= gx <= sky_x2 + 1.0 and
                         sky_y1 - 1.0 <= gy <= sky_y2 + 1.0)
            if on_edge and in_border:
                # Raised curb: 3 voxels high
                for dz in [0, PITCH, PITCH * 2]:
                    add((gx, gy, round(curb_z + dz, 1)), COLORS['concrete_pillar'])

    # Skylight glass — transparent panels spanning the opening
    glass_z = curb_z + PITCH * 3  # Z=59.5 — sits above the curb
    glass_beam_spacing = 2.0
    for x in np.arange(sky_x1, sky_x2 + PITCH, PITCH * 0.6):
        for y in np.arange(sky_y1, sky_y2 + PITCH, PITCH * 0.6):
            gx, gy = snap_grid(x, y)
            if is_in_skylight(gx, gy):
                # Glass beams running along X (every 2m)
                near_beam = (abs(gy - round(gy / glass_beam_spacing) * glass_beam_spacing) < PITCH * 1.5)
                if near_beam:
                    # Steel frame member
                    add((gx, gy, round(glass_z, 1)), COLORS['concrete_pillar'])
                    add((gx, gy, round(glass_z - PITCH, 1)), COLORS['concrete_pillar'])
                else:
                    # Glass panel
                    add((gx, gy, round(glass_z, 1)), COLORS['building_glass_bright'])
                    add((gx, gy, round(glass_z - PITCH, 1)), COLORS['building_glass'])

    # X-brace truss across the skylight (diagonal structural members)
    for t in np.arange(0, 1.0, 0.05):
        # Diagonal 1: (sky_x1, sky_y1) → (sky_x2, sky_y2)
        bx1 = sky_x1 + t * (sky_x2 - sky_x1)
        by1 = sky_y1 + t * (sky_y2 - sky_y1)
        # Diagonal 2: (sky_x2, sky_y1) → (sky_x1, sky_y2)
        bx2 = sky_x2 - t * (sky_x2 - sky_x1)
        by2 = sky_y1 + t * (sky_y2 - sky_y1)
        for bx, by in [(bx1, by1), (bx2, by2)]:
            gx, gy = snap_grid(bx, by)
            if is_in_skylight(gx, gy):
                add((gx, gy, round(glass_z + PITCH, 1)), COLORS['building_wall_dark'])
    print(f"  Arched roof with glass skylight added")

    # ================================================================
    #  5. SOUTH ENTRANCE PORTICO — widened to 16m
    # ================================================================
    portico_y = hy1  # south face Y=-47
    portico_w = 16.0; portico_d = 5.0
    px1 = atrium_cx - portico_w / 2; px2 = atrium_cx + portico_w / 2
    py1 = portico_y - portico_d; py2 = portico_y  # Y=-52 to Y=-47

    # Portico floor (raised 1 step above ground)
    portico_floor = filled_rect(px1, py1, px2, py2, height=2, z_offset=ground_z + PITCH)
    for pos in portico_floor: add(pos, COLORS['path_gray'])

    # Portico columns — 6 columns (3 pairs)
    col_x_positions = [px1 + 1.5, atrium_cx, px2 - 1.5]
    for col_x in col_x_positions:
        for col_y in [py1 + 0.5, py1 + portico_d - 0.5]:
            for ch in np.arange(ground_z + PITCH, ground_z + 8.0, PITCH):
                for cd in [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]:
                    add(snap_grid(col_x + cd[0], col_y + cd[1], ch), COLORS['concrete_pillar'])

    # Portico canopy (elevated flat slab at Z=46.0)
    canopy_z = ground_z + 8.0
    canopy = filled_rect(px1, py1, px2, py2, height=2, z_offset=canopy_z)
    for pos in canopy: add(pos, COLORS['building_roof'])
    # Canopy edge trim
    for cx_ in np.arange(px1, px2 + PITCH, PITCH * 0.6):
        add(snap_grid(cx_, py2, canopy_z), COLORS['building_roof_dark'])
        add(snap_grid(cx_, py1, canopy_z), COLORS['building_roof_dark'])
    for cy_ in np.arange(py1, py2 + PITCH, PITCH * 0.6):
        for edge_x in [px1, px2]:
            gx2, gy2 = snap_grid(edge_x, cy_)
            add((gx2, gy2, round(canopy_z, 1)), COLORS['building_roof_dark'])

    # Grand entry steps (3 wide steps in front of portico)
    for step_i in range(3):
        step_y1 = py1 - (step_i + 1) * 1.5
        step_y2 = py1 - step_i * 1.5
        step_z = ground_z - step_i * PITCH * 2  # each step slightly lower
        step_w = portico_w + 2.0  # wider than portico
        sx1 = atrium_cx - step_w / 2; sx2 = atrium_cx + step_w / 2
        step_floor = filled_rect(sx1, step_y1, sx2, step_y2, height=2, z_offset=step_z)
        for pos in step_floor: add(pos, COLORS['path_gray'] if step_i % 2 == 0 else COLORS['path_dark'])
    print(f"  Widened south portico (16m) with grand steps added")

    # ================================================================
    #  6. INTERIOR COLUMNS — framing the atrium void
    # ================================================================
    col_positions = [
        (void_x1, void_y1), (void_x1, void_y2),
        (void_x2, void_y1), (void_x2, void_y2),
        (void_x1, void_cy), (void_x2, void_cy),
    ]
    for col_x, col_y in col_positions:
        for ch in np.arange(ground_z + 2.0, ground_z + atrium_height, PITCH):
            for cd in [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]:
                add(snap_grid(col_x + cd[0], col_y + cd[1], ch), COLORS['concrete_pillar'])
    print(f"  Interior columns added")

    # ================================================================
    #  7. STEPPED TERRACES — amphitheater around void
    # ================================================================
    terrace_z_levels = []  # track terrace levels for balustrades
    for step in range(4):
        step_z = ground_z + 1.0 + step * 4.0
        step_margin = step * 2.5
        for terrace_y, y_dir in [(void_y2, 1), (void_y1, -1)]:
            ty1 = terrace_y + step_margin * y_dir
            ty2 = min(hy2, max(hy1, terrace_y + (step_margin + 6.0) * y_dir))
            if ty1 == ty2: continue
            sx1 = max(hx1 + step_margin, void_x1 - step_margin * 0.5)
            sx2 = min(hx2 - step_margin, void_x2 + step_margin * 0.5)
            if sx2 <= sx1: continue
            terrace = filled_rect(sx1, min(ty1, ty2), sx2, max(ty1, ty2), height=2, z_offset=step_z)
            for pos in terrace: add(pos, COLORS['path_gray'] if step % 2 == 0 else COLORS['path_dark'])
        terrace_z_levels.append(step_z)
    print(f"  Stepped terraces added")

    # ================================================================
    #  8. BALUSTRADES — low walls along void edges on each terrace
    # ================================================================
    rail_h = 1.0  # 1m high railings
    for step in range(4):
        step_z = terrace_z_levels[step] + PITCH * 2  # top of terrace floor
        step_margin = step * 2.5

        # Railings along the 4 void edges, inset by the step margin
        ra_x1 = void_x1 - step_margin * 0.5
        ra_x2 = void_x2 + step_margin * 0.5
        ra_y1 = void_y1 - step_margin * 0.5
        ra_y2 = void_y2 + step_margin * 0.5

        # Only add railings where terrace exists (rough check)
        if step_z <= ground_z + atrium_height - 2.0:
            for edge_name, ex1, ex2, ey1, ey2 in [
                ('N', ra_x1, ra_x2, ra_y2, ra_y2),
                ('S', ra_x1, ra_x2, ra_y1, ra_y1),
                ('E', ra_x2, ra_x2, ra_y1, ra_y2),
                ('W', ra_x1, ra_x1, ra_y1, ra_y2),
            ]:
                for x in np.arange(ex1, ex2 + PITCH, PITCH * 0.6):
                    for y in np.arange(ey1, ey2 + PITCH, PITCH * 0.6):
                        gx, gy = snap_grid(x, y)
                        if not (ra_x1 - 0.5 <= gx <= ra_x2 + 0.5 and ra_y1 - 0.5 <= gy <= ra_y2 + 0.5):
                            continue
                        # Every 2m gap for visual openness
                        gap_pos = round((gx + gy) / 2.0 / PITCH) * PITCH
                        if abs(gap_pos % 2.0) < PITCH * 0.5: continue
                        for rh in np.arange(0, rail_h, PITCH):
                            add((gx, gy, round(step_z + rh, 1)), COLORS['plaza_accent'])

        # Top rail (handrail cap) — continuous
        top_rail_z = step_z + rail_h
        for x in np.arange(ra_x1, ra_x2 + PITCH, PITCH * 0.5):
            for y in [ra_y1, ra_y2]:
                gx, gy = snap_grid(x, y)
                if ra_x1 - 0.5 <= gx <= ra_x2 + 0.5:
                    add((gx, gy, round(top_rail_z, 1)), COLORS['building_wall'])
        for y in np.arange(ra_y1, ra_y2 + PITCH, PITCH * 0.5):
            for x in [ra_x1, ra_x2]:
                gx, gy = snap_grid(x, y)
                if ra_y1 - 0.5 <= gy <= ra_y2 + 0.5:
                    add((gx, gy, round(top_rail_z, 1)), COLORS['building_wall'])
    print(f"  Balustrades added")

    # ================================================================
    #  9. ESCALATOR SYSTEM — crossing diagonals within void
    # ================================================================
    def build_escalator(x1, y1, z1, x2, y2, z2, width=1.5):
        """Build a single escalator as a diagonal band from (x1,y1,z1) to (x2,y2,z2)."""
        dx = x2 - x1; dy = y2 - y1; dz = z2 - z1
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        steps = int(length / (PITCH * 0.6))
        px = -dy / max(abs(dx) + abs(dy), 0.01)  # perpendicular direction x
        py = dx / max(abs(dx) + abs(dy), 0.01)    # perpendicular direction y

        for i in range(steps):
            t = i / max(steps - 1, 1)
            cx = x1 + t * dx; cy = y1 + t * dy; cz = z1 + t * dz
            gx, gy, gz = snap_grid(cx, cy, cz)
            # Step tread
            add((gx, gy, gz), COLORS['path_dark'])
            # Side rails (glass on both sides)
            for sd in [-1, 1]:
                sx = round((cx + sd * px * width / 2) / PITCH) * PITCH
                sy = round((cy + sd * py * width / 2) / PITCH) * PITCH
                for rh in [0, PITCH, PITCH * 2]:
                    add((sx, sy, round(gz + rh, 1)), COLORS['building_glass'])
            # Handrail on top of glass
            for sd in [-1, 1]:
                sx = round((cx + sd * px * width / 2) / PITCH) * PITCH
                sy = round((cy + sd * py * width / 2) / PITCH) * PITCH
                add((sx, sy, round(gz + PITCH * 3, 1)), COLORS['concrete_pillar'])

    # Escalator pair 1: G/F south to 2/F north  (Z=39→43)
    build_escalator(-3.0, -41.0, 39.5, 3.0, -35.0, 43.5)
    # Escalator pair 2: G/F north to 2/F south  (Z=39→43, cross with pair 1)
    build_escalator(3.0, -35.0, 39.5, -3.0, -41.0, 43.5)
    # Escalator pair 3: 2/F east to 3/F west  (Z=43→47)
    build_escalator(3.5, -32.0, 43.5, -3.5, -44.0, 47.5)
    # Escalator pair 4: 2/F west to 3/F east  (Z=43→47)
    build_escalator(-3.5, -44.0, 43.5, 3.5, -32.0, 47.5)
    print(f"  Escalator system added")

    # ================================================================
    # 10. NORTH CONNECTING BRIDGE — glass-roofed link to Academic Arc
    # ================================================================
    bridge_start_y = hy2  # Y=-29 (north wall)
    bridge_end_y = -16.0   # approximate arc inner wall
    bridge_len = abs(bridge_end_y - bridge_start_y)  # ~13m
    bridge_width = 8.0
    bridge_cx = atrium_cx  # X=0
    bx1 = bridge_cx - bridge_width / 2; bx2 = bridge_cx + bridge_width / 2

    # Bridge floor
    bridge_floor = filled_rect(bx1, bridge_start_y, bx2, bridge_end_y, height=2, z_offset=ground_z)
    for pos in bridge_floor: add(pos, COLORS['path_gray'])

    # Bridge columns — paired columns every 4m
    for dist in np.arange(0, bridge_len + PITCH, 4.0):
        col_y = bridge_start_y + dist
        for col_x in [bx1 + 1.0, bx2 - 1.0]:
            for ch in np.arange(ground_z, ground_z + 5.0, PITCH):
                for cd in [(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]:
                    gcol_x, gcol_y = snap_grid(col_x + cd[0], col_y + cd[1])
                    add((gcol_x, gcol_y, round(ch, 1)), COLORS['concrete_pillar'])

    # Bridge glass roof — shallow arch
    roof_z_base = ground_z + 5.0  # Z=43.0
    for dist in np.arange(0, bridge_len, PITCH * 0.6):
        by_ = bridge_start_y + dist
        for w in np.arange(-bridge_width/2, bridge_width/2, PITCH * 0.6):
            gbx, gby = snap_grid(bridge_cx + w, by_)
            if bx1 <= gbx <= bx2 and bridge_start_y <= gby <= bridge_end_y:
                # Arch shape: center 1.5m higher than edges
                w_frac = abs(w) / (bridge_width / 2)
                arch_rise = (1.0 - w_frac) * 1.5
                roof_z = roof_z_base + arch_rise
                # Steel frame every 2m along bridge
                near_frame = (abs(dist - round(dist / 2.0) * 2.0) < PITCH * 1.5)
                if near_frame:
                    add((gbx, gby, round(roof_z, 1)), COLORS['concrete_pillar'])
                    add((gbx, gby, round(roof_z - PITCH, 1)), COLORS['concrete_pillar'])
                else:
                    add((gbx, gby, round(roof_z, 1)), COLORS['building_glass_bright'])
                    add((gbx, gby, round(roof_z - PITCH, 1)), COLORS['building_glass'])

    # Bridge side railings (low glass walls)
    for side_x in [bx1, bx2]:
        for dist in np.arange(0, bridge_len, PITCH * 0.6):
            by_ = bridge_start_y + dist
            gsx, gsy = snap_grid(side_x, by_)
            if bridge_start_y <= gsy <= bridge_end_y:
                for rh in np.arange(0, 1.5, PITCH):
                    add((gsx, gsy, round(ground_z + rh, 1)), COLORS['building_glass'])
                add((gsx, gsy, round(ground_z + 1.5, 1)), COLORS['concrete_pillar'])
    print(f"  North bridge to Academic Arc added")

    # ================================================================
    # 11. "MUSHROOM" OBSERVATION DECK — cantilevered platform
    # ================================================================
    deck_cx = void_x2 + 1.0  # X=5.0, on east edge of void
    deck_cy = void_cy  # Y=-38
    deck_r = 5.0  # semicircular, radius 5m
    deck_z = ground_z + 9.0  # Z=47.0 (3/F level)

    # Semicircular platform extending east from void edge
    for x in np.arange(deck_cx - deck_r, deck_cx + deck_r + PITCH, PITCH * 0.6):
        for y in np.arange(deck_cy - deck_r, deck_cy + deck_r + PITCH, PITCH * 0.6):
            gx, gy = snap_grid(x, y)
            dist = np.sqrt((gx - deck_cx)**2 + (gy - deck_cy)**2)
            if dist > deck_r: continue
            # Only right half (east of void) and inside building envelope
            if gx < void_x2: continue
            if gy > hy2 or gy < hy1: continue
            # Platform floor
            add((gx, gy, round(deck_z, 1)), COLORS['plaza_accent'])
            add((gx, gy, round(deck_z - PITCH, 1)), COLORS['path_gray'])

    # Deck edge railing
    for ang in np.arange(-np.pi/2, np.pi/2, 0.08):
        rx = deck_cx + deck_r * np.cos(ang)
        ry = deck_cy + deck_r * np.sin(ang)
        grx, gry = snap_grid(rx, ry)
        if grx < void_x2: continue
        if gry > hy2 or gry < hy1: continue
        for rh in np.arange(0, 1.5, PITCH):
            add((grx, gry, round(deck_z + rh, 1)), COLORS['building_glass'])
        add((grx, gry, round(deck_z + 1.5, 1)), COLORS['concrete_pillar'])

    # Support brackets underneath (cantilever)
    for i in range(3):
        bx = deck_cx + 1.0 + i * 1.5
        by = deck_cy + (i - 1) * 2.5
        gbx, gby = snap_grid(bx, by)
        for bz in np.arange(ground_z + 2.0, deck_z - PITCH, PITCH):
            add((gbx, gby, round(bz, 1)), COLORS['concrete_pillar'])
    print(f"  Observation deck added")

    # ================================================================
    # 12. INTERIOR FURNISHINGS — info desk, seating, LED screen
    # ================================================================
    # Information desk (east side of void, ground floor)
    desk_cx = 6.0; desk_cy = -38.0
    desk_w = 3.0; desk_d = 2.0; desk_h = 2.0
    desk_x1 = desk_cx - desk_w/2; desk_x2 = desk_cx + desk_w/2
    desk_y1 = desk_cy - desk_d/2; desk_y2 = desk_cy + desk_d/2

    # Desk counter body
    desk_body = filled_rect(desk_x1, desk_y1, desk_x2, desk_y2, height=3, z_offset=ground_z + 2.0)
    for pos in desk_body: add(pos, COLORS['building_wall'])
    # Desk countertop (lighter accent)
    desk_top = filled_rect(desk_x1 - 0.3, desk_y1 - 0.3, desk_x2 + 0.3, desk_y2 + 0.3, height=1,
                           z_offset=ground_z + 3.5)
    for pos in desk_top: add(pos, COLORS['plaza_accent'])

    # Seating benches (west side of void, ground floor)
    bench_y_positions = [-41.0, -38.0, -35.0]
    for bi, bench_y in enumerate(bench_y_positions):
        bench_x = -6.0
        for bw in np.arange(-1.0, 1.0 + PITCH, PITCH * 0.6):
            for bd in np.arange(-0.3, 0.3 + PITCH, PITCH * 0.6):
                gbx, gby = snap_grid(bench_x + bw, bench_y + bd)
                # Bench seat at 1m height
                add((gbx, gby, round(ground_z + 2.0, 1)), COLORS['path_dark'])
                # Bench legs at ends
                if abs(bw) > 0.5:
                    add((gbx, gby, round(ground_z + 0.5, 1)), COLORS['concrete_pillar'])

    # Large LED screen on north interior wall
    screen_cx = atrium_cx; screen_cy = hy2 + 0.5  # just inside north wall
    screen_w = 6.0; screen_h = 3.0
    screen_z_base = ground_z + 3.0
    for x in np.arange(screen_cx - screen_w/2, screen_cx + screen_w/2 + PITCH, PITCH * 0.6):
        for z in np.arange(screen_z_base, screen_z_base + screen_h, PITCH * 0.6):
            gx, gz = snap_grid(x, screen_cy)[0], round(z / PITCH) * PITCH
            add((gx, round(screen_cy / PITCH) * PITCH, gz), COLORS['path_dark'])  # dark screen
    # Screen frame
    for x in np.arange(screen_cx - screen_w/2 - 0.3, screen_cx + screen_w/2 + 0.8, PITCH * 0.6):
        for z in [screen_z_base - PITCH, screen_z_base + screen_h]:
            gx = round(x / PITCH) * PITCH
            gz = round(z / PITCH) * PITCH
            add((gx, round(screen_cy / PITCH) * PITCH, gz), COLORS['concrete_pillar'])
    print(f"  Interior furnishings added")

    # ================================================================
    # 13. SOUTH WALKWAY — glass-roofed colonnade to plaza
    # ================================================================
    walk_start_x = atrium_cx; walk_start_y = hy1  # Y=-47
    walk_dir_x = 0.0; walk_dir_y = -1.0
    walk_len = 18.0; walk_width = 6.0
    walk_roof_z = ground_z + 5.0  # Z=43.0

    for dist in np.arange(0, walk_len, PITCH * 0.8):
        wcx = walk_start_x + walk_dir_x * dist
        wcy = walk_start_y + walk_dir_y * dist
        for w in np.arange(-walk_width/2, walk_width/2, PITCH * 0.8):
            px_perp = -walk_dir_y; py_perp = walk_dir_x
            wx = wcx + px_perp * w; wy = wcy + py_perp * w
            # Floor
            add(snap_grid(wx, wy, ground_z), COLORS['path_gray'])

            # Glass roof with shallow arch (center 1m higher than edges)
            w_frac = abs(w) / (walk_width / 2)
            arch_rise = (1.0 - w_frac) * 1.0
            roof_z = walk_roof_z + arch_rise
            # Steel frame every 3m
            near_frame = (abs(dist - round(dist / 3.0) * 3.0) < PITCH * 1.5)
            if near_frame:
                add(snap_grid(wx, wy, roof_z), COLORS['concrete_pillar'])
                add(snap_grid(wx, wy, roof_z - PITCH), COLORS['concrete_pillar'])
            else:
                add(snap_grid(wx, wy, roof_z), COLORS['building_glass_bright'])
                add(snap_grid(wx, wy, roof_z - PITCH), COLORS['building_glass'])

            # Side glass railings (instead of full columns, let light through)
            if abs(w) > walk_width/2 - 0.8:
                for rh in np.arange(0, 1.5, PITCH):
                    add(snap_grid(wx, wy, ground_z + rh), COLORS['building_glass'])
                add(snap_grid(wx, wy, ground_z + 1.5), COLORS['concrete_pillar'])

            # Structural columns at edges, every 3m
            if abs(dist - round(dist / 3.0) * 3.0) < PITCH and abs(w) > walk_width/2 - 1.2:
                for col_z in np.arange(ground_z, ground_z + 5.0, PITCH):
                    add(snap_grid(wx, wy, col_z), COLORS['concrete_pillar'])
    print(f"  Glass-roofed south walkway added")

    # ================================================================
    # 14. SURROUNDING GROUND
    # ================================================================
    ground_margin = 12
    for x in np.arange(hx1 - ground_margin, hx2 + ground_margin, PITCH):
        for y in np.arange(hy1 - ground_margin, hy2 + ground_margin, PITCH):
            gx, gy = snap_grid(x, y)
            if hx1 - 1 <= gx <= hx2 + 1 and hy1 - 1 <= gy <= hy2 + 1: continue
            # Also skip portico area
            if px1 - 1 <= gx <= px2 + 1 and py1 - 1 <= gy <= py2 + 1: continue
            # Skip north bridge area
            if bx1 - 1 <= gx <= bx2 + 1 and bridge_start_y - 1 <= gy <= bridge_end_y + 1: continue
            add((gx, gy, ground_z), COLORS['ground_gray'])
    print(f"  Surrounding ground added")

    all_positions, all_colors = [], []
    for pos, color in pos_color.items():
        all_positions.append(pos); all_colors.append(color)

    print(f"  TOTAL: {len(all_positions)} voxels")
    return all_positions, all_colors


def build_library():
    """Build HKUST Library — entrance atrium, curved glass facade, LG1, roof garden."""
    print("\n--- Building Library ---")

    lib_cx = 40.0; lib_cy = 10.0; ground_z = 38.0
    lib_width = 20.0; lib_depth = 30.0; lib_height = 20.0  # 5 floors @ 4m

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    def add_tree(tx, ty, tz):
        gx, gy = snap_grid(tx, ty)
        th = 3 + int(abs(hash(str((tx, ty))) % 1000) / 250)
        for hh in range(th):
            for tdx in [-0.5, 0.0, 0.5]:
                for tdy in [-0.5, 0.0, 0.5]:
                    if abs(tdx) + abs(tdy) <= 1.0:
                        add(snap_grid(gx + tdx, gy + tdy, tz + hh * PITCH), COLORS['trunk_brown'])
        for cl_r, cl_c in [(1.5, 'tree_dark'), (1.0, 'tree_green')]:
            layer = filled_circle((gx, gy), cl_r, z=tz + th * PITCH + (2.0 - cl_r))
            for pos in layer: add(pos, COLORS[cl_c])

    hx1 = lib_cx - lib_width / 2; hx2 = lib_cx + lib_width / 2
    hy1 = lib_cy - lib_depth / 2; hy2 = lib_cy + lib_depth / 2
    roof_z = ground_z + lib_height

    print(f"  Position: ({lib_cx}, {lib_cy}), {lib_width:.0f}m x {lib_depth:.0f}m")

    # ═══ FLOOR SLABS & WALLS ═══
    for floor in range(5):
        fz = ground_z + floor * 4.0
        is_ground = (floor == 0)
        slab = filled_rect(hx1, hy1, hx2, hy2, height=2, z_offset=fz)
        for pos in slab: add(pos, COLORS['building_wall'])

        # East wall: curved glass facade (gentle curve, R=80m)
        wall_z_start = fz + 1.0
        for wz in np.arange(wall_z_start, fz + 4.0, PITCH):
            for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
                gy = round(y / PITCH) * PITCH
                # West wall (faces arc interior)
                if not (is_ground and abs(gy - lib_cy) < 3.0):
                    add(snap_grid(hx1, y, wz), COLORS['building_glass_dark'])
                # East wall (sea-facing): curved glass
                curve_offset = 1.0 * (1.0 - ((gy - lib_cy) / (lib_depth/2))**2)
                if not (is_ground and abs(gy - lib_cy) < 3.0):
                    add(snap_grid(hx2 + curve_offset, y, wz), COLORS['building_glass_bright'])

        # North & South end walls (with entrance cutout on north, ground floor)
        for hz in np.arange(fz + 1.0, fz + 4.0, PITCH):
            for x in np.arange(hx1, hx2 + PITCH, PITCH * 0.8):
                gx = round(x / PITCH) * PITCH
                if not (is_ground and abs(gx - lib_cx) < 2.0):
                    add(snap_grid(x, hy1, hz), COLORS['building_wall'])
                if not (is_ground and abs(gx - lib_cx) < 2.0):
                    add(snap_grid(x, hy2, hz), COLORS['building_wall'])

    # ═══ ENTRANCE ATRIUM (north face, 2-story glass) ═══
    ent_cx = lib_cx; ent_cy = hy2; ent_w = 6.0
    for ez in np.arange(ground_z + 0.5, ground_z + 8.0, PITCH):
        for ex in np.arange(ent_cx - ent_w/2, ent_cx + ent_w/2 + PITCH, PITCH * 0.6):
            add(snap_grid(ex, ent_cy, ez), COLORS['building_glass_bright'])
    # Entrance columns
    for col_off in [-ent_w/2 - 0.5, ent_w/2 + 0.5]:
        for ch in np.arange(ground_z, ground_z + 8.0, PITCH):
            add(snap_grid(ent_cx + col_off, ent_cy + 0.5, ch), COLORS['concrete_pillar'])

    # ═══ NORTH PORTICO (concourse connection) ═══
    port_x1, port_x2 = ent_cx - 5.0, ent_cx + 5.0
    port_y1, port_y2 = ent_cy, ent_cy + 4.0
    port_z = ground_z + 8.0
    canopy = filled_rect(port_x1, port_y1, port_x2, port_y2, height=1, z_offset=port_z)
    for pos in canopy: add(pos, COLORS['building_roof_dark'])
    for pcx in [port_x1 + 1.0, port_x2 - 1.0]:
        for pch in np.arange(ground_z, port_z, PITCH):
            add(snap_grid(pcx, port_y2 - 0.5, pch), COLORS['concrete_pillar'])

    # ═══ LG1 LEVEL (below ground, west side) ═══
    lg1_z = ground_z - 4.0
    lg1_h = 4.0
    lg1_x1, lg1_x2 = hx1 + 2.0, hx2 - 2.0
    lg1_y1, lg1_y2 = hy1 + 2.0, hy2 - 2.0
    for lz in np.arange(lg1_z, ground_z, PITCH):
        for x in np.arange(lg1_x1, lg1_x2 + PITCH, PITCH * 0.6):
            add(snap_grid(x, lg1_y1, lz), COLORS['building_wall_dark'])
            add(snap_grid(x, lg1_y2, lz), COLORS['building_wall_dark'])
        for y in np.arange(lg1_y1, lg1_y2 + PITCH, PITCH * 0.6):
            add(snap_grid(lg1_x1, y, lz), COLORS['building_wall_dark'])
        # Narrow slit windows
        for wy in np.arange(lg1_y1, lg1_y2, 2.5):
            add(snap_grid(lg1_x2, wy, lz), COLORS['building_glass_dark'])

    # ═══ READING ROOM GOLD BAND (3/F, east facade) ═══
    gold_z = ground_z + 12.0  # 3/F
    for gy in np.arange(hy1 + 4.0, hy2 - 4.0, PITCH * 0.4):
        gx, gy_snapped = snap_grid(hx2 + 1.5, gy)
        add((gx, gy_snapped, gold_z), COLORS['sundial_gold'])
        add((gx, gy_snapped, gold_z + 0.5), COLORS['sundial_gold'])

    # ═══ VERTICAL CIRCULATION CORE (west side, dark band) ═══
    core_x1, core_x2 = hx1 - 1.0, hx1 + 2.0
    core_y1, core_y2 = lib_cy - 3.0, lib_cy + 3.0
    for cz in np.arange(ground_z, roof_z, PITCH):
        for cx in np.arange(core_x1, core_x2 + PITCH, PITCH * 0.6):
            add(snap_grid(cx, core_y1, cz), COLORS['building_wall_dark'])
            add(snap_grid(cx, core_y2, cz), COLORS['building_wall_dark'])
        for cy in np.arange(core_y1, core_y2 + PITCH, PITCH * 0.6):
            add(snap_grid(core_x1, cy, cz), COLORS['building_wall_dark'])
            add(snap_grid(core_x2, cy, cz), COLORS['building_wall_dark'])

    # ═══ ROOF ═══
    roof = filled_rect(hx1, hy1, hx2, hy2, height=2, z_offset=roof_z)
    for pos in roof: add(pos, COLORS['building_roof'])
    # Parapet
    for x in [hx1, hx2]:
        for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
            add(snap_grid(x, y, roof_z + 2.0), COLORS['building_roof_dark'])
            add(snap_grid(x, y, roof_z + 2.5), COLORS['building_roof_dark'])
    for y in [hy1, hy2]:
        for x in np.arange(hx1, hx2 + PITCH, PITCH * 0.8):
            add(snap_grid(x, y, roof_z + 2.0), COLORS['building_roof_dark'])

    # ═══ ROOF GARDEN (east half of roof) ═══
    for rx in np.arange(lib_cx, hx2 + 1.0, PITCH * 0.6):
        for ry in np.arange(hy1 + 2.0, hy2 - 1.0, PITCH * 0.6):
            add(snap_grid(rx, ry, roof_z + 2.5), COLORS['hillside_grass'])
    # Roof trees
    for (rtx, rty) in [(lib_cx + 3, lib_cy + 6), (lib_cx + 6, lib_cy - 4), (lib_cx + 3, lib_cy)]:
        add_tree(rtx, rty, roof_z + 2.5)

    # ═══ BRIDGE TO ARC (west side) ═══
    bridge_y = lib_cy; bridge_start_x = hx1; bridge_end_x = 28.0
    bridge_z = ground_z + 8.0
    for bx in np.arange(bridge_end_x, bridge_start_x, PITCH * 0.8):
        for bw in np.arange(-2.5, 2.5, PITCH * 0.8):
            gx, gy = snap_grid(bx, bridge_y + bw)
            add((gx, gy, bridge_z), COLORS['path_gray'])
            add((gx, gy, bridge_z + 0.5), COLORS['path_gray'])
        # Glass canopy roof with arch
        gx = round(bx / PITCH) * PITCH
        for bw in np.arange(-2.0, 2.0, PITCH * 0.6):
            gy = round(bridge_y + bw, 1)
            arch = 1.0 * (1.0 - ((gx - (bridge_start_x + bridge_end_x)/2) / (abs(bridge_start_x - bridge_end_x)/2))**2)
            arch = max(0, arch)
            add((gx, gy, bridge_z + 3.0 + arch), COLORS['building_glass_bright'])

    # ═══ GROUND ═══
    for x in np.arange(hx1 - 10, hx2 + 10, PITCH):
        for y in np.arange(hy1 - 10, hy2 + 10, PITCH):
            gx, gy = snap_grid(x, y)
            if hx1 - 1 <= gx <= hx2 + 1 and hy1 - 1 <= gy <= hy2 + 1: continue
            add((gx, gy, ground_z), COLORS['ground_gray'])

    all_positions, all_colors = [], []
    for pos, color in pos_color.items():
        all_positions.append(pos); all_colors.append(color)
    print(f"  TOTAL: {len(all_positions)} voxels")
    return all_positions, all_colors


def build_lsk():
    """Build the Lee Shau Kee Business Building — independent south-west building.

    Real HKUST: LSK is a separate, unconnected building south-west of the
    Academic Building, further uphill. Not physically connected to the arc.
    """
    print("\n--- Building Lee Shau Kee Business Building ---")

    lsk_cx = -50.0; lsk_cy = -50.0; ground_z = 38.0
    lsk_width = 20.0; lsk_depth = 40.0; lsk_height = 12.0  # 3 floors, 4th=6-story tower

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    def add_tree(tx, ty, tz):
        gx, gy = snap_grid(tx, ty)
        th = 4 + int(abs(hash(str((tx, ty))) % 1000) / 250)
        for hh in range(th):
            add(snap_grid(gx, gy, tz + hh * PITCH), COLORS['trunk_brown'])
        for cl_r, cl_c in [(2.0, 'tree_dark'), (1.5, 'tree_green')]:
            layer = filled_circle((gx, gy), cl_r, z=tz + th * PITCH + (3.0 - cl_r))
            for pos in layer: add(pos, COLORS[cl_c])

    hy1 = lsk_cy - lsk_depth / 2; hy2 = lsk_cy + lsk_depth / 2
    hx1 = lsk_cx - lsk_width / 2; hx2 = lsk_cx + lsk_width / 2
    roof_z = ground_z + lsk_height

    print(f"  Position: ({lsk_cx}, {lsk_cy}), {lsk_width:.0f}m x {lsk_depth:.0f}m")

    # ═══ 3 MAIN FLOORS (parallelogram offset for angled facade) ═══
    for floor in range(3):
        fz = ground_z + floor * 4.0
        # North wall shifts east by 5m relative to south (angled)
        offset_n = 5.0 * (floor / 3.0)  # progressive offset
        slab_hy1 = hy1; slab_hy2 = hy2
        slab_hx1_n = hx1 + offset_n * 0.5; slab_hx2_n = hx2 + offset_n * 0.5
        slab_hx1_s = hx1 - offset_n * 0.5; slab_hx2_s = hx2 - offset_n * 0.5

        # Floor slab (use average boundaries)
        avg_hx1 = (slab_hx1_s + slab_hx1_n) / 2; avg_hx2 = (slab_hx2_s + slab_hx2_n) / 2
        slab = filled_rect(avg_hx1, hy1, avg_hx2, hy2, height=2, z_offset=fz)
        for pos in slab: add(pos, COLORS['building_wall'])

        # Glass walls (east/west)
        for wz in np.arange(fz + 1.0, fz + 4.0, PITCH):
            for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
                t = (y - hy1) / (hy2 - hy1)
                wx_off = offset_n * (t - 0.5)
                add(snap_grid(hx1 + wx_off, y, wz), COLORS['building_glass_dark'])
                add(snap_grid(hx2 + wx_off, y, wz), COLORS['building_glass_bright'])
            # South wall
            for x in np.arange(avg_hx1, avg_hx2 + PITCH, PITCH * 0.8):
                if not (floor == 0 and abs(x - lsk_cx) < 3.0):
                    add(snap_grid(x, hy1, wz), COLORS['building_wall'])
            # North wall (with entrance cutout ground floor)
            for x in np.arange(avg_hx1, avg_hx2 + PITCH, PITCH * 0.8):
                if floor == 0 and abs(x - lsk_cx) < 2.0: continue
                add(snap_grid(x, hy2, wz), COLORS['building_wall'])

    # ═══ INTERNAL COURTYARD (6m x 10m void through all floors) ═══
    court_x1, court_x2 = lsk_cx - 3.0, lsk_cx + 3.0
    court_y1, court_y2 = lsk_cy - 5.0, lsk_cy + 5.0
    court_roof_z = roof_z
    court_glass = filled_rect(court_x1, court_y1, court_x2, court_y2, height=1, z_offset=court_roof_z)
    for pos in court_glass: add(pos, COLORS['building_glass_bright'])
    # Internal tree
    add_tree(lsk_cx, lsk_cy, ground_z + 0.5)

    # ═══ SOUTH TERRACED STEP-BACK ═══
    for floor in range(1, 3):
        terrace_z = ground_z + floor * 4.0 + 0.5
        tb_hy1 = hy1 + floor * 3.0  # step back
        for tx in np.arange(hx1 + 2.0, hx2 - 1.0, PITCH * 0.6):
            for ty in np.arange(hy1, tb_hy1, PITCH * 0.6):
                add(snap_grid(tx, ty, terrace_z), COLORS['path_gray'])
        # Glass railing
        for rx in np.arange(hx1 + 2.0, hx2 - 1.0, PITCH * 0.6):
            for rz in np.arange(terrace_z + 0.5, terrace_z + 2.0, PITCH):
                add(snap_grid(rx, tb_hy1, rz), COLORS['building_glass'])

    # ═══ 6-STORY FACULTY TOWER (east third, 24m tall) ═══
    tower_x1, tower_x2 = lsk_cx + 3.0, hx2
    tower_y1, tower_y2 = lsk_cy - 10.0, lsk_cy + 10.0
    tower_z = roof_z; tower_h = 12.0
    for tf in range(3):
        tfz = tower_z + tf * 4.0
        t_slab = filled_rect(tower_x1, tower_y1, tower_x2, tower_y2, height=2, z_offset=tfz)
        for pos in t_slab: add(pos, COLORS['building_wall_dark'])
        for wz in np.arange(tfz + 1.0, tfz + 4.0, PITCH):
            for y in np.arange(tower_y1, tower_y2 + PITCH, PITCH * 0.8):
                add(snap_grid(tower_x1, y, wz), COLORS['building_glass'])
                add(snap_grid(tower_x2, y, wz), COLORS['building_glass_bright'])
    # Tower roof
    t_roof = filled_rect(tower_x1, tower_y1, tower_x2, tower_y2, height=2, z_offset=tower_z + tower_h)
    for pos in t_roof: add(pos, COLORS['building_roof'])

    # ═══ MAIN ENTRANCE + PLAZA (north face) ═══
    ent_w = 8.0
    ent_cx = lsk_cx
    for ez in np.arange(ground_z + 0.5, ground_z + 5.0, PITCH):
        for ex in np.arange(ent_cx - ent_w/2, ent_cx + ent_w/2 + PITCH, PITCH * 0.6):
            add(snap_grid(ex, hy2, ez), COLORS['building_glass_bright'])
    # Portico canopy (4 columns, 4m projection)
    can_x1, can_x2 = ent_cx - ent_w/2 - 1.5, ent_cx + ent_w/2 + 1.5
    can_y1, can_y2 = hy2, hy2 + 4.0
    can_z = ground_z + 4.5
    canopy = filled_rect(can_x1, can_y1, can_x2, can_y2, height=1, z_offset=can_z)
    for pos in canopy: add(pos, COLORS['building_roof_dark'])
    for pcx in [can_x1 + 1.0, ent_cx - 1.5, ent_cx + 1.5, can_x2 - 1.0]:
        for pch in np.arange(ground_z, can_z, PITCH):
            add(snap_grid(pcx, can_y2 - 0.5, pch), COLORS['concrete_pillar'])
    # Gold LSK signage
    for sx in np.arange(ent_cx - 3.0, ent_cx + 3.0, PITCH * 0.6):
        add(snap_grid(sx, hy2, can_z + 0.5), COLORS['sundial_gold'])

    # Entrance plaza
    plz_y1, plz_y2 = can_y2, can_y2 + 12.0
    plz_x1, plz_x2 = ent_cx - 8.0, ent_cx + 8.0
    plz_fill = filled_rect(plz_x1, plz_y1, plz_x2, plz_y2, height=2, z_offset=ground_z)
    for pos in plz_fill: add(pos, COLORS['path_gray'])
    for bx in [plz_x1, plz_x2]:
        for by in np.arange(plz_y1, plz_y2, PITCH * 0.6):
            add(snap_grid(bx, by, ground_z + 0.5), COLORS['plaza_accent'])
    # Plaza trees
    for (ptx, pty) in [(lsk_cx - 3, plz_y1 + 4), (lsk_cx + 3, plz_y1 + 4), (lsk_cx, plz_y1 + 9)]:
        add_tree(ptx, pty, ground_z + 1.0)

    # ═══ EAST-WEST BRIDGE CONNECTOR (to Academic Arc, Z=42) ═══
    br_start_x = hx2; br_end_x = -32.0; br_y = lsk_cy; br_z = ground_z + 4.0
    for bx in np.arange(br_end_x, br_start_x, PITCH * 0.6):
        for bw in np.arange(-2.0, 2.0, PITCH * 0.6):
            gx, gy = snap_grid(bx, br_y + bw)
            add((gx, gy, round(br_z, 1)), COLORS['path_gray'])
        # Glass roof
        gx = round(bx / PITCH) * PITCH
        for bw in np.arange(-1.5, 1.5, PITCH * 0.6):
            gy = round(br_y + bw, 1)
            add((gx, gy, br_z + 3.5), COLORS['building_glass_bright'])

    # ═══ ROOF ═══
    roof = filled_rect(hx1, hy1, hx2, hy2, height=2, z_offset=roof_z)
    for pos in roof: add(pos, COLORS['building_roof'])
    for x in [hx1, hx2]:
        for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
            add(snap_grid(x, y, roof_z + 2.0), COLORS['building_roof_dark'])

    # ═══ GROUND ═══
    for x in np.arange(hx1 - 8, hx2 + 8, PITCH):
        for y in np.arange(hy1 - 8, hy2 + 8, PITCH):
            gx, gy = snap_grid(x, y)
            if hx1 - 1 <= gx <= hx2 + 1 and hy1 - 1 <= gy <= hy2 + 1: continue
            if plz_x1 <= gx <= plz_x2 and plz_y1 <= gy <= plz_y2: continue
            add((gx, gy, ground_z), COLORS['ground_gray'])

    all_positions, all_colors = [], []
    for pos, color in pos_color.items():
        all_positions.append(pos); all_colors.append(color)
    print(f"  TOTAL: {len(all_positions)} voxels")
    return all_positions, all_colors


def build_coastline():
    """Build ocean, beach, and coastal rocks south of the academic complex.

    HKUST sits on a hillside overlooking Port Shelter (牛尾海).
    The academic podium is elevated ~38m above sea level.
    The coastline runs roughly along the southern edge.
    """
    print("\n--- Building Coastline & Ocean ---")

    ocean_y_start = -185.0   # shoreline Y (north edge of ocean)
    ocean_y_end = -350.0     # far edge of ocean
    ocean_x_start = -150.0
    ocean_x_end = 220.0
    sea_level = -1.0

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    # Ocean — 5 layers deep with wave-patterned surface
    # FIXED: np.arange from smaller (ocean_y_end) to larger (ocean_y_start)
    ocean_depths = [-4.0, -3.0, -2.0, -1.0, 0.0]
    for oz in ocean_depths:
        is_surface = (oz >= -0.5)
        for x in np.arange(ocean_x_start, ocean_x_end, PITCH * 2):
            for y in np.arange(ocean_y_end, ocean_y_start + PITCH, PITCH * 2):
                gx, gy = snap_grid(x, y)
                if gy > ocean_y_start - 1: continue

                dist = abs(gy - ocean_y_start)

                if is_surface:
                    # Wave pattern on surface: sin-based texture
                    wave = math.sin(gx * 0.3) * math.cos(gy * 0.25)
                    # White foam near shore
                    if dist < 6:
                        color = COLORS['white'] if abs(wave) > 0.3 else COLORS['water_light']
                    elif dist < 15:
                        color = COLORS['water_light'] if wave > 0 else COLORS['water_blue']
                    elif dist < 40:
                        color = COLORS['water_blue'] if wave > -0.2 else COLORS['water_dark']
                    else:
                        color = COLORS['deep_ocean']
                elif oz < -2.5:
                    color = COLORS['deep_ocean']
                elif oz < -1.5:
                    color = COLORS['water_dark']
                else:
                    color = COLORS['water_blue']

                add((gx, gy, round(oz, 1)), color)

    print(f"  Ocean (5 layers + wave surface): {len(pos_color)} items so far")

    # Irregular shoreline using sin/cos modulation
    for x in np.arange(ocean_x_start, ocean_x_end, PITCH * 0.8):
        shore_y = ocean_y_start + math.sin(x * 0.06) * 4.0 + math.cos(x * 0.12) * 3.0
        for y_off in np.arange(-2, 3, PITCH):
            gy = round((shore_y + y_off) / PITCH) * PITCH
            gx = round(x / PITCH) * PITCH
            if abs(y_off) < PITCH * 2:
                add((round(gx, 1), round(gy, 1), 0.0), COLORS['beach_sand'])
            else:
                add((round(gx, 1), round(gy, 1), 0.0), COLORS['sand_beige'])

    print(f"  Irregular beach added")

    # Coastal rocks
    rock_positions = [
        (-30, -183), (-15, -187), (0, -182), (20, -186), (35, -180),
        (50, -185), (65, -182), (80, -187), (90, -183), (70, -188),
        (-40, -188), (-20, -190), (-50, -185), (100, -182), (95, -190)
    ]
    for rx, ry in rock_positions:
        rock_r = 1.5 + abs(hash(f"r{rx}{ry}")) % 3
        for rh in np.arange(0, 3.5, PITCH):
            for rdx in np.arange(-rock_r, rock_r + PITCH, PITCH * 0.8):
                for rdy in np.arange(-rock_r, rock_r + PITCH, PITCH * 0.8):
                    if rdx**2 + rdy**2 <= rock_r**2:
                        add(snap_grid(rx + rdx, ry + rdy, rh), COLORS['coast_rock'])

    print(f"  Coastal rocks added")

    # Hillside slope — two segments for terraced effect
    podium_z = 38.0

    # Segment 1: podium edge to lower terrace (steep main hillside)
    terrace1_y_start = -155.0  # lower terrace (track level)
    terrace1_y_end = -85.0     # podium edge
    terrace1_z_start = 2.0
    terrace1_z_end = podium_z

    for x in np.arange(-120, 200, PITCH * 0.8):
        for y in np.arange(terrace1_y_start, terrace1_y_end, PITCH * 0.8):
            gx, gy = snap_grid(x, y)
            if gy >= terrace1_y_end: continue
            t = (gy - terrace1_y_start) / (terrace1_y_end - terrace1_y_start)
            t = max(0, min(1, t))
            slope_z = terrace1_z_start + t * (terrace1_z_end - terrace1_z_start)
            gz = round(slope_z / PITCH) * PITCH
            if t > 0.7:
                color = COLORS['hillside_grass']
            elif t > 0.4:
                color = COLORS['hillside_dirt'] if (int(gx) + int(gy)) % 5 != 0 else COLORS['hillside_grass']
            elif t > 0.15:
                color = COLORS['hillside_dirt']
            else:
                color = COLORS['hillside_rock']
            add((gx, gy, round(gz, 1)), color)
            for fill_z in np.arange(0, gz, PITCH * 2):
                add((gx, gy, round(fill_z, 1)), COLORS['hillside_rock'])

    print(f"  Segment 1 (podium to lower terrace): added")

    # Segment 2: lower terrace to shoreline (gentle coastal slope)
    terrace2_y_start = -185.0  # shoreline
    terrace2_y_end = -155.0    # lower terrace
    terrace2_z_start = 0.0
    terrace2_z_end = 2.0

    for x in np.arange(ocean_x_start, ocean_x_end, PITCH * 0.8):
        for y in np.arange(terrace2_y_start, terrace2_y_end, PITCH * 0.8):
            gx, gy = snap_grid(x, y)
            if gy >= terrace2_y_end: continue
            t = (gy - terrace2_y_start) / (terrace2_y_end - terrace2_y_start)
            t = max(0, min(1, t))
            slope_z = terrace2_z_start + t * (terrace2_z_end - terrace2_z_start)
            gz = round(slope_z / PITCH) * PITCH
            color = COLORS['hillside_grass'] if t > 0.5 else COLORS['beach_sand']
            add((gx, gy, round(gz, 1)), color)
            for fill_z in np.arange(-2, gz, PITCH * 2):
                add((gx, gy, round(fill_z, 1)), COLORS['hillside_rock'])

    print(f"  Segment 2 (lower terrace to shoreline): added")

    # Seawall — raised concrete edge along shoreline
    for x in np.arange(ocean_x_start, ocean_x_end, PITCH * 1.5):
        shore_y = ocean_y_start + math.sin(x * 0.06) * 4.0 + math.cos(x * 0.12) * 3.0
        sw_y = round(shore_y / PITCH) * PITCH
        sw_x = round(x / PITCH) * PITCH
        for sw_h in np.arange(0, 1.5, PITCH):
            for sw_w in np.arange(-1.0, 1.0, PITCH * 0.8):
                add(snap_grid(sw_x, sw_y + sw_w, sw_h), COLORS['coast_rock'])

    print(f"  Seawall: added along shoreline")

    all_positions, all_colors = [], []
    for pos, color in pos_color.items():
        all_positions.append(pos); all_colors.append(color)

    print(f"  TOTAL: {len(all_positions)} voxels")
    return all_positions, all_colors


# ═══════════════════════════════════════════════════════════════
#  NEW CAMPUS BUILDINGS — Part C & D
# ═══════════════════════════════════════════════════════════════

def build_cyt():
    """Build Cheng Yu Tung Building — south of Academic Arc, 8 floors, with bridge connection."""
    print("\n--- Building CYT (Cheng Yu Tung Building) ---")

    cx, cy = 0.0, -75.0; ground_z = 38.0
    w, d, h = 28.0, 18.0, 32.0  # 8 floors @ 4m

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    hx1, hx2 = cx - w/2, cx + w/2
    hy1, hy2 = cy - d/2, cy + d/2
    roof_z = ground_z + h

    # ═══ MAIN BUILDING STRUCTURE ═══
    for floor in range(8):
        fz = ground_z + floor * 4.0
        # Floor slabs
        slab = filled_rect(hx1, hy1, hx2, hy2, height=2, z_offset=fz)
        for pos in slab: add(pos, COLORS['building_wall'])

        # Glass curtain walls on east/west long sides
        is_ground = (floor == 0)
        for wz in np.arange(fz + 1.0, fz + 4.0, PITCH):
            for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
                gy_snapped = round(y / PITCH) * PITCH
                # East wall: brighter glass (morning sun)
                if not (is_ground and abs(gy_snapped - cy) < 3.0):
                    add(snap_grid(hx1, y, wz), COLORS['building_glass'])
                # West wall: slightly darker glass
                if not (is_ground and abs(gy_snapped - cy) < 3.0):
                    add(snap_grid(hx2, y, wz), COLORS['building_glass_bright'])

        # Vertical concrete fins on east/west facades (every 3m)
        for y_pos in np.arange(hy1, hy2 + PITCH, PITCH * 2):
            fy = round(y_pos / PITCH) * PITCH
            if abs(fy - cy) % 3.0 < 0.5:
                for fz2 in np.arange(fz + 1.0, fz + 4.0, PITCH):
                    add(snap_grid(hx1, fy, fz2), COLORS['concrete_pillar'])
                    add(snap_grid(hx2, fy, fz2), COLORS['concrete_pillar'])

        # Horizontal spandrel bands between floors (on east/west)
        if floor > 0:
            for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
                add(snap_grid(hx1, y, fz - 0.5), COLORS['building_wall_dark'])
                add(snap_grid(hx2, y, fz - 0.5), COLORS['building_wall_dark'])

        # North & South end walls with punched window grid
        for hz in np.arange(fz + 1.0, fz + 4.0, PITCH):
            for x in np.arange(hx1, hx2 + PITCH, PITCH * 0.8):
                gx = round(x / PITCH) * PITCH
                window = (abs(gx - cx) % 3.0 < 1.5) and (abs(hz - fz - 2.0) < PITCH)
                c = COLORS['building_glass'] if window else COLORS['building_wall']
                if not (is_ground and abs(gx) < 3.0):
                    add(snap_grid(x, hy1, hz), c)
                if not (is_ground and abs(gx) < 3.0):
                    add(snap_grid(x, hy2, hz), c)

    # ═══ ROOF ═══
    roof = filled_rect(hx1, hy1, hx2, hy2, height=2, z_offset=roof_z)
    for pos in roof: add(pos, COLORS['building_roof'])

    # Roof parapet
    for x in np.arange(hx1, hx2 + PITCH, PITCH * 0.8):
        gx, gy = snap_grid(x, hy1)
        add((gx, gy, roof_z + 2.0), COLORS['building_roof_dark'])
        add((gx, gy, roof_z + 2.5), COLORS['building_roof_dark'])
        gx, gy = snap_grid(x, hy2)
        add((gx, gy, roof_z + 2.0), COLORS['building_roof_dark'])
        add((gx, gy, roof_z + 2.5), COLORS['building_roof_dark'])
    for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
        gx, gy = snap_grid(hx1, y)
        add((gx, gy, roof_z + 2.0), COLORS['building_roof_dark'])
        gx, gy = snap_grid(hx2, y)
        add((gx, gy, roof_z + 2.0), COLORS['building_roof_dark'])

    # Roof mechanical penthouse (east half)
    mech_x1, mech_x2 = 2.0, 12.0; mech_y1, mech_y2 = -82.0, -70.0
    mech_h = 8  # 4m = 8 half-meter layers
    mech_walls = filled_rect(mech_x1, mech_y1, mech_x2, mech_y2, height=mech_h, z_offset=roof_z + 2.0)
    for pos in mech_walls: add(pos, COLORS['building_wall_dark'])
    # Metal louver strips on mechanical room
    for lx in np.arange(mech_x1 + 1.0, mech_x2, PITCH * 3):
        for lz in np.arange(roof_z + 2.5, roof_z + 2.0 + mech_h, PITCH):
            for ly in np.arange(mech_y1, mech_y2 + PITCH, PITCH * 0.8):
                add(snap_grid(lx, ly, lz), COLORS['metal_gray'])
    mech_roof = filled_rect(mech_x1, mech_y1, mech_x2, mech_y2, height=1, z_offset=roof_z + 2.0 + mech_h)
    for pos in mech_roof: add(pos, COLORS['building_roof'])

    # Cooling tower (cylinder on mechanical roof)
    tower_cx, tower_cy = 7.0, -76.0; tower_r = 1.5; tower_h = 3.0
    for tz in np.arange(roof_z + 2.0 + mech_h, roof_z + 2.0 + mech_h + tower_h, PITCH):
        ring = circle_ring((tower_cx, tower_cy), tower_r, 0.5, height=1, z_offset=tz)
        for pos in ring: add(pos, COLORS['metal_gray'])
    top_fill = filled_circle((tower_cx, tower_cy), tower_r, z=roof_z + 2.0 + mech_h + tower_h)
    for pos in top_fill: add(pos, COLORS['metal_gray'])

    # HVAC duct on west roof half
    duct_x1, duct_x2 = -10.0, -6.0; duct_y1, duct_y2 = -82.0, -74.0
    duct = filled_rect(duct_x1, duct_y1, duct_x2, duct_y2, height=2, z_offset=roof_z + 2.5)
    for pos in duct: add(pos, COLORS['metal_gray'])

    # ═══ MAIN ENTRANCE LOBBY (north face, facing Academic Arc) ═══
    ent_cx, ent_cy = cx, hy2  # center of north wall
    ent_w = 8.0  # entrance width
    ent_h = 6.0  # entrance height (ground + 1st floor)

    # Recessed glass entrance
    for ez in np.arange(ground_z + 0.5, ground_z + ent_h, PITCH):
        for ex in np.arange(ent_cx - ent_w/2, ent_cx + ent_w/2 + PITCH, PITCH * 0.6):
            add(snap_grid(ex, ent_cy, ez), COLORS['building_glass_bright'])

    # Entrance frame columns
    for cz_off in [-ent_w/2 - 0.5, ent_w/2 + 0.5]:
        for ch in np.arange(ground_z, ground_z + ent_h + 1.0, PITCH):
            add(snap_grid(ent_cx + cz_off, ent_cy + 0.5, ch), COLORS['concrete_pillar'])
            add(snap_grid(ent_cx + cz_off, ent_cy - 0.5, ch), COLORS['concrete_pillar'])

    # Entrance canopy (projects north from building)
    can_x1, can_x2 = ent_cx - ent_w/2 - 1.5, ent_cx + ent_w/2 + 1.5
    can_y1, can_y2 = ent_cy, ent_cy + 4.0  # 4m projection north
    can_z = ground_z + ent_h
    canopy = filled_rect(can_x1, can_y1, can_x2, can_y2, height=1, z_offset=can_z)
    for pos in canopy: add(pos, COLORS['building_roof_dark'])

    # Canopy support columns (2 at outer corners)
    for col_x in [can_x1 + 1.0, can_x2 - 1.0]:
        for ch in np.arange(ground_z, can_z, PITCH):
            add(snap_grid(col_x, can_y2 - 0.5, ch), COLORS['concrete_pillar'])

    # Entrance steps (3 grand steps)
    for step_i in range(3):
        step_y = ent_cy + 3.0 + step_i * 1.5
        step_w = ent_w + 4.0 + step_i * 2.0
        sx1, sx2 = ent_cx - step_w/2, ent_cx + step_w/2
        sy1, sy2 = step_y, step_y + 1.5
        step_z = ground_z - (3 - step_i) * 0.5
        step_fill = filled_rect(sx1, sy1, sx2, sy2, height=1, z_offset=step_z)
        for pos in step_fill:
            add(pos, COLORS['path_gray'] if step_i % 2 == 0 else COLORS['path_dark'])

    # Gold CYT signage band (north facade, 2/F level)
    sign_z = ground_z + 6.0
    sign_x1, sign_x2 = cx - 4.0, cx + 4.0
    for sx in np.arange(sign_x1, sign_x2 + PITCH, PITCH * 0.6):
        gx, gy = snap_grid(sx, ent_cy)
        add((gx, gy, sign_z), COLORS['sundial_gold'])
        add((gx, gy, sign_z + 0.5), COLORS['sundial_gold'])

    # ═══ CONNECTION BRIDGE to Academic Arc ═══
    # From CYT north wall (Y=-66) to arc inner wall (~Y=-32), 34m span
    br_start_y = hy2  # -66
    br_end_y = -32.0
    br_cx = 0.0
    br_w = 5.0  # bridge width
    br_fl_z = ground_z + 4.0  # bridge floor at Z=42 (2/F level)
    br_col_spacing = 5.0

    # Bridge floor
    br_x1, br_x2 = br_cx - br_w/2, br_cx + br_w/2
    for by in np.arange(br_start_y, br_end_y, PITCH * 0.6):
        for bx in np.arange(br_x1, br_x2 + PITCH, PITCH * 0.6):
            gx, gy = snap_grid(bx, by)
            add((gx, gy, br_fl_z), COLORS['path_gray'])
            add((gx, gy, br_fl_z + 0.5), COLORS['path_gray'])

    # Bridge support columns (paired every 5m)
    for col_y in np.arange(br_start_y + 2.0, br_end_y, br_col_spacing):
        for col_x in [br_x1 + 0.5, br_x2 - 0.5]:
            for ch in np.arange(ground_z, br_fl_z, PITCH):
                add(snap_grid(col_x, col_y, ch), COLORS['concrete_pillar'])

    # Bridge glass roof (arched)
    br_roof_z = br_fl_z + 4.0  # roof base at Z=46
    for by in np.arange(br_start_y, br_end_y, PITCH * 0.6):
        gy = round(by / PITCH) * PITCH
        # Arch: center 1.5m higher than edges
        arch_rise = 1.5 * (1.0 - ((gy - (br_start_y + br_end_y)/2) / ((br_end_y - br_start_y)/2))**2)
        arch_rise = max(0, arch_rise)
        for bx in np.arange(br_x1, br_x2 + PITCH, PITCH * 0.6):
            gx = round(bx / PITCH) * PITCH
            for rz in np.arange(0, 1.5, PITCH):
                roof_z_pos = br_roof_z + arch_rise + rz
                if abs(gx - br_cx) < br_w/2 - 1.0:
                    add((gx, gy, round(roof_z_pos, 1)), COLORS['building_glass_bright'])
        # Steel frame ribs every 3m
        if abs(gy - br_start_y) % 3.0 < 0.5 or abs(gy - br_end_y) % 3.0 < 0.5:
            for bx in [br_x1, br_x2]:
                gx = round(bx / PITCH) * PITCH
                for rz in np.arange(0, 2.5, PITCH):
                    add((gx, gy, round(br_roof_z + rz, 1)), COLORS['concrete_pillar'])

    # Bridge side railings (glass panels)
    for by in np.arange(br_start_y, br_end_y, PITCH * 0.6):
        gy = round(by / PITCH) * PITCH
        for bx in [br_x1 - 0.5, br_x2 + 0.5]:
            gx = round(bx / PITCH) * PITCH
            for rz in np.arange(br_fl_z + 0.5, br_fl_z + 2.0, PITCH):
                add((gx, gy, round(rz, 1)), COLORS['building_glass'])
        # Handrail cap
        for bx in [br_x1 - 0.5, br_x2 + 0.5]:
            gx = round(bx / PITCH) * PITCH
            add((gx, gy, br_fl_z + 2.0), COLORS['concrete_pillar'])

    # ═══ LECTURE THEATRE POD (south face) ═══
    pod_cx, pod_cy = 0.0, hy1  # center of south wall
    pod_r = 5.0; pod_h = 6.0; pod_w = 10.0
    # Semi-cylindrical pod protruding south
    for angle in np.arange(-90, 90, 3):
        rad = math.radians(angle)
        px = pod_cx + pod_r * math.sin(rad)
        py = pod_cy - pod_r * math.cos(rad)  # southward
        for pz in np.arange(ground_z, ground_z + pod_h, PITCH):
            for dw in np.arange(-pod_w/2, pod_w/2, PITCH * 0.6):
                gx, gy = snap_grid(px, py + dw * math.cos(rad))
                add((gx, gy, round(pz, 1)), COLORS['building_wall_dark'])
    # Clerestory window band at top of pod
    for angle in np.arange(-90, 90, 3):
        rad = math.radians(angle)
        px = pod_cx + pod_r * math.sin(rad)
        py = pod_cy - pod_r * math.cos(rad)
        for dw in np.arange(-pod_w/2, pod_w/2, PITCH * 0.6):
            gx, gy = snap_grid(px, py + dw * math.cos(rad))
            for wz in np.arange(ground_z + pod_h - 2.0, ground_z + pod_h, PITCH):
                add((gx, gy, round(wz, 1)), COLORS['building_glass_bright'])
    # Pod roof (sloped fill)
    pod_roof_z = ground_z + pod_h
    for px in np.arange(pod_cx - pod_r, pod_cx + pod_r + PITCH, PITCH * 0.6):
        for py2 in np.arange(pod_cy - pod_r - pod_w/2, pod_cy - pod_r + pod_w/2 + PITCH, PITCH * 0.6):
            gx, gy = snap_grid(px, py2)
            dist_from_cy = math.sqrt((gx - pod_cx)**2 + (gy - pod_cy)**2)
            if dist_from_cy > pod_r + 1.0: continue
            add((gx, gy, pod_roof_z), COLORS['building_roof'])

    # ═══ GROUND PLAZA & LANDSCAPING ═══
    # Paved apron around building (2m wide)
    apron_margin = 2.0
    for x in np.arange(hx1 - apron_margin, hx2 + apron_margin + PITCH, PITCH * 0.6):
        for y in np.arange(hy1 - apron_margin, hy2 + apron_margin + PITCH, PITCH * 0.6):
            gx, gy = snap_grid(x, y)
            # Only the apron (outside building, inside margin)
            in_building = (hx1 - 0.5 <= gx <= hx2 + 0.5 and hy1 - 0.5 <= gy <= hy2 + 0.5)
            in_apron = (hx1 - apron_margin <= gx <= hx2 + apron_margin and
                        hy1 - apron_margin <= gy <= hy2 + apron_margin)
            in_pod = (math.sqrt((gx - pod_cx)**2 + (gy - (pod_cy - pod_r/2))**2) < pod_r + 2.0)
            in_steps = (-6.0 <= gx <= 6.0 and hy2 <= gy <= hy2 + 7.0)
            if not in_building and in_apron and not in_pod and not in_steps:
                add((gx, gy, ground_z), COLORS['path_gray'])

    # Extended ground fill
    for x in np.arange(hx1 - 12, hx2 + 12, PITCH):
        for y in np.arange(hy1 - 12, hy2 + 12, PITCH):
            gx, gy = snap_grid(x, y)
            if hx1 - apron_margin <= gx <= hx2 + apron_margin and hy1 - apron_margin <= gy <= hy2 + apron_margin: continue
            in_pod = (math.sqrt((gx - pod_cx)**2 + (gy - (pod_cy - pod_r/2))**2) < pod_r + 4.0)
            if in_pod: continue
            add((gx, gy, ground_z), COLORS['ground_gray'])

    # Trees around building (using deterministic positions)
    tree_positions = [
        (-16, -70), (16, -70), (-16, -80), (16, -80),  # corners
        (0, -88), (-8, -88), (8, -88),  # south side
        (-16, -75), (16, -75),  # east/west
    ]
    for tx, ty in tree_positions:
        gx, gy = snap_grid(tx, ty)
        # Trunk
        trunk_h = 4 + int(abs(hash(str((tx, ty))) % 1000) / 250)
        for th in range(trunk_h):
            for tdx in [-0.5, 0.0, 0.5]:
                for tdy in [-0.5, 0.0, 0.5]:
                    if abs(tdx) + abs(tdy) <= 1.0:
                        add(snap_grid(gx + tdx, gy + tdy, ground_z + th * PITCH), COLORS['trunk_brown'])
        # Canopy layers
        for cl_r, cl_c in [(2.0, 'tree_dark'), (1.5, 'tree_green'), (1.0, 'tree_bright')]:
            layer = filled_circle((gx, gy), cl_r, z=ground_z + trunk_h * PITCH + (3.0 - cl_r))
            for pos in layer: add(pos, COLORS[cl_c])

    # ═══ RETURN ═══
    all_positions, all_colors = [], []
    for pos, color in pos_color.items():
        all_positions.append(pos); all_colors.append(color)
    print(f"  TOTAL: {len(pos_color)} voxels")
    return all_positions, all_colors


def build_halls():
    """Build UG Student Residence Halls — terraced towers with balconies, common rooms, landscaping."""
    print("\n--- Building Student Residence Halls ---")

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    # Helper: add a tree at a position
    def add_tree(tx, ty, tz):
        gx, gy = snap_grid(tx, ty)
        trunk_h = 4 + int(abs(hash(str((tx, ty))) % 1000) / 250)
        for th in range(trunk_h):
            for tdx in [-0.5, 0.0, 0.5]:
                for tdy in [-0.5, 0.0, 0.5]:
                    if abs(tdx) + abs(tdy) <= 1.0:
                        add(snap_grid(gx + tdx, gy + tdy, tz + th * PITCH), COLORS['trunk_brown'])
        for cl_r, cl_c in [(2.0, 'tree_dark'), (1.5, 'tree_green'), (1.0, 'tree_bright')]:
            layer = filled_circle((gx, gy), cl_r, z=tz + trunk_h * PITCH + (3.0 - cl_r))
            for pos in layer: add(pos, COLORS[cl_c])

    # Tower definitions with accent colors
    halls = [
        {'cx': 50, 'cy': -110, 'z': 32.0, 'w': 12, 'd': 12, 'h': 24, 'name': 'UG I-II', 'accent': 'plaza_accent'},
        {'cx': 68, 'cy': -120, 'z': 24.0, 'w': 12, 'd': 12, 'h': 24, 'name': 'UG III-IV', 'accent': 'building_wall_rib'},
        {'cx': 50, 'cy': -140, 'z': 16.0, 'w': 14, 'd': 12, 'h': 28, 'name': 'UG V-VI', 'accent': 'building_glass_dark'},
        {'cx': 72, 'cy': -155, 'z': 8.0, 'w': 14, 'd': 12, 'h': 28, 'name': 'UG VII-IX', 'accent': 'plaza_stone_mid'},
        {'cx': 95, 'cy': -165, 'z': 4.0, 'w': 12, 'd': 14, 'h': 20, 'name': 'PG Hall', 'accent': 'sundial_gold'},
    ]

    for hi, hall in enumerate(halls):
        cx, cy = hall['cx'], hall['cy']
        ground_z = hall['z']
        w, d, h = hall['w'], hall['d'], hall['h']
        n_floors = int(h / 4)
        hx1, hx2 = cx - w/2, cx + w/2
        hy1, hy2 = cy - d/2, cy + d/2
        roof_z = ground_z + h
        accent = hall['accent']

        for floor in range(n_floors):
            fz = ground_z + floor * 4.0
            is_top = (floor == n_floors - 1)

            # Floor slab
            slab = filled_rect(hx1, hy1, hx2, hy2, height=2, z_offset=fz)
            for pos in slab: add(pos, COLORS['building_wall'])

            # Windows on all 4 walls
            for wz in np.arange(fz + 1.0, min(fz + 4.0, roof_z), PITCH):
                # East/West walls
                for y in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
                    wy = round(y / PITCH) * PITCH
                    is_window = (abs(wy - cy) % 2.0 < PITCH)
                    if is_top:
                        # Top floor: all glass for common room
                        add(snap_grid(hx1, y, wz), COLORS['building_glass_bright'])
                        add(snap_grid(hx2, y, wz), COLORS['building_glass_bright'])
                    else:
                        c = COLORS['building_glass'] if is_window else COLORS['building_wall']
                        add(snap_grid(hx1, y, wz), c)
                        add(snap_grid(hx2, y, wz), c)
                # North/South walls
                for x in np.arange(hx1, hx2 + PITCH, PITCH * 0.8):
                    wx = round(x / PITCH) * PITCH
                    is_window = (abs(wx - cx) % 2.0 < PITCH)
                    if is_top:
                        add(snap_grid(x, hy1, wz), COLORS['building_glass_bright'])
                        add(snap_grid(x, hy2, wz), COLORS['building_glass_bright'])
                    else:
                        c = COLORS['building_glass'] if is_window else COLORS['building_wall']
                        add(snap_grid(x, hy1, wz), c)
                        add(snap_grid(x, hy2, wz), c)

            # South-facing balconies (sea view side, hy1 = south)
            if not is_top:
                balc_y = hy1  # south wall outer face
                for bx in np.arange(hx1 + 1.5, hx2 - 1.0, 4.0):
                    balc_x1 = bx; balc_x2 = bx + 2.5
                    bz = fz + 0.5
                    # Balcony floor slab (projects 2m south)
                    for bfx in np.arange(balc_x1, balc_x2 + PITCH, PITCH * 0.6):
                        for bfy in np.arange(balc_y - 2.0, balc_y, PITCH * 0.6):
                            gx, gy = snap_grid(bfx, bfy)
                            add((gx, gy, round(bz, 1)), COLORS['building_wall'])
                    # Balcony glass railing
                    for bgx in np.arange(balc_x1, balc_x2 + PITCH, PITCH * 0.6):
                        gx, gy = snap_grid(bgx, balc_y - 2.0)
                        for rgz in np.arange(bz + 0.5, bz + 2.0, PITCH):
                            add((gx, gy, round(rgz, 1)), COLORS['building_glass'])
                    # Corner posts
                    for cpx in [balc_x1, balc_x2]:
                        gx, gy = snap_grid(cpx, balc_y - 2.0)
                        for chz in np.arange(bz, bz + 2.0, PITCH):
                            add((gx, gy, round(chz, 1)), COLORS['concrete_pillar'])

            # Accent band at mid-height on east/west walls (floor 2-3)
            if 1 <= floor <= 2:
                for ay in np.arange(hy1, hy2 + PITCH, PITCH * 0.8):
                    add(snap_grid(hx1, ay, fz + 0.5), COLORS[accent])
                    add(snap_grid(hx2, ay, fz + 0.5), COLORS[accent])

        # ═══ ENTRANCE LOBBY (north face, hy2 = north) ═══
        ent_w = 4.0
        ent_cx = cx
        for ez in np.arange(ground_z + 0.5, ground_z + 4.0, PITCH):
            for ex in np.arange(ent_cx - ent_w/2, ent_cx + ent_w/2 + PITCH, PITCH * 0.6):
                add(snap_grid(ex, hy2, ez), COLORS['building_glass_bright'])
        # Entrance columns
        for col_off in [-ent_w/2 - 0.5, ent_w/2 + 0.5]:
            for ch in np.arange(ground_z, ground_z + 5.0, PITCH):
                add(snap_grid(ent_cx + col_off, hy2 + 0.5, ch), COLORS['concrete_pillar'])
        # Canopy
        can_x1, can_x2 = ent_cx - ent_w/2 - 1.0, ent_cx + ent_w/2 + 1.0
        can_y1, can_y2 = hy2, hy2 + 3.0
        can_z = ground_z + 3.5
        canopy = filled_rect(can_x1, can_y1, can_x2, can_y2, height=1, z_offset=can_z)
        for pos in canopy: add(pos, COLORS['building_roof_dark'])
        # Gold name panel above entrance
        for sx in np.arange(ent_cx - 2.0, ent_cx + 2.0 + PITCH, PITCH * 0.6):
            add(snap_grid(sx, hy2, can_z + 0.5), COLORS['sundial_gold'])

        # ═══ ROOF ═══
        roof = filled_rect(hx1, hy1, hx2, hy2, height=2, z_offset=roof_z)
        for pos in roof: add(pos, COLORS['building_roof'])

        # Rooftop water tank (corner of roof)
        tank_x, tank_y = cx + w/4, cy + d/4
        tank_w, tank_d = 3.0, 3.0
        tx1, tx2 = tank_x - tank_w/2, tank_x + tank_w/2
        ty1, ty2 = tank_y - tank_d/2, tank_y + tank_d/2
        tank = filled_rect(tx1, ty1, tx2, ty2, height=4, z_offset=roof_z + 2.0)
        for pos in tank: add(pos, COLORS['metal_gray'])
        # Vent pipe
        for vh in np.arange(roof_z + 2.0 + 4.0, roof_z + 2.0 + 6.0, PITCH):
            add(snap_grid(tank_x, tank_y, vh), COLORS['metal_gray'])

        # ═══ LANDSCAPING around tower ═══
        # Trees scattered near the tower (north side mainly)
        tree_offsets = [
            (cx - w/2 - 3, cy + d/2 + 2), (cx + w/2 + 3, cy + d/2 + 2),
            (cx - w/2 - 2, cy - d/2 - 2), (cx + w/2 + 2, cy - d/2 - 2),
        ]
        for tx, ty in tree_offsets:
            add_tree(tx, ty, ground_z)

        # Small paved path connecting tower entrance to main walkway
        for px in np.arange(ent_cx - 1.0, ent_cx + 1.0, PITCH * 0.6):
            for py in np.arange(hy2, hy2 + 5.0, PITCH * 0.6):
                gx, gy = snap_grid(px, py)
                add((gx, gy, ground_z), COLORS['path_gray'])

        # ═══ CONNECTING BRIDGES between adjacent halls (glass-roofed) ═══
        if hi < len(halls) - 1:
            next_h = halls[hi + 1]
            ncx, ncy = next_h['cx'], next_h['cy']
            bridge_z = min(roof_z, next_h['z'] + next_h['h']) - 4.0
            mid_x = (cx + ncx) / 2
            mid_y = (cy + ncy) / 2
            br_w = 3.0
            # Bridge floor
            for bx in np.arange(min(cx, ncx), max(cx, ncx), PITCH * 0.6):
                for bw in np.arange(-br_w/2, br_w/2, PITCH * 0.6):
                    gx, gy = snap_grid(bx, mid_y + bw)
                    add((gx, gy, round(bridge_z, 1)), COLORS['path_gray'])
                    add((gx, gy, round(bridge_z + 0.5, 1)), COLORS['path_gray'])
            # Glass roof (arched)
            br_roof_z = bridge_z + 3.5
            for bx in np.arange(min(cx, ncx), max(cx, ncx), PITCH * 0.6):
                gx = round(bx / PITCH) * PITCH
                span = abs(ncx - cx)
                arch = 1.0 * (1.0 - ((gx - (cx + ncx)/2) / (span/2))**2)
                arch = max(0, arch)
                for bw in np.arange(-br_w/2, br_w/2, PITCH * 0.6):
                    gy = round(mid_y + bw / PITCH * PITCH, 1)
                    for rz in np.arange(0, 1.0, PITCH):
                        add((gx, gy, round(br_roof_z + arch + rz, 1)), COLORS['building_glass_bright'])
            # Side railings
            for bx in np.arange(min(cx, ncx), max(cx, ncx), PITCH * 0.6):
                gx = round(bx / PITCH) * PITCH
                for side_gy in [mid_y - br_w/2, mid_y + br_w/2]:
                    for rz in np.arange(bridge_z + 0.5, bridge_z + 2.0, PITCH):
                        add((gx, round(side_gy, 1), round(rz, 1)), COLORS['building_glass'])

        # Ground fill around tower
        for x in np.arange(hx1 - 6, hx2 + 6, PITCH):
            for y in np.arange(hy1 - 6, hy2 + 6, PITCH):
                gx, gy = snap_grid(x, y)
                if hx1 - 1 <= gx <= hx2 + 1 and hy1 - 1 <= gy <= hy2 + 1: continue
                add((gx, gy, ground_z), COLORS['ground_gray'])

        print(f"    {hall['name']}: Z={ground_z:.0f}, {n_floors}F, accent={accent}")

    print(f"  TOTAL: {len(pos_color)} voxels")
    return list(pos_color.keys()), list(pos_color.values())


def build_pool():
    """Build swimming pools — 50m outdoor pool, diving, stands, changing rooms."""
    print("\n--- Building Swimming Pools ---")

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    # ═══ 50m OUTDOOR POOL ═══
    pool_cx, pool_cy = 82.0, -135.0; pool_z = 7.0
    pw, pd = 50.0, 15.0  # Olympic-length outdoor pool

    px1, px2 = pool_cx - pw/2, pool_cx + pw/2
    py1, py2 = pool_cy - pd/2, pool_cy + pd/2

    # Main pool water (deepen at north end for diving well)
    main_section = filled_rect(px1, py1, px2, pool_cy + 2.5, height=2, z_offset=pool_z)
    for pos in main_section: add(pos, COLORS['water_blue'])
    # Deep end (north 10m, Z-1)
    deep_y1, deep_y2 = pool_cy + 2.5, py2
    deep_section = filled_rect(px1, deep_y1, px2, deep_y2, height=2, z_offset=pool_z - 1.0)
    for pos in deep_section: add(pos, COLORS['water_dark'])

    # ═══ LANE MARKINGS (6 lanes) ═══
    lane_spacing = 2.5
    for lane_i in range(-2, 3):
        lane_y = pool_cy + lane_i * lane_spacing
        for lx in np.arange(px1 + 2.0, px2 - 2.0, PITCH * 0.5):
            add(snap_grid(lx, lane_y, pool_z + 0.5), COLORS['track_white'])

    # ═══ DIVING BOARDS (north end) ═══
    for d_off_x, d_h in [(-3.0, 2), (3.0, 6)]:  # 1m and 3m boards
        dx = pool_cx + d_off_x; dy = py2 - 0.5
        # Support column
        for dhz in np.arange(pool_z + 0.5, pool_z + 0.5 + d_h * PITCH, PITCH):
            add(snap_grid(dx, dy, dhz), COLORS['concrete_pillar'])
        # Diving plank (projects over water)
        plank_y = dy - 2.0
        for plx in np.arange(dx - 0.5, dx + 0.5 + PITCH, PITCH * 0.6):
            add(snap_grid(plx, plank_y, pool_z + 0.5 + d_h * PITCH), COLORS['path_gray'])

    # Starting blocks (south end, 6 blocks)
    for lane_i in range(-2, 3):
        bx = pool_cx + lane_i * lane_spacing; by = py1 + 0.5
        for bh in range(2):
            add(snap_grid(bx, by, pool_z + 0.5 + bh * PITCH), COLORS['building_wall'])

    # ═══ POOL DECK (3m wide, checker board) ═══
    d_off = 3.0
    for x in np.arange(px1 - d_off, px2 + d_off, PITCH * 0.6):
        for y_side in [py1 - d_off, py2 + d_off]:
            gx, gy = snap_grid(x, y_side)
            checker = (abs(int(gx / PITCH) + int(gy / PITCH)) % 2 == 0)
            c = COLORS['plaza_accent'] if checker else COLORS['path_gray']
            add((gx, gy, round(pool_z, 1)), c)
    for y in np.arange(py1 - d_off, py2 + d_off + PITCH, PITCH * 0.6):
        for x_side in [px1 - d_off, px2 + d_off]:
            gx, gy = snap_grid(x_side, y)
            checker = (abs(int(gx / PITCH) + int(gy / PITCH)) % 2 == 0)
            c = COLORS['plaza_accent'] if checker else COLORS['path_gray']
            add((gx, gy, round(pool_z, 1)), c)

    # Pool edge curb (0.5m raised)
    for lx in np.arange(px1, px2 + PITCH, PITCH * 0.6):
        for side_y in [py1, py2]:
            gx, gy = snap_grid(lx, side_y)
            add((gx, gy, pool_z + 0.5), COLORS['plaza_stone_dark'])
    for ly in np.arange(py1, py2 + PITCH, PITCH * 0.6):
        for side_x in [px1, px2]:
            gx, gy = snap_grid(side_x, ly)
            add((gx, gy, pool_z + 0.5), COLORS['plaza_stone_dark'])

    # ═══ SPECTATOR STANDS (west side) ═══
    stand_cx = px1 - d_off - 3.0; stand_cy = pool_cy
    stand_w = 4.0; stand_l = 20.0  # width and length of stands
    sx1, sx2 = stand_cx - stand_w/2, stand_cx + stand_w/2
    sy1, sy2 = stand_cy - stand_l/2, stand_cy + stand_l/2
    for tier in range(4):
        tz = pool_z + 0.5 + tier * 1.0
        tx1 = stand_cx - stand_w/2 - tier * 0.5
        tx2 = stand_cx + stand_w/2 - tier * 0.5
        seat = filled_rect(tx1, sy1, tx2, sy2, height=2, z_offset=tz)
        for pos in seat: add(pos, COLORS['path_gray'])
        # Riser (vertical face)
        for ry in np.arange(sy1, sy2 + PITCH, PITCH * 0.6):
            add(snap_grid(tx2, ry, tz + 1.0), COLORS['building_wall'])

    # Stand canopy roof
    roof_z = pool_z + 5.0
    rx1 = sx1 - 1.0; rx2 = sx2 + 1.0
    stand_roof = filled_rect(rx1, sy1 - 1.0, rx2, sy2 + 1.0, height=1, z_offset=roof_z)
    for pos in stand_roof: add(pos, COLORS['building_roof'])
    # Roof support columns
    for col_y_pos in np.arange(sy1, sy2 + PITCH, PITCH * 5):
        for ch in np.arange(pool_z, roof_z, PITCH):
            add(snap_grid(rx1, col_y_pos, ch), COLORS['concrete_pillar'])
            add(snap_grid(rx2, col_y_pos, ch), COLORS['concrete_pillar'])

    # ═══ CHANGING ROOM BUILDING (east side) ═══
    ch_cx, ch_cy = px2 + d_off + 4.0, pool_cy
    ch_w, ch_d, ch_h = 14.0, 8.0, 5.0
    chx1, chx2 = ch_cx - ch_w/2, ch_cx + ch_w/2
    chy1, chy2 = ch_cy - ch_d/2, ch_cy + ch_d/2
    # Walls
    for chz in np.arange(pool_z, pool_z + ch_h, PITCH):
        for x in np.arange(chx1, chx2 + PITCH, PITCH * 0.6):
            add(snap_grid(x, chy1, chz), COLORS['building_wall'])
            add(snap_grid(x, chy2, chz), COLORS['building_wall'])
        for y in np.arange(chy1, chy2 + PITCH, PITCH * 0.6):
            add(snap_grid(chx1, y, chz), COLORS['building_wall'])
            add(snap_grid(chx2, y, chz), COLORS['building_wall_dark'])
    # Windows (west face facing pool)
    for wz in np.arange(pool_z + 1.0, pool_z + ch_h - 1.0, PITCH):
        for wg_y in np.arange(chy1 + 1.5, chy2 - 1.0, 3.0):
            add(snap_grid(chx1, wg_y, wz), COLORS['building_glass_dark'])
    # Door facing pool
    for dz in np.arange(pool_z, pool_z + 3.0, PITCH):
        add(snap_grid(chx1, ch_cy, dz), COLORS['building_glass_bright'])
    # Roof
    ch_roof = filled_rect(chx1, chy1, chx2, chy2, height=2, z_offset=pool_z + ch_h)
    for pos in ch_roof: add(pos, COLORS['building_roof'])

    # ═══ PERIMETER FENCE ═══
    fence_margin = d_off + 2.0
    fence_posts = []  # corners of pool complex
    for fx in [px1 - fence_margin, px2 + fence_margin]:
        for fy in [py1 - fence_margin, py2 + fence_margin]:
            fence_posts.append((fx, fy))
    # Simple fence posts every 3m along edges
    for fx in np.arange(px1 - fence_margin, px2 + fence_margin, 3.0):
        for fy_side in [py1 - fence_margin, py2 + fence_margin]:
            for fz in np.arange(pool_z, pool_z + 2.0, PITCH):
                add(snap_grid(fx, fy_side, fz), COLORS['metal_gray'])
    for fy in np.arange(py1 - fence_margin, py2 + fence_margin, 3.0):
        for fx_side in [px1 - fence_margin, px2 + fence_margin]:
            for fz in np.arange(pool_z, pool_z + 2.0, PITCH):
                add(snap_grid(fx_side, fy, fz), COLORS['metal_gray'])

    # ═══ GROUND FILL ═══
    for x in np.arange(px1 - fence_margin - 4, px2 + fence_margin + ch_w, PITCH):
        for y in np.arange(py1 - fence_margin - 4, py2 + fence_margin + 4, PITCH):
            gx, gy = snap_grid(x, y)
            in_pool = (px1 <= gx <= px2 and py1 <= gy <= py2)
            in_changing = (chx1 - 0.5 <= gx <= chx2 + 0.5 and chy1 - 0.5 <= gy <= chy2 + 0.5)
            if not in_pool and not in_changing:
                add((gx, gy, pool_z), COLORS['ground_gray'])

    print(f"  TOTAL: {len(pos_color)} voxels")
    return list(pos_color.keys()), list(pos_color.values())


def build_spring():
    """Build Tianyi Spring (天一泉) — fountain jets, paving, inscription stone."""
    print("\n--- Building Tianyi Spring ---")

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    cx, cy = 0.0, -55.0; ground_z = 36.0
    basin_r = 4.0

    # ═══ SURROUNDING PAVING (12m x 12m square) ═══
    pave_half = 6.0
    for x in np.arange(cx - pave_half, cx + pave_half, PITCH * 0.6):
        for y in np.arange(cy - pave_half, cy + pave_half, PITCH * 0.6):
            gx, gy = snap_grid(x, y)
            if math.sqrt((gx-cx)**2 + (gy-cy)**2) > basin_r + 0.5:
                add((gx, gy, ground_z), COLORS['path_gray'])
    # Paving border
    for bx in np.arange(cx - pave_half, cx + pave_half, PITCH * 0.6):
        add(snap_grid(bx, cy - pave_half, ground_z + 0.5), COLORS['plaza_accent'])
        add(snap_grid(bx, cy + pave_half, ground_z + 0.5), COLORS['plaza_accent'])
    for by in np.arange(cy - pave_half, cy + pave_half, PITCH * 0.6):
        add(snap_grid(cx - pave_half, by, ground_z + 0.5), COLORS['plaza_accent'])
        add(snap_grid(cx + pave_half, by, ground_z + 0.5), COLORS['plaza_accent'])

    # ═══ CIRCULAR WATER BASIN ═══
    basin = filled_circle((cx, cy), basin_r, z=ground_z)
    for pos in basin: add(pos, COLORS['water_light'])

    # Basin edge (decorative alternating blocks)
    for angle in np.arange(0, 360, 5):
        rad = math.radians(angle)
        ex = cx + basin_r * math.cos(rad); ey = cy + basin_r * math.sin(rad)
        gx, gy = snap_grid(ex, ey)
        alt = (int(angle / 15) % 2 == 0)
        c = COLORS['plaza_stone_dark'] if alt else COLORS['plaza_stone_mid']
        add((gx, gy, ground_z + PITCH), c)

    # ═══ FIVE STONES (5 continents) ═══
    stone_positions = [
        (cx, cy, 1.0, 0.8),
        (cx - 1.5, cy - 1.0, 0.8, 0.6),
        (cx + 1.5, cy - 1.0, 0.7, 0.5),
        (cx - 0.5, cy + 1.5, 0.7, 0.5),
        (cx + 0.5, cy + 1.5, 0.6, 0.5),
    ]
    for sx, sy, sh, sw in stone_positions:
        for dx in np.arange(-sw/2, sw/2, PITCH * 0.6):
            for dy in np.arange(-sw/2, sw/2, PITCH * 0.6):
                gx, gy = snap_grid(sx + dx, sy + dy)
                for sz in np.arange(ground_z + 0.5, ground_z + 0.5 + sh, PITCH):
                    add((gx, gy, round(sz, 1)), COLORS['plaza_stone_dark'])

    # ═══ WATER JETS (5 fountain columns) ═══
    jet_positions = [
        (cx, cy),  # center
        (cx - 1.5, cy - 1.5), (cx + 1.5, cy - 1.5),
        (cx - 1.5, cy + 1.5), (cx + 1.5, cy + 1.5),
    ]
    for jx, jy in jet_positions:
        jet_h = 1.5 if (jx, jy) == (cx, cy) else 1.0
        for jh in np.arange(ground_z + 1.0, ground_z + 1.0 + jet_h, PITCH):
            add(snap_grid(jx, jy, jh), COLORS['water_light'])
        # Spray ring at top
        for spray_ang in np.arange(0, 360, 45):
            srad = math.radians(spray_ang)
            gx, gy = snap_grid(jx + 0.5 * math.cos(srad), jy + 0.5 * math.sin(srad))
            add((gx, gy, ground_z + 1.0 + jet_h), COLORS['water_light'])

    # ═══ INSCRIPTION STONE (south edge) ═══
    ins_x, ins_y = cx, cy - pave_half + 1.0
    for ih in np.arange(ground_z + 0.5, ground_z + 2.0, PITCH):
        for dx in np.arange(-0.5, 0.5, PITCH * 0.6):
            add(snap_grid(ins_x + dx, ins_y, ih), COLORS['plaza_stone_dark'])
    # Gold face (south-facing)
    add(snap_grid(ins_x, ins_y - 0.5, ground_z + 1.0), COLORS['sundial_gold'])
    add(snap_grid(ins_x, ins_y - 0.5, ground_z + 1.5), COLORS['sundial_gold'])

    # ═══ SEMICIRCULAR SEATING WALL (north half of basin) ═══
    wall_r = basin_r + 1.0
    for angle in np.arange(-90, 90, 5):
        rad = math.radians(angle)
        wx = cx + wall_r * math.cos(rad); wy = cy + wall_r * math.sin(rad)
        gx, gy = snap_grid(wx, wy)
        add((gx, gy, ground_z + 0.5), COLORS['building_wall'])
        add((gx, gy, ground_z + 1.0), COLORS['building_wall'])

    print(f"  TOTAL: {len(pos_color)} voxels")
    return list(pos_color.keys()), list(pos_color.values())


def build_shaw():
    """Build Shaw Auditorium — Henning Larsen's 3-ring elliptical landmark (2021).

    Real building: 3 stacked white elliptical rings (all same size) alternating
    with tall glass curtain walls. The rings are thick horizontal cantilevered
    floor slabs, NOT nested concentric walls of decreasing size.
    """
    print("\n--- Building Shaw Auditorium (Henning Larsen design) ---")

    cx, cy = 120.0, -190.0
    ground_z = 2.0
    a_outer, b_outer = 28.0, 18.0   # outer ellipse semi-axes (all 3 rings)
    a_glass, b_glass = 27.5, 17.5   # glass line (0.5m inside ring edge)
    a_inner, b_inner = 24.5, 14.5   # inner edge of ring (3.5m cantilever)
    a_core, b_core = 12.0, 7.0      # central auditorium core

    # Ring vertical positions (2.5m = 5 voxels thick)
    ring1_z = 5.5   # lower ring base
    ring2_z = 13.0  # middle ring base
    ring3_z = 20.0  # top ring / roof base
    ring_h = 2.5    # ring vertical thickness in meters

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    # Helper: add a tree
    def add_tree(tx, ty, tz):
        gx, gy = snap_grid(tx, ty)
        th = 4 + int(abs(hash(str((tx, ty))) % 1000) / 250)
        for hh in range(th):
            for tdx in [-0.5, 0.0, 0.5]:
                for tdy in [-0.5, 0.0, 0.5]:
                    if abs(tdx) + abs(tdy) <= 1.0:
                        add(snap_grid(gx + tdx, gy + tdy, tz + hh * PITCH), COLORS['trunk_brown'])
        for cl_r, cl_c in [(2.0, 'tree_dark'), (1.5, 'tree_green'), (1.0, 'tree_bright')]:
            layer = filled_circle((gx, gy), cl_r, z=tz + th * PITCH + (3.0 - cl_r))
            for pos in layer: add(pos, COLORS[cl_c])

    # Helper: fill an elliptical annular region (ring floor plate)
    def filled_elliptical_ring(cx, cy, a_out, b_out, a_in, b_in, z_base, height_m, color):
        """Fill the region between outer and inner ellipses, for height_m meters."""
        n_layers = int(height_m / PITCH)
        for x in np.arange(cx - a_out, cx + a_out + PITCH, PITCH * 0.8):
            for y in np.arange(cy - b_out, cy + b_out + PITCH, PITCH * 0.8):
                gx, gy = snap_grid(x, y)
                out_t = ((gx-cx)/a_out)**2 + ((gy-cy)/b_out)**2
                if out_t > 1.0: continue
                in_t = ((gx-cx)/a_in)**2 + ((gy-cy)/b_in)**2
                if in_t < 1.0: continue
                for lz in range(n_layers):
                    add((gx, gy, round(z_base + lz * PITCH, 1)), COLORS[color])

    # ═══ A: GROUND PLATFORM ═══
    platform = filled_ellipse((cx, cy), a_outer + 8, b_outer + 8, z_offset=ground_z)
    for pos in platform: add(pos, COLORS['path_gray'])

    # ═══ B: GROUND-LEVEL GLASS LOBBY (Z=2.0 → 5.5) ═══
    # Continuous glass curtain wall at ground level, set 0.5m inside ring edge
    lobby_z_top = ring1_z
    lobby_h = lobby_z_top - ground_z  # 3.5m
    for angle in np.arange(0, 360, 2):
        rad = math.radians(angle)
        gx = cx + a_glass * math.cos(rad)
        gy = cy + b_glass * math.sin(rad)
        gx, gy = snap_grid(gx, gy)
        # Skip entrance openings (glass replaced later by entrance treatment)
        is_entrance = False
        for ent_ang, ent_span in [(90, 14), (270, 12), (0, 10), (180, 10)]:
            half = ent_span / 2
            if abs((angle - ent_ang + 180) % 360 - 180) < half:
                is_entrance = True; break
        for lz in np.arange(ground_z, lobby_z_top, PITCH):
            if is_entrance:
                add((gx, gy, round(lz, 1)), COLORS['glass_facetted'])
            else:
                add((gx, gy, round(lz, 1)), COLORS['glass_facetted'])
                # Vertical mullion every 3 degrees
                if angle % 3 == 0:
                    add((gx, gy, round(lz, 1)), COLORS['champagne_bronze'])

    # ═══ C: RING 1 — LOWER RING (Z=5.5 → 8.0) ═══
    filled_elliptical_ring(cx, cy, a_outer, b_outer, a_inner, b_inner,
                           ring1_z, ring_h, 'shaw_white')
    # Champagne bronze trim on outer edge of ring
    for angle in np.arange(0, 360, 1):
        rad = math.radians(angle)
        rx = cx + (a_outer - 0.3) * math.cos(rad)
        ry = cy + (b_outer - 0.3) * math.sin(rad)
        for lz in np.arange(ring1_z, ring1_z + ring_h, PITCH):
            add(snap_grid(rx, ry, lz), COLORS['champagne_bronze'])
    # Bronze trim on inner edge
    for angle in np.arange(0, 360, 2):
        rad = math.radians(angle)
        rx = cx + (a_inner + 0.3) * math.cos(rad)
        ry = cy + (b_inner + 0.3) * math.sin(rad)
        for lz in np.arange(ring1_z, ring1_z + ring_h, PITCH):
            add(snap_grid(rx, ry, lz), COLORS['champagne_bronze'])

    # ═══ D: MIDDLE GLASS CURTAIN WALL (Z=8.0 → 13.0) ═══
    glass1_top = ring2_z  # 13.0
    for angle in np.arange(0, 360, 2):
        rad = math.radians(angle)
        gx = cx + a_glass * math.cos(rad)
        gy = cy + b_glass * math.sin(rad)
        gx, gy = snap_grid(gx, gy)
        for gz in np.arange(ring1_z + ring_h, glass1_top, PITCH):
            add((gx, gy, round(gz, 1)), COLORS['glass_facetted'])
            # Vertical mullions every 2.5 degrees for facetted effect
            if angle % 5 < 2:
                add((gx, gy, round(gz, 1)), COLORS['champagne_bronze'])

    # ═══ E: RING 2 — MIDDLE RING (Z=13.0 → 15.5) ═══
    filled_elliptical_ring(cx, cy, a_outer, b_outer, a_inner, b_inner,
                           ring2_z, ring_h, 'shaw_white')
    for angle in np.arange(0, 360, 1):
        rad = math.radians(angle)
        rx = cx + (a_outer - 0.3) * math.cos(rad)
        ry = cy + (b_outer - 0.3) * math.sin(rad)
        for lz in np.arange(ring2_z, ring2_z + ring_h, PITCH):
            add(snap_grid(rx, ry, lz), COLORS['champagne_bronze'])
    for angle in np.arange(0, 360, 2):
        rad = math.radians(angle)
        rx = cx + (a_inner + 0.3) * math.cos(rad)
        ry = cy + (b_inner + 0.3) * math.sin(rad)
        for lz in np.arange(ring2_z, ring2_z + ring_h, PITCH):
            add(snap_grid(rx, ry, lz), COLORS['champagne_bronze'])

    # ═══ F: UPPER GLASS CURTAIN WALL (Z=15.5 → 20.0) ═══
    glass2_top = ring3_z  # 20.0
    for angle in np.arange(0, 360, 2):
        rad = math.radians(angle)
        gx = cx + a_glass * math.cos(rad)
        gy = cy + b_glass * math.sin(rad)
        gx, gy = snap_grid(gx, gy)
        for gz in np.arange(ring2_z + ring_h, glass2_top, PITCH):
            add((gx, gy, round(gz, 1)), COLORS['glass_facetted'])
            if angle % 5 < 2:
                add((gx, gy, round(gz, 1)), COLORS['champagne_bronze'])

    # ═══ G: RING 3 — TOP RING / ROOF (Z=20.0 → 22.5) ═══
    filled_elliptical_ring(cx, cy, a_outer, b_outer, a_inner, b_inner,
                           ring3_z, ring_h, 'shaw_white')
    for angle in np.arange(0, 360, 1):
        rad = math.radians(angle)
        rx = cx + (a_outer - 0.3) * math.cos(rad)
        ry = cy + (b_outer - 0.3) * math.sin(rad)
        for lz in np.arange(ring3_z, ring3_z + ring_h, PITCH):
            add(snap_grid(rx, ry, lz), COLORS['champagne_bronze'])

    # ═══ H: CORE AUDITORIUM (Z=3.0 → 22.5) ═══
    # The auditorium is a solid elliptical volume inside the rings
    # Lower portion: bamboo-clad walls (visible from outside through glass)
    # Upper portion: white walls
    core_z_bottom = ground_z + 1.0
    core_z_top = ring3_z
    bamboo_top = ring2_z  # transition to white above ring 2
    for x in np.arange(cx - a_core, cx + a_core + PITCH, PITCH * 0.8):
        for y in np.arange(cy - b_core, cy + b_core + PITCH, PITCH * 0.8):
            gx, gy = snap_grid(x, y)
            if ((gx-cx)/a_core)**2 + ((gy-cy)/b_core)**2 > 1.0: continue
            for cz in np.arange(core_z_bottom, core_z_top, PITCH):
                if cz < bamboo_top:
                    add((gx, gy, round(cz, 1)), COLORS['bamboo_clad'])
                else:
                    add((gx, gy, round(cz, 1)), COLORS['shaw_white'])
    # South-facing glass wall (sea view) — from ring1 to ring3 level
    for x in np.arange(cx - a_core, cx + a_core + PITCH, PITCH * 0.8):
        for cz in np.arange(ring1_z, core_z_top - 2.0, PITCH):
            gx = round(x / PITCH) * PITCH
            gy = round(cy - b_core)
            if ((gx-cx)/a_core)**2 + ((gy-cy)/b_core)**2 <= 1.0:
                add((gx, gy, round(cz, 1)), COLORS['glass_facetted'])

    # ═══ I: FLY TOWER (Z=22.5 → 28.0, east side of core) ═══
    tower_x1 = cx + 1.5; tower_x2 = cx + a_core - 1.5
    tower_y1 = cy - b_core * 0.5; tower_y2 = cy + b_core * 0.5
    tower_z = ring3_z + ring_h  # 22.5
    tower_top = tower_z + 6.0
    # Main tower body — slightly tapered (wider at base, narrower at top)
    tower_mid_z = tower_z + 3.0
    tower_body = filled_rect(tower_x1, tower_y1, tower_x2, tower_y2,
                             height=int(3.0/PITCH), z_offset=tower_z)
    for pos in tower_body: add(pos, COLORS['shaw_white'])
    # Upper half slightly narrower
    tower_body2 = filled_rect(tower_x1 + 1.0, tower_y1 + 0.5,
                              tower_x2 - 1.0, tower_y2 - 0.5,
                              height=int(3.0/PITCH), z_offset=tower_mid_z)
    for pos in tower_body2: add(pos, COLORS['shaw_white'])
    # Vertical champagne bronze accent fins
    for vx in np.arange(tower_x1 + 1.0, tower_x2, 3.0):
        for vz in np.arange(tower_z, tower_top, PITCH * 0.5):
            add(snap_grid(vx, tower_y1, vz), COLORS['champagne_bronze'])
            add(snap_grid(vx, tower_y2, vz), COLORS['champagne_bronze'])
    # Metal equipment platform on top
    for px in np.arange(tower_x1 + 1.0, tower_x2 - 1.0 + PITCH, PITCH * 0.6):
        for py in np.arange(tower_y1 + 0.5, tower_y2 - 0.5 + PITCH, PITCH * 0.6):
            add(snap_grid(px, py, tower_top), COLORS['metal_gray'])
    # Small mechanical penthouse on platform
    mech = filled_rect(tower_x1 + 2.0, tower_y1 + 1.0, tower_x2 - 2.0, tower_y2 - 1.0,
                       height=4, z_offset=tower_top + 0.5)
    for pos in mech: add(pos, COLORS['building_roof_dark'])

    # ═══ J: FOUR ENTRANCES ═══
    # Each entrance: recessed glass portal + flanking columns + canopy lip
    entrances = [
        (90, 14, True),    # North — main, with Grand Foyer
        (270, 12, False),  # South — sea view
        (0, 10, False),    # East
        (180, 10, False),  # West
    ]
    for ent_ang, ent_span, is_main in entrances:
        half_span = ent_span / 2
        # Recessed glass portal (set back 2m from glass line)
        recess_a = a_glass - 2.0; recess_b = b_glass - 2.0
        for la in np.arange(ent_ang - half_span, ent_ang + half_span, 1):
            rad = math.radians(la)
            lx = cx + recess_a * math.cos(rad)
            ly = cy + recess_b * math.sin(rad)
            gx, gy = snap_grid(lx, ly)
            for lz in np.arange(ground_z + 0.5, ring1_z, PITCH):
                add((gx, gy, round(lz, 1)), COLORS['glass_facetted'])
            # Door frame (champagne bronze)
            if la == ent_ang:
                for lz in np.arange(ground_z + 0.5, ring1_z, PITCH):
                    add(snap_grid(lx, ly, lz), COLORS['champagne_bronze'])
        # Columns flanking the entrance (concrete pillars)
        for col_off in [-half_span, half_span]:
            col_rad = math.radians(ent_ang + col_off)
            col_x = cx + (a_glass - 1.0) * math.cos(col_rad)
            col_y = cy + (b_glass - 1.0) * math.sin(col_rad)
            for ch in np.arange(ground_z, ring1_z + ring_h + 1.0, PITCH):
                add(snap_grid(col_x, col_y, ch), COLORS['concrete_pillar'])
                add(snap_grid(col_x + 0.5, col_y + 0.5, ch), COLORS['concrete_pillar'])
        # Canopy lip (champagne bronze strip on ring 1 underside at entrance)
        for ca in np.arange(ent_ang - half_span - 2, ent_ang + half_span + 2, 1):
            rad = math.radians(ca)
            can_x = cx + (a_outer + 1.0) * math.cos(rad)
            can_y = cy + (b_outer + 1.0) * math.sin(rad)
            add(snap_grid(can_x, can_y, ring1_z - 0.5), COLORS['champagne_bronze'])
        # For main entrance: extended canopy projecting further
        if is_main:
            for ca in np.arange(ent_ang - half_span - 4, ent_ang + half_span + 4, 1):
                rad = math.radians(ca)
                can_x = cx + (a_outer + 5.0) * math.cos(rad)
                can_y = cy + (b_outer + 5.0) * math.sin(rad)
                for cz in [ring1_z - 0.5, ring1_z]:
                    add(snap_grid(can_x, can_y, cz), COLORS['building_roof_dark'])

    # ═══ K: GRAND FOYER (stepped seating inside north entrance) ═══
    # 3 wide steps rising from ground level, visible through north glass
    foyer_cx = cx; foyer_cy = cy + b_glass - 3.0  # just inside north glass
    for step in range(3):
        step_z = ground_z + step * 0.5
        step_y1 = foyer_cy + step * 2.0
        step_y2 = step_y1 + 2.5
        step_x1 = foyer_cx - 8.0; step_x2 = foyer_cx + 8.0
        step_area = filled_rect(step_x1, step_y1, step_x2, step_y2,
                                height=1, z_offset=step_z)
        for pos in step_area: add(pos, COLORS['path_gray'])
        # Step riser face (white)
        for sx in np.arange(step_x1, step_x2 + PITCH, PITCH * 0.6):
            add(snap_grid(sx, step_y1, step_z + 0.5), COLORS['shaw_white'])

    # ═══ L: ROOF — PV PANELS & SKYLIGHTS ═══
    roof_z = ring3_z + ring_h  # 22.5
    # Skylight strips (2 curved bands following the ellipse)
    for sk_i, sk_scale in enumerate([0.7, 0.55]):
        sk_a = a_outer * sk_scale; sk_b = b_outer * sk_scale
        for angle in np.arange(0, 360, 2):
            rad = math.radians(angle)
            sx = cx + sk_a * math.cos(rad)
            sy = cy + sk_b * math.sin(rad)
            add(snap_grid(sx, sy, roof_z), COLORS['skylight_glass'])
    # PV panel zones — rectangular patches on east and west sides of roof
    for pv_sign in [-1, 1]:
        pv_cx_offset = pv_sign * 8.0
        for pv_i in range(3):
            pv_x = cx + pv_cx_offset + pv_i * 2.5
            for pv_y in np.arange(cy - 12.0, cy + 12.0, 4.0):
                pv_patch = filled_rect(pv_x, pv_y, pv_x + 3.5, pv_y + 3.5,
                                       height=1, z_offset=roof_z + 0.5)
                for pos in pv_patch: add(pos, COLORS['pv_panel'])
                # Metal frame around each panel
                for fx in np.arange(pv_x, pv_x + 4.0, PITCH * 0.6):
                    add(snap_grid(fx, pv_y, roof_z + 0.5), COLORS['metal_gray'])
                    add(snap_grid(fx, pv_y + 3.5, roof_z + 0.5), COLORS['metal_gray'])
                for fy in np.arange(pv_y, pv_y + 4.0, PITCH * 0.6):
                    add(snap_grid(pv_x, fy, roof_z + 0.5), COLORS['metal_gray'])
                    add(snap_grid(pv_x + 3.5, fy, roof_z + 0.5), COLORS['metal_gray'])

    # ═══ M: ENTRANCE PLAZA & LANDSCAPE ═══
    # North entrance plaza
    plz_cy = cy + b_outer + 6.0
    plz_cx = cx
    plz_w, plz_d = 26.0, 20.0
    plz_x1, plz_x2 = plz_cx - plz_w/2, plz_cx + plz_w/2
    plz_y1, plz_y2 = plz_cy, plz_cy + plz_d
    plaza = filled_rect(plz_x1, plz_y1, plz_x2, plz_y2, height=2, z_offset=ground_z)
    for pos in plaza: add(pos, COLORS['path_gray'])
    # Plaza border accent
    for bx in np.arange(plz_x1, plz_x2 + PITCH, PITCH * 0.6):
        add(snap_grid(bx, plz_y1, ground_z + 0.5), COLORS['plaza_accent'])
        add(snap_grid(bx, plz_y2, ground_z + 0.5), COLORS['plaza_accent'])
    # Champagne bronze bollard lights
    for bl_x in np.arange(plz_x1 + 2.0, plz_x2 - 1.0, 5.0):
        add(snap_grid(bl_x, plz_y1 + 0.5, ground_z + 0.5), COLORS['champagne_bronze'])
        add(snap_grid(bl_x, plz_y1 + 0.5, ground_z + 1.0), COLORS['sundial_gold'])
    # Trees in 2 rows
    for tx in [plz_cx - plz_w/4, plz_cx + plz_w/4]:
        for ty in [plz_y1 + 5, plz_y1 + 15]:
            add_tree(tx, ty, ground_z + 1.0)
    # Benches
    for bx, by in [(plz_cx - 6, plz_y1 + 3), (plz_cx + 6, plz_y1 + 3)]:
        for bw in np.arange(0, 3.0, PITCH * 0.6):
            add(snap_grid(bx + bw, by, ground_z + 0.5), COLORS['bench_brown'])

    # Curved access road from north (S-curve approach)
    road_w = 4.0
    for rt in np.arange(0, 1.0, 0.02):
        rr = 38.0
        ra = 55 + rt * 35
        rrad = math.radians(ra)
        rx = cx + rr * math.cos(rrad)
        ry = cy + rr * math.sin(rrad)
        for dw in np.arange(-road_w/2, road_w/2, PITCH * 0.6):
            gx, gy = snap_grid(rx + dw * math.sin(rrad), ry - dw * math.cos(rrad))
            add((gx, gy, ground_z), COLORS['path_dark'])

    # South side grass mounds
    for angle in np.arange(180, 360, 5):
        rad = math.radians(angle)
        gx = cx + (a_outer + 9) * math.cos(rad)
        gy = cy + (b_outer + 9) * math.sin(rad)
        gx, gy = snap_grid(gx, gy)
        add((gx, gy, ground_z + 0.5), COLORS['hillside_grass'])

    # South viewing terrace (paved area overlooking sea)
    for sx in np.arange(cx - 10, cx + 10 + PITCH, PITCH * 0.6):
        sy = cy - b_outer - 3.0
        add(snap_grid(sx, sy, ground_z), COLORS['path_gray'])

    # ═══ N: PERIMETER TREES ═══
    for tree_ang in [30, 60, 120, 150, 210, 240, 300, 330]:
        rad = math.radians(tree_ang)
        tx = cx + (a_outer + 12) * math.cos(rad)
        ty = cy + (b_outer + 12) * math.sin(rad)
        add_tree(tx, ty, ground_z + 1.0)

    print(f"  TOTAL: {len(pos_color)} voxels")
    return list(pos_color.keys()), list(pos_color.values())


def build_north_gate():
    """Build North Gate (北閘) — stone pillars, roundabout, bus terminus, University Road.

    The real North Gate is a transportation hub at the end of University Road,
    NOT a ceremonial arch with a red wall. Key features:
    - Stone pillars with university name/emblem (the iconic photo spot)
    - Roundabout connecting University Road to campus
    - Linear bus terminus with covered waiting area
    - Taxi stand, security post, landscaping
    """
    print("\n--- Building North Gate ---")

    cx, cy = 0.0, 60.0
    ground_z = 38.0

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    def add_tree(tx, ty, tz):
        gx, gy = snap_grid(tx, ty)
        th = 4 + int(abs(hash(str((tx, ty))) % 1000) / 250)
        for hh in range(th):
            for tdx in [-0.5, 0.0, 0.5]:
                for tdy in [-0.5, 0.0, 0.5]:
                    if abs(tdx) + abs(tdy) <= 1.0:
                        add(snap_grid(gx + tdx, gy + tdy, tz + hh * PITCH), COLORS['trunk_brown'])
        for cl_r, cl_c in [(2.0, 'tree_dark'), (1.5, 'tree_green'), (1.0, 'tree_bright')]:
            layer = filled_circle((gx, gy), cl_r, z=tz + th * PITCH + (3.0 - cl_r))
            for pos in layer: add(pos, COLORS[cl_c])

    # ═══ A: UNIVERSITY ROAD (approach from Clear Water Bay Road) ═══
    # 8m wide dual carriageway from north
    road_y1, road_y2 = 85.0, 125.0
    road_w = 8.0
    for ry in np.arange(road_y1, road_y2 + PITCH, PITCH * 0.6):
        for rx in np.arange(cx - road_w/2, cx + road_w/2 + PITCH, PITCH * 0.6):
            add(snap_grid(rx, ry, ground_z), COLORS['road_asphalt'])
    # Center divider line
    for ry in np.arange(road_y1, road_y2 + PITCH, PITCH * 0.6):
        add(snap_grid(cx, ry, ground_z + 0.5), COLORS['plaza_accent'])
    # Sidewalks (2m each side)
    for ry in np.arange(road_y1, road_y2 + PITCH, PITCH * 0.6):
        for rx in np.arange(cx - road_w/2 - 2, cx - road_w/2 + PITCH, PITCH * 0.6):
            add(snap_grid(rx, ry, ground_z + 0.5), COLORS['path_gray'])
        for rx in np.arange(cx + road_w/2, cx + road_w/2 + 2 + PITCH, PITCH * 0.6):
            add(snap_grid(rx, ry, ground_z + 0.5), COLORS['path_gray'])

    # ═══ B: ROUNDABOUT ═══
    rb_cx, rb_cy = cx, cy + 15.0  # center at (0, 75)
    rb_outer_r = 14.0; rb_inner_r = 7.0
    rb_island_r = 6.5
    # Outer ring (road surface)
    for angle in np.arange(0, 360, 1):
        rad = math.radians(angle)
        for r in np.arange(rb_inner_r, rb_outer_r + PITCH, PITCH * 0.7):
            rx = rb_cx + r * math.cos(rad)
            ry = rb_cy + r * math.sin(rad)
            gx, gy = snap_grid(rx, ry)
            add((gx, gy, ground_z), COLORS['road_asphalt'])
    # Central island
    rb_island = filled_circle((rb_cx, rb_cy), rb_island_r, z=ground_z + 0.5)
    for pos in rb_island: add(pos, COLORS['hillside_grass'])
    # Inner curb
    for angle in np.arange(0, 360, 2):
        rad = math.radians(angle)
        rx = rb_cx + rb_inner_r * math.cos(rad)
        ry = rb_cy + rb_inner_r * math.sin(rad)
        add(snap_grid(rx, ry, ground_z + 0.5), COLORS['plaza_stone_dark'])
    # Outer curb
    for angle in np.arange(0, 360, 2):
        rad = math.radians(angle)
        rx = rb_cx + rb_outer_r * math.cos(rad)
        ry = rb_cy + rb_outer_r * math.sin(rad)
        add(snap_grid(rx, ry, ground_z + 0.5), COLORS['plaza_stone_dark'])
    # Central flower bed
    rb_flower = filled_circle((rb_cx, rb_cy), 2.5, z=ground_z + 1.0)
    for pos in rb_flower: add(pos, COLORS['red_bird'])
    # 4 trees on island
    for tree_ang in [45, 135, 225, 315]:
        rad = math.radians(tree_ang)
        tx = rb_cx + 4.5 * math.cos(rad)
        ty = rb_cy + 4.5 * math.sin(rad)
        add_tree(tx, ty, ground_z + 0.5)

    # ═══ C: STONE PILLARS GATEWAY ═══
    # Two large stone pillars flanking the entrance at Y=60
    for px in [-6.0, 6.0]:
        pillar_h = 8.0  # 8m tall
        # Pillar body: 2m x 2m cross-section (4x4 voxels)
        for pdx in np.arange(px - 1.0, px + 1.0, PITCH * 0.8):
            for pdy in np.arange(cy - 1.0, cy + 1.0, PITCH * 0.8):
                for pz in np.arange(ground_z, ground_z + pillar_h, PITCH):
                    add(snap_grid(pdx, pdy, pz), COLORS['stone_pillar'])
        # Gold cap on top
        for pdx in np.arange(px - 1.0, px + 1.0, PITCH * 0.8):
            for pdy in np.arange(cy - 1.0, cy + 1.0, PITCH * 0.8):
                add(snap_grid(pdx, pdy, ground_z + pillar_h), COLORS['gate_gold'])
        # Gold university emblem/name blocks on pillar face (south-facing)
        emblem_z = ground_z + 3.5
        for ez in np.arange(emblem_z, emblem_z + 1.5, PITCH):
            add(snap_grid(px, cy + 0.5, ez), COLORS['gate_gold'])
            add(snap_grid(px + 0.5, cy + 0.5, ez), COLORS['gate_gold'])
        # Secondary name block
        for ez in np.arange(emblem_z + 2.0, emblem_z + 2.5, PITCH):
            add(snap_grid(px, cy + 0.5, ez), COLORS['gate_gold'])
    # Paved entry path between pillars
    entry_path = filled_rect(-4.0, cy - 2.0, 4.0, cy + 2.0, height=2, z_offset=ground_z)
    for pos in entry_path: add(pos, COLORS['path_gray'])

    # ═══ D: BUS TERMINUS (west side, linear parallel bays) ═══
    bus_start_x = -30.0; bus_start_y = 80.0
    n_bays = 5
    bay_w = 3.0; bay_len = 18.0
    bay_spacing = 4.5
    for bi in range(n_bays):
        bx1 = bus_start_x + bi * bay_spacing
        bx2 = bx1 + bay_w
        by1 = bus_start_y; by2 = bus_start_y + bay_len
        # Bay road surface
        bay = filled_rect(bx1, by1, bx2, by2, height=2, z_offset=ground_z)
        for pos in bay: add(pos, COLORS['road_asphalt'])
        # Curb on both sides
        for by in np.arange(by1, by2 + PITCH, PITCH * 0.6):
            add(snap_grid(bx1 - 0.5, by, ground_z + 1.0), COLORS['plaza_stone_dark'])
            add(snap_grid(bx2 + 0.5, by, ground_z + 1.0), COLORS['plaza_stone_dark'])
    # Large covered waiting area (canopy over passenger zone)
    canopy_x1 = bus_start_x - 1.0
    canopy_x2 = bus_start_x + (n_bays - 1) * bay_spacing + bay_w + 1.0
    canopy_y1 = bus_start_y + bay_len - 4.0
    canopy_y2 = bus_start_y + bay_len + 3.0
    canopy_z = ground_z + 3.0
    # Canopy roof
    for cx2 in np.arange(canopy_x1, canopy_x2 + PITCH, PITCH * 0.6):
        for cy2 in np.arange(canopy_y1, canopy_y2 + PITCH, PITCH * 0.6):
            add(snap_grid(cx2, cy2, canopy_z), COLORS['canopy_white'])
            add(snap_grid(cx2, cy2, canopy_z + 0.5), COLORS['canopy_white'])
    # Canopy support columns
    for col_x in np.arange(canopy_x1 + 1.0, canopy_x2, 3.5):
        for col_y in [canopy_y1 + 0.5, canopy_y2 - 0.5]:
            for ch in np.arange(ground_z + 0.5, canopy_z, PITCH):
                add(snap_grid(col_x, col_y, ch), COLORS['concrete_pillar'])
    # Passenger waiting platform (raised)
    platform_area = filled_rect(canopy_x1 + 0.5, canopy_y1 + 0.5,
                                canopy_x2 - 0.5, canopy_y2 - 0.5,
                                height=1, z_offset=ground_z + 0.5)
    for pos in platform_area: add(pos, COLORS['path_gray'])

    # ═══ E: TAXI STAND (east side of roundabout) ═══
    taxi_x1, taxi_x2 = 14.0, 19.0
    taxi_y1, taxi_y2 = 58.0, 70.0
    taxi_area = filled_rect(taxi_x1, taxi_y1, taxi_x2, taxi_y2, height=2, z_offset=ground_z)
    for pos in taxi_area: add(pos, COLORS['road_asphalt'])
    # Curb
    for ty in np.arange(taxi_y1, taxi_y2 + PITCH, PITCH * 0.6):
        add(snap_grid(taxi_x1 - 0.5, ty, ground_z + 1.0), COLORS['plaza_stone_dark'])
    # Taxi waiting shelter (small canopy)
    taxi_shelter_y = taxi_y2 - 3.0
    for tsx in np.arange(taxi_x1, taxi_x2 + PITCH, PITCH * 0.6):
        add(snap_grid(tsx, taxi_shelter_y, ground_z + 3.0), COLORS['canopy_white'])
    for tsc in [taxi_x1 + 0.5, taxi_x2 - 0.5]:
        for tsh in np.arange(ground_z + 0.5, ground_z + 3.0, PITCH):
            add(snap_grid(tsc, taxi_shelter_y, tsh), COLORS['concrete_pillar'])
    # Gold TAXI marker
    for tmx in np.arange(taxi_x1 + 1.0, taxi_x2 - 1.0, PITCH * 0.6):
        add(snap_grid(tmx, (taxi_y1 + taxi_y2)/2, ground_z + 0.5), COLORS['sundial_gold'])

    # ═══ F: SECURITY POST ═══
    sec_cx, sec_cy = 0.0, cy - 4.0  # just south of pillars
    sec_w, sec_d, sec_h = 5.0, 3.0, 3.0
    sx1, sx2 = sec_cx - sec_w/2, sec_cx + sec_w/2
    sy1, sy2 = sec_cy - sec_d/2, sec_cy + sec_d/2
    # Walls
    for sz in np.arange(ground_z, ground_z + sec_h, PITCH):
        for sx in np.arange(sx1, sx2 + PITCH, PITCH * 0.6):
            add(snap_grid(sx, sy1, sz), COLORS['building_wall'])
            add(snap_grid(sx, sy2, sz), COLORS['building_wall'])
        for sy in np.arange(sy1, sy2 + PITCH, PITCH * 0.6):
            add(snap_grid(sx1, sy, sz), COLORS['building_wall'])
            add(snap_grid(sx2, sy, sz), COLORS['building_wall'])
    # Windows (north and south faces)
    for sz in np.arange(ground_z + 1.0, ground_z + sec_h - 0.5, PITCH):
        add(snap_grid(sec_cx - 1.0, sy2, sz), COLORS['building_glass'])
        add(snap_grid(sec_cx + 1.0, sy2, sz), COLORS['building_glass'])
    # Roof
    sec_roof = filled_rect(sx1, sy1, sx2, sy2, height=1, z_offset=ground_z + sec_h)
    for pos in sec_roof: add(pos, COLORS['building_roof'])
    # Gate barrier (horizontal bar across entrance)
    barrier_y = cy - 2.0
    for bx2 in np.arange(-4.0, 4.0, PITCH * 0.6):
        add(snap_grid(bx2, barrier_y, ground_z + 1.5), COLORS['metal_gray'])
    # Barrier support post
    for bh in np.arange(ground_z, ground_z + 1.5, PITCH):
        add(snap_grid(-4.5, barrier_y, bh), COLORS['metal_gray'])
        add(snap_grid(4.5, barrier_y, bh), COLORS['metal_gray'])

    # ═══ G: COVERED WALKWAY (bus terminus → pillars → campus) ═══
    # Walkway from bus stop to gateway area
    walk_start_y = canopy_y2
    walk_end_y = cy + 3.0
    walk_x = bus_start_x + (n_bays - 1) * bay_spacing / 2  # middle of bus area
    walk_w = 3.0
    walkway = filled_rect(walk_x - walk_w/2, walk_end_y,
                          walk_x + walk_w/2, walk_start_y,
                          height=2, z_offset=ground_z)
    for pos in walkway: add(pos, COLORS['path_gray'])
    # Covered canopy over walkway
    for wx2 in np.arange(walk_x - walk_w/2, walk_x + walk_w/2 + PITCH, PITCH * 0.6):
        for wy2 in np.arange(walk_end_y, walk_start_y + PITCH, PITCH * 0.6):
            add(snap_grid(wx2, wy2, ground_z + 3.0), COLORS['canopy_white'])
    # Walkway support columns (every 4m)
    for wc_y in np.arange(walk_end_y + 1.0, walk_start_y, 4.0):
        for wc_x in [walk_x - 1.0, walk_x + 1.0]:
            for wh in np.arange(ground_z + 0.5, ground_z + 3.0, PITCH):
                add(snap_grid(wc_x, wc_y, wh), COLORS['concrete_pillar'])

    # ═══ H: CAMPUS APPROACH ROAD (south from roundabout) ═══
    # Road from roundabout southward into campus
    campus_road_y1 = cy - 2.0; campus_road_y2 = rb_cy - rb_outer_r
    for ry in np.arange(campus_road_y1, campus_road_y2 + PITCH, PITCH * 0.6):
        for rx in np.arange(cx - road_w/2, cx + road_w/2 + PITCH, PITCH * 0.6):
            add(snap_grid(rx, ry, ground_z), COLORS['road_asphalt'])

    # ═══ I: LANDSCAPING & TREES ═══
    # Trees flanking University Road
    for tree_y in [88, 95, 102, 110, 118]:
        add_tree(cx - 7, tree_y, ground_z + 1.0)
        add_tree(cx + 7, tree_y, ground_z + 1.0)
    # Trees around roundabout perimeter
    for tree_ang in [0, 30, 60, 120, 150, 180, 210, 240, 300, 330]:
        rad = math.radians(tree_ang)
        tx = rb_cx + (rb_outer_r + 3) * math.cos(rad)
        ty = rb_cy + (rb_outer_r + 3) * math.sin(rad)
        add_tree(tx, ty, ground_z + 1.0)
    # Trees near stone pillars
    for tree_sign in [-1, 1]:
        add_tree(cx + tree_sign * 10, cy - 5, ground_z + 1.0)
        add_tree(cx + tree_sign * 12, cy + 8, ground_z + 1.0)

    # ═══ J: GROUND FILL ═══
    fill_x1, fill_x2 = cx - 35, cx + 35
    fill_y1, fill_y2 = cy - 10, cy + 65
    for x in np.arange(fill_x1, fill_x2 + PITCH, PITCH):
        for y in np.arange(fill_y1, fill_y2 + PITCH, PITCH):
            gx, gy = snap_grid(x, y)
            # Skip roundabout area
            if ((gx - rb_cx)**2 + (gy - rb_cy)**2) < (rb_outer_r + 1.5)**2: continue
            # Skip University Road
            if road_y1 <= gy <= road_y2 and cx - road_w/2 - 2 <= gx <= cx + road_w/2 + 2: continue
            # Skip bus terminus area
            if bus_start_x - 2 <= gx <= canopy_x2 + 2 and bus_start_y - 1 <= gy <= canopy_y2 + 1: continue
            # Skip taxi area
            if taxi_x1 - 1 <= gx <= taxi_x2 + 1 and taxi_y1 - 1 <= gy <= taxi_y2 + 1: continue
            # Skip campus road
            if campus_road_y1 <= gy <= campus_road_y2 and cx - road_w/2 - 1 <= gx <= cx + road_w/2 + 1: continue
            add((gx, gy, ground_z), COLORS['ground_gray'])

    print(f"  TOTAL: {len(pos_color)} voxels")
    return list(pos_color.keys()), list(pos_color.values())


def build_garden():
    """Build Chinese Garden — irregular pond, multiple pavilions, moon gate, lotus, bamboo."""
    print("\n--- Building Chinese Garden ---")

    pos_color = {}
    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    cx, cy = -65.0, 35.0; ground_z = 38.0

    # ═══ IRREGULAR POND (composite of 3 ellipses) ═══
    pond_shapes = [
        (cx, cy, 10.0, 6.0),           # Main pond
        (cx - 7.0, cy - 2.0, 4.0, 5.0),  # West lobe
        (cx + 5.0, cy + 3.0, 3.0, 2.0),  # East channel
    ]
    pond_set = set()
    for pcx, pcy, pa, pb in pond_shapes:
        pond_fill = filled_ellipse((pcx, pcy), pa, pb, z_offset=ground_z - 1.0)
        for pos in pond_fill: pond_set.add(pos)
    for pos in pond_set: add(pos, COLORS['water_light'])

    # Pond edge stones (on perimeter of all pond sections)
    for pcx, pcy, pa, pb in pond_shapes:
        for angle in np.arange(0, 360, 3):
            rad = math.radians(angle)
            rx = pcx + pa * math.cos(rad); ry = pcy + pb * math.sin(rad)
            gx, gy = snap_grid(rx, ry)
            for sz in [ground_z, ground_z + PITCH]:
                add((gx, gy, round(sz, 1)), COLORS['plaza_stone_dark'])

    # ═══ MAIN PAVILION (double-tier roof, gold ridge ornament) ═══
    pav_cx, pav_cy = cx + 8, cy + 5
    pav_base = filled_circle((pav_cx, pav_cy), 3.0, z=ground_z + 1.0)
    for pos in pav_base: add(pos, COLORS['path_gray'])
    for col_ang in [0, 90, 180, 270]:
        rad = math.radians(col_ang)
        col_x = pav_cx + 2.0 * math.cos(rad); col_y = pav_cy + 2.0 * math.sin(rad)
        for ch in np.arange(ground_z + 1.5, ground_z + 5.0, PITCH):
            add(snap_grid(col_x, col_y, ch), COLORS['concrete_pillar'])
    # Lower roof tier
    for x in np.arange(pav_cx - 3.5, pav_cx + 3.5, PITCH * 0.6):
        for y in np.arange(pav_cy - 3.5, pav_cy + 3.5, PITCH * 0.6):
            gx, gy = snap_grid(x, y); dist = math.sqrt((gx-pav_cx)**2 + (gy-pav_cy)**2)
            if dist > 3.5: continue
            roof_h = ground_z + 5.0 + (1.0 - dist / 3.5) * 1.5
            add((gx, gy, round(roof_h, 1)), COLORS['building_roof'])
    # Upper (smaller) roof tier
    for x in np.arange(pav_cx - 2.0, pav_cx + 2.0, PITCH * 0.6):
        for y in np.arange(pav_cy - 2.0, pav_cy + 2.0, PITCH * 0.6):
            gx, gy = snap_grid(x, y); dist = math.sqrt((gx-pav_cx)**2 + (gy-pav_cy)**2)
            if dist > 2.0: continue
            roof_h2 = ground_z + 6.0 + (1.0 - dist / 2.0) * 1.0
            add((gx, gy, round(roof_h2, 1)), COLORS['building_roof_dark'])
    # Gold ridge ornament
    for rx in np.arange(pav_cx - 1.0, pav_cx + 1.0, PITCH * 0.6):
        add(snap_grid(rx, pav_cy, ground_z + 7.0), COLORS['sundial_gold'])

    # ═══ MEDITATION PAVILION (west side) ═══
    mp_cx, mp_cy = cx - 12, cy + 2
    mp_base = filled_circle((mp_cx, mp_cy), 2.0, z=ground_z + 1.0)
    for pos in mp_base: add(pos, COLORS['path_gray'])
    for col_ang in [0, 120, 240]:
        rad = math.radians(col_ang)
        col_x = mp_cx + 1.5 * math.cos(rad); col_y = mp_cy + 1.5 * math.sin(rad)
        for ch in np.arange(ground_z + 1.5, ground_z + 4.0, PITCH):
            add(snap_grid(col_x, col_y, ch), COLORS['concrete_pillar'])
    for x in np.arange(mp_cx - 3.5, mp_cx + 3.5, PITCH * 0.6):
        for y in np.arange(mp_cy - 3.5, mp_cy + 3.5, PITCH * 0.6):
            gx, gy = snap_grid(x, y); dist = math.sqrt((gx-mp_cx)**2 + (gy-mp_cy)**2)
            if dist > 3.5: continue
            rh = ground_z + 4.0 + (1.0 - dist / 3.5) * 1.0
            add((gx, gy, round(rh, 1)), COLORS['building_roof'])

    # ═══ VIEWING PLATFORM (south edge) ═══
    vp_cx, vp_cy = cx + 2, cy - 8
    vp_fill = filled_rect(vp_cx - 1.5, vp_cy - 1.5, vp_cx + 1.5, vp_cy + 1.5, height=2, z_offset=ground_z)
    for pos in vp_fill: add(pos, COLORS['path_gray'])

    # ═══ MOON GATE (east approach) ═══
    mg_cx, mg_cy = cx + 14, cy - 1
    mg_radius = 2.0
    # Wall segment
    wall_w, wall_h = 5.0, 4.0
    for x in np.arange(mg_cx - wall_w/2, mg_cx + wall_w/2, PITCH * 0.6):
        for hz in np.arange(ground_z, ground_z + wall_h, PITCH):
            gx = round(x / PITCH) * PITCH
            dist_from_center = abs(gx - mg_cx)
            # Create the circular opening
            if dist_from_center < mg_radius - 0.5:
                # Inside the opening - check if within the circle
                vert_dist = abs(hz - (ground_z + wall_h/2))
                if dist_from_center**2 + vert_dist**2 < mg_radius**2:
                    continue  # skip - this is the opening
            add((gx, mg_cy, round(hz, 1)), COLORS['plaza_stone_dark'])
            add((gx, mg_cy + 0.5, round(hz, 1)), COLORS['plaza_stone_dark'])

    # ═══ LOTUS PLANTS (on pond surface) ═══
    for li in range(10):
        lx = cx + (hash(f"lotus{li}") % 18 - 9)
        ly = cy + (hash(f"lotusy{li}") % 12 - 6)
        gx, gy = snap_grid(lx, ly)
        # Green pad
        add((gx, gy, ground_z - 0.5), COLORS['tree_green'])
        if li % 3 == 0:
            # Pink/gold flower
            add((gx, gy, ground_z), COLORS['red_bird_bright'] if li < 5 else COLORS['sundial_gold'])

    # ═══ WINDING PATHS ═══
    path_waypoints = [
        (cx + 14, cy - 1),   # moon gate
        (cx + 8, cy + 5),     # main pavilion
        (cx - 4, cy - 2),     # bridge area
        (cx - 12, cy + 2),    # meditation pavilion
        (cx + 2, cy - 8),     # viewing platform
    ]
    for wi in range(len(path_waypoints) - 1):
        x1, y1 = path_waypoints[wi]; x2, y2 = path_waypoints[wi + 1]
        steps = 20
        for st in range(steps):
            t = st / steps
            px = x1 + (x2 - x1) * t; py = y1 + (y2 - y1) * t
            for pw in np.arange(-0.8, 0.8, PITCH * 0.6):
                gx, gy = snap_grid(px + pw * 0.3, py + pw * 0.3)
                add((gx, gy, ground_z), COLORS['path_gray'])

    # ═══ CURVED BRIDGE ═══
    for bx in np.arange(cx - 4, cx + 4, PITCH * 0.6):
        by = cy - 6 - 1
        for bw in np.arange(-0.8, 0.8, PITCH * 0.6):
            gx, gy = snap_grid(bx, by + bw)
            arch_h = 0.5 * (1 - ((gx - cx) / 5)**2)
            add((gx, gy, round(ground_z + abs(arch_h), 1)), COLORS['path_dark'])

    # ═══ TAIHU SCHOLAR ROCKS (clustered formations) ═══
    rock_clusters = [
        (cx + 3, cy - 5), (cx + 6, cy + 8), (cx - 5, cy - 3),
        (cx - 8, cy + 6), (cx - 2, cy + 2),
    ]
    for ri, (rcx, rcy) in enumerate(rock_clusters):
        for _ in range(5 + ri % 3):
            rx = rcx + (hash(f"rock{ri}{_}") % 300 - 150) / 100.0
            ry = rcy + (hash(f"rocky{ri}{_}") % 300 - 150) / 100.0
            gx, gy = snap_grid(rx, ry)
            rock_h = 2 + int(abs(hash(f"rockh{ri}{_}")) % 3)
            for rh in np.arange(ground_z, ground_z + rock_h * PITCH, PITCH):
                add((gx, gy, round(rh, 1)), COLORS['hillside_rock'] if ri < 3 else COLORS['coast_rock'])

    # ═══ BAMBOO GROVE (north boundary) ═══
    for bi in range(18):
        bx = cx - 10 + (hash(f"bamboo{bi}") % 2000) / 100.0
        by = cy + 10 + (hash(f"bambooy{bi}") % 600) / 100.0
        gx, gy = snap_grid(bx, by)
        for bh in np.arange(ground_z, ground_z + 5.0, PITCH):
            add((gx, gy, round(bh, 1)), COLORS['tree_green'])
        # Leaf cluster at top
        for lx in np.arange(gx - 1.0, gx + 1.0, PITCH * 0.6):
            for ly in np.arange(gy - 1.0, gy + 1.0, PITCH * 0.6):
                glx, gly = snap_grid(lx, ly)
                if abs(glx - gx) + abs(gly - gy) <= 2.0:
                    add((glx, gly, ground_z + 5.0), COLORS['tree_bright'])

    print(f"  TOTAL: {len(pos_color)} voxels")
    return list(pos_color.keys()), list(pos_color.values())


def build_campus():
    """Build the complete unified HKUST campus — all landmarks together.

    Combines: plaza, academic arc, atrium, track, coastline, and hillside terrain.
    All landmarks share a common coordinate system.
    """
    print("\n" + "="*60)
    print("  BUILDING COMPLETE HKUST CAMPUS")
    print("="*60)

    all_positions, all_colors = [], []
    pos_color = {}

    def add(p, c):
        if p not in pos_color: pos_color[p] = c

    def merge_from_builder(build_fn, name):
        t0 = time.time()
        positions, colors = build_fn()
        for pos, color in zip(positions, colors):
            add(pos, color)
        print(f"  Merged {name}: {len(positions):,} voxels [{time.time()-t0:.0f}s]")

    # Build each component: terrain/coastline first (background),
    # then buildings overlay (first-writer-wins = buildings on top)
    merge_from_builder(build_coastline, "Coastline")       # terrain background
    merge_from_builder(build_shaw, "Shaw Auditorium")      # southern edge
    merge_from_builder(build_track, "Track")               # sports lower terrace
    merge_from_builder(build_pool, "Swimming Pools")       # pool near halls
    merge_from_builder(build_halls, "Student Halls")       # residence towers
    merge_from_builder(build_spring, "Tianyi Spring")      # fountain
    merge_from_builder(build_garden, "Chinese Garden")     # garden NW
    merge_from_builder(build_plaza, "Plaza")               # outdoor plaza
    merge_from_builder(build_atrium, "Atrium")             # entrance hall
    merge_from_builder(build_cyt, "CYT Building")          # south of arc
    merge_from_builder(build_lsk, "LSK Business Building") # west wing
    merge_from_builder(build_library, "Library")           # east wing
    merge_from_builder(build_academic, "Academic Arc")     # main building
    merge_from_builder(build_north_gate, "North Gate")     # entrance gate

    # ═══ PIER / WATER SPORTS CENTER (coastline near X=70, Y=-188) ═══
    pier_cx, pier_cy, pier_z = 70.0, -188.0, 0.0
    # Jetty (15m x 3m, extending 12m into water southward)
    jetty_x1, jetty_x2 = pier_cx - 1.5, pier_cx + 1.5
    jetty_y1, jetty_y2 = pier_cy, pier_cy - 12.0
    for jx in np.arange(jetty_x1, jetty_x2 + PITCH, PITCH * 0.6):
        for jy in np.arange(jetty_y2, jetty_y1 + PITCH, PITCH * 0.6):
            gx, gy = snap_grid(jx, jy)
            add((gx, gy, pier_z + 1.0), COLORS['path_gray'])
            add((gx, gy, pier_z + 1.5), COLORS['path_gray'])
    # Support piles
    for p_y in np.arange(jetty_y2 + 2.0, jetty_y1 - 1.0, 4.0):
        for p_x in [jetty_x1, (jetty_x1 + jetty_x2) / 2, jetty_x2]:
            for pz in np.arange(pier_z, pier_z + 1.5, PITCH):
                add(snap_grid(p_x, p_y, pz), COLORS['concrete_pillar'])
    # Boathouse at land end (5m x 4m)
    bh_x1, bh_x2 = jetty_x1 - 1.0, jetty_x2 + 1.0
    bh_y1, bh_y2 = jetty_y1, jetty_y1 + 4.0
    for bz in np.arange(pier_z + 1.5, pier_z + 4.5, PITCH):
        for bx in np.arange(bh_x1, bh_x2 + PITCH, PITCH * 0.6):
            add(snap_grid(bx, bh_y1, bz), COLORS['building_wall'])
            add(snap_grid(bx, bh_y2, bz), COLORS['building_wall'])
        for by in np.arange(bh_y1, bh_y2 + PITCH, PITCH * 0.6):
            add(snap_grid(bh_x1, by, bz), COLORS['building_wall'])
            add(snap_grid(bh_x2, by, bz), COLORS['building_glass'])
    bh_roof = filled_rect(bh_x1, bh_y1, bh_x2, bh_y2, height=1, z_offset=pier_z + 4.5)
    for pos in bh_roof: add(pos, COLORS['building_roof'])
    # Small moored boats (2 ellipses)
    for (boat_y, boat_x) in [(pier_cy - 14.0, pier_cx - 3.0), (pier_cy - 15.0, pier_cx + 3.0)]:
        boat = filled_ellipse((boat_x, boat_y), 2.0, 0.8, z_offset=pier_z + 0.5)
        for pos in boat: add(pos, COLORS['track_white'])
    print(f"  Pier + Water Sports Center added")

    # ═══ CAMPUS CONNECTING PATHS ═══
    def add_path_segment(x1, y1, x2, y2, z, width=2.0):
        steps = int(math.sqrt((x2-x1)**2 + (y2-y1)**2) / PITCH * 2)
        for st in range(steps):
            t = st / max(steps, 1)
            px = x1 + (x2 - x1) * t; py = y1 + (y2 - y1) * t
            for pw in np.arange(-width/2, width/2, PITCH * 0.6):
                gx, gy = snap_grid(px, py + pw * 0.3)
                add((gx, gy, z), COLORS['path_gray'])

    # Key path connections
    add_path_segment(0, -84, 0, -66, 38.0, 2.5)        # Plaza south to CYT north
    add_path_segment(50, -110, 50, -66, 32.0, 2.0)       # Halls UG I to CYT
    add_path_segment(-50, -50, -32, -32, 38.0, 2.0)      # LSK to Arc west
    add_path_segment(40, 25, 80, -155, 38.0, 2.0)        # Library to Track
    add_path_segment(68, -120, 82, -135, 24.0, 2.0)      # Halls to Pool
    add_path_segment(-50, -30, -65, 35, 38.0, 2.0)       # LSK to Garden
    add_path_segment(85, -155, 120, -190, 2.0, 2.5)      # Track to Shaw
    print(f"  Campus path network added")

    # ═══ SCATTERED VEGETATION IN GAPS ═══
    for vi in range(25):
        vsx = (hash(f"vegx{vi}") % 300 - 100)  # X: -100 to 200
        vsy = (hash(f"vegy{vi}") % 400 - 300)  # Y: -300 to 100
        gx, gy = snap_grid(vsx, vsy)
        # Only add in empty areas (don't overwrite buildings)
        if (gx, gy, 38.0) in pos_color: continue
        z_for_this = 38.0
        if vsy < -185: z_for_this = -1.0  # near ocean
        elif vsy < -80: z_for_this = 2.0  # lower terrace
        # Small tree
        th = 4 + int(abs(hash(f"vth{vi}") % 1000) / 250)
        for hh in range(th):
            add(snap_grid(gx, gy, z_for_this + hh * PITCH), COLORS['trunk_brown'])
        for cl_r, cl_c in [(2.0, 'tree_dark'), (1.5, 'tree_green')]:
            layer = filled_circle((gx, gy), cl_r, z=z_for_this + th * PITCH + (3.0 - cl_r))
            for pos in layer: add(pos, COLORS[cl_c])
    print(f"  Scattered vegetation added")

    # Convert to lists
    for pos, color in pos_color.items():
        all_positions.append(pos); all_colors.append(color)

    print(f"\n  CAMPUS TOTAL: {len(all_positions):,} voxels")
    return all_positions, all_colors


# ═══════════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════════

def export_voxel_json(positions, colors, landmark_name, pitch=PITCH):
    """Export to JSON format compatible with viewer_voxel.html."""
    xs = [p[0] for p in positions]; ys = [p[1] for p in positions]; zs = [p[2] for p in positions]
    bbox = {
        "min": [round(min(xs), 1), round(min(ys), 1), round(min(zs), 1)],
        "max": [round(max(xs), 1), round(max(ys), 1), round(max(zs), 1)],
    }
    # Compute categories from colors
    BUILDING_C = {'building_wall','building_wall_dark','building_wall_rib','building_glass',
                  'building_glass_dark','building_glass_bright','building_roof','building_roof_dark',
                  'concrete_pillar','metal_gray','bench_brown','plaza_accent','plaza_stone',
                  'plaza_stone_mid','plaza_stone_dark','path_gray','path_dark','white','sundial_gold','red_bird','red_bird_bright'}
    TERRAIN_C = {'ground_gray','ground_brown','hillside_grass','hillside_dirt','hillside_rock',
                 'coast_rock','beach_sand','sand_beige','tree_green','tree_dark','tree_bright',
                 'trunk_brown','trunk_dark','field_green','field_alt_green'}
    WATER_C = {'water_blue','water_light','water_dark','deep_ocean'}
    cats = {"building":0,"terrain":0,"water":0,"other":0}
    for color in colors:
        cname = [k for k,v in COLORS.items() if v == tuple(color)]
        if cname:
            cn = cname[0]
            if cn in BUILDING_C: cats["building"] += 1
            elif cn in TERRAIN_C: cats["terrain"] += 1
            elif cn in WATER_C: cats["water"] += 1
            else: cats["other"] += 1
        else:
            cats["other"] += 1
    data = {"pitch": pitch, "count": len(positions), "bbox": bbox,
            "positions": positions, "colors": colors, "categories": cats}
    out_path = OUTPUT_DIR / f"voxel_{landmark_name}.json"
    json_str = json.dumps(data, separators=(",", ":"))
    out_path.write_text(json_str)
    kb = len(json_str) / 1024
    print(f"\n  → {out_path.name}: {len(positions):,} voxels, {kb:.0f} KB")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build clean Minecraft voxel structures")
    parser.add_argument("--landmark", choices=['track', 'plaza', 'academic', 'atrium', 'library', 'lsk',
                        'coastline', 'cyt', 'halls', 'pool', 'spring', 'shaw', 'gate', 'garden',
                        'campus', 'all'], required=True)
    args = parser.parse_args()
    builders = {'track': build_track, 'plaza': build_plaza, 'academic': build_academic,
                'atrium': build_atrium, 'library': build_library, 'lsk': build_lsk,
                'coastline': build_coastline, 'cyt': build_cyt, 'halls': build_halls,
                'pool': build_pool, 'spring': build_spring, 'shaw': build_shaw,
                'gate': build_north_gate, 'garden': build_garden, 'campus': build_campus}
    landmarks = list(builders.keys()) if args.landmark == 'all' else [args.landmark]
    for lm in landmarks:
        t0 = time.time()
        positions, colors = builders[lm]()
        export_voxel_json(positions, colors, lm)
        print(f"  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()

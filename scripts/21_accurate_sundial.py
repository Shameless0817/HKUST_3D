#!/usr/bin/env python3
"""
Replace the simple red cylinder/cone sundial markers with an accurate
equatorial ring sundial ("Circle of Time" / 时间之轮) model.

Reads the textured+piazza GLB from script 19, removes the old marker
primitives (if any), and adds accurate sundial geometry.

Sundial components:
  1. Stepped circular base (3 tiers, dark stone)
  2. Central pillar (cast steel, dark red-brown)
  3. Equatorial ring (inclined ~67.7° from horizontal, cast steel)
  4. Gnomon rod (through ring center, parallel to Earth's axis)
  5. "Flying bird" support arms (V-shaped curved struts)
  6. Plaza ground disc (red-brown brick color)
  7. Square boundary ring (orange)
"""
import json, struct, io, math
from pathlib import Path
import numpy as np

PROJECT = Path("/home/zliki/HKUST_3D")
INPUT_GLB = PROJECT / "output/demo/hkust_piazza.glb"   # from script 19
OUTPUT_GLB = PROJECT / "output/demo/hkust_piazza.glb"

# HKUST latitude
HK_LATITUDE = 22.337  # degrees north
RING_INCLINATION = 90.0 - HK_LATITUDE  # ~67.66° from horizontal


def parse_glb(path):
    data = Path(path).read_bytes()
    offset = 12
    json_len = struct.unpack_from('<I', data, offset)[0]
    offset += 8
    gltf = json.loads(data[offset:offset+json_len])
    offset += json_len
    bin_data = b''
    if offset < len(data):
        bin_len = struct.unpack_from('<I', data, offset)[0]
        offset += 8
        bin_data = data[offset:offset+bin_len]
    return gltf, bin_data


def write_glb(gltf, bin_data, path):
    json_str = json.dumps(gltf, separators=(',', ':'))
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('utf-8')
    while len(bin_data) % 4 != 0:
        bin_data += b'\x00'
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', 0x46546C67))
        f.write(struct.pack('<I', 2))
        f.write(struct.pack('<I', total))
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(struct.pack('<I', 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack('<I', len(bin_data)))
        f.write(struct.pack('<I', 0x004E4942))
        f.write(bin_data)


def read_accessor(acc_idx, accessors, buffer_views, bin_data):
    acc = accessors[acc_idx]
    bv_idx = acc.get('bufferView')
    if bv_idx is None:
        return np.array([])
    bv = buffer_views[bv_idx]
    off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    comp_type = acc['componentType']
    acc_type = acc['type']
    ts = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    tc = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    elem = ts[comp_type]
    ncomp = tc[acc_type]
    total = count * ncomp
    dtype = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
             5125: np.uint32, 5126: np.float32}[comp_type]
    raw = np.frombuffer(bin_data, dtype=dtype, count=total, offset=off)
    if acc_type in ('VEC2', 'VEC3', 'VEC4'):
        raw = raw.reshape(-1, ncomp)
    return raw.copy()


# ── Rotation helpers ─────────────────────────────────────

def rot_x(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotate_verts(verts, R):
    """Rotate vertices around origin using rotation matrix R."""
    return verts @ R.T


# ── Geometry builders ────────────────────────────────────

def make_torus(center, major_r, tube_r, color, major_segs=64, tube_segs=10):
    """Torus ring in XY plane, centered at `center`."""
    cx, cy, cz = center
    verts = []
    for i in range(major_segs):
        theta = 2 * np.pi * i / major_segs
        ring_x = cx + major_r * np.cos(theta)
        ring_y = cy + major_r * np.sin(theta)
        for j in range(tube_segs):
            phi = 2 * np.pi * j / tube_segs
            verts.append([
                ring_x + tube_r * np.cos(phi) * np.cos(theta),
                ring_y + tube_r * np.cos(phi) * np.sin(theta),
                cz + tube_r * np.sin(phi),
            ])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(major_segs):
        i_next = (i + 1) % major_segs
        for j in range(tube_segs):
            j_next = (j + 1) % tube_segs
            a = i * tube_segs + j
            b = i * tube_segs + j_next
            c = i_next * tube_segs + j_next
            d = i_next * tube_segs + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces = np.array(faces, dtype=np.uint32)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


def make_cylinder_mesh(base_center, radius, height, color, sections=24):
    """Cylinder along Z axis, base at base_center."""
    cx, cy, cz = base_center
    verts = [[cx, cy, cz], [cx, cy, cz + height]]  # 0:bottom, 1:top center
    for i in range(sections):
        angle = 2 * np.pi * i / sections
        verts.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle), cz])
    for i in range(sections):
        angle = 2 * np.pi * i / sections
        verts.append([cx + radius * np.cos(angle), cy + radius * np.sin(angle), cz + height])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    # Bottom fan
    for i in range(sections):
        b0, b1 = 2 + i, 2 + (i + 1) % sections
        faces.append([0, b1, b0])
    # Top fan
    for i in range(sections):
        t0, t1 = 2 + sections + i, 2 + sections + (i + 1) % sections
        faces.append([1, t0, t1])
    # Side walls
    for i in range(sections):
        b0, b1 = 2 + i, 2 + (i + 1) % sections
        t0, t1 = 2 + sections + i, 2 + sections + (i + 1) % sections
        faces.append([b0, t0, t1])
        faces.append([b0, t1, b1])
    faces = np.array(faces, dtype=np.uint32)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


def make_rod(center, direction, length, radius, color, sections=8):
    """Thin cylinder rod along `direction` (unit vector), centered at `center`."""
    # Build along Z first
    half = length / 2
    verts = []
    verts.append([0, 0, -half])  # bottom center
    verts.append([0, 0, half])   # top center
    for i in range(sections):
        angle = 2 * np.pi * i / sections
        verts.append([radius * np.cos(angle), radius * np.sin(angle), -half])
    for i in range(sections):
        angle = 2 * np.pi * i / sections
        verts.append([radius * np.cos(angle), radius * np.sin(angle), half])
    verts = np.array(verts, dtype=np.float32)

    # Rotate from Z to `direction`
    z_axis = np.array([0., 0., 1.])
    d = np.array(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    if np.abs(np.dot(z_axis, d) - 1) < 1e-9:
        R = np.eye(3)
    elif np.abs(np.dot(z_axis, d) + 1) < 1e-9:
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        v = np.cross(z_axis, d)
        v = v / np.linalg.norm(v)
        c = np.dot(z_axis, d)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    verts = rotate_verts(verts, R) + np.array(center)

    faces = []
    # Bottom fan
    for i in range(sections):
        b0, b1 = 2 + i, 2 + (i + 1) % sections
        faces.append([0, b1, b0])
    # Top fan
    for i in range(sections):
        t0, t1 = 2 + sections + i, 2 + sections + (i + 1) % sections
        faces.append([1, t0, t1])
    # Side walls
    for i in range(sections):
        b0, b1 = 2 + i, 2 + (i + 1) % sections
        t0, t1 = 2 + sections + i, 2 + sections + (i + 1) % sections
        faces.append([b0, t0, t1])
        faces.append([b0, t1, b1])
    faces = np.array(faces, dtype=np.uint32)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


def make_curved_strut(start, end, control_offset, radius, color, segments=16, ring_segs=8):
    """Curved tube from start to end via a quadratic Bezier bend.
    control_offset is a vector added to the midpoint for the control point."""
    mid = (start + end) / 2
    control = mid + np.array(control_offset)

    # Sample centerline points
    centerline = []
    for i in range(segments + 1):
        t = i / segments
        # Quadratic Bezier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
        pt = ((1-t)**2 * start + 2*(1-t)*t * control + t**2 * end)
        centerline.append(pt)
    centerline = np.array(centerline)

    # Build tube: for each centerline point, create a ring
    verts = []
    for ci, cp in enumerate(centerline):
        # Tangent direction
        if ci == 0:
            tangent = centerline[1] - centerline[0]
        elif ci == segments:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[ci+1] - centerline[ci-1]
        t_len = np.linalg.norm(tangent)
        if t_len < 1e-9:
            tangent = np.array([0., 0., 1.])
        else:
            tangent = tangent / t_len

        # Build a basis: find two perpendicular vectors to tangent
        if abs(tangent[0]) < 0.9:
            u = np.cross(tangent, [1., 0., 0.])
        else:
            u = np.cross(tangent, [0., 1., 0.])
        u = u / np.linalg.norm(u)
        v = np.cross(tangent, u)

        for j in range(ring_segs):
            angle = 2 * np.pi * j / ring_segs
            verts.append(cp + radius * (np.cos(angle) * u + np.sin(angle) * v))
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(segments):
        for j in range(ring_segs):
            j_next = (j + 1) % ring_segs
            a = i * ring_segs + j
            b = i * ring_segs + j_next
            c = (i+1) * ring_segs + j_next
            d = (i+1) * ring_segs + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces = np.array(faces, dtype=np.uint32)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


def make_arc_strut(center, radius, start_angle, end_angle, z_height, tube_r, color, segments=24, ring_segs=8):
    """Horizontal arc tube in XY plane at given z_height, centered at center."""
    verts = []
    for i in range(segments + 1):
        frac = i / segments
        angle = start_angle + frac * (end_angle - start_angle)
        arc_x = center[0] + radius * np.cos(angle)
        arc_y = center[1] + radius * np.sin(angle)
        # Tangent direction in XY plane
        tx, ty = -np.sin(angle), np.cos(angle)
        for j in range(ring_segs):
            phi = 2 * np.pi * j / ring_segs
            # Normal in XY plane points radially outward: (cos(angle), sin(angle))
            # Binormal is Z
            verts.append([
                arc_x + tube_r * np.cos(phi) * np.cos(angle),
                arc_y + tube_r * np.cos(phi) * np.sin(angle),
                z_height + tube_r * np.sin(phi),
            ])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(segments):
        for j in range(ring_segs):
            j_next = (j + 1) % ring_segs
            a = i * ring_segs + j
            b = i * ring_segs + j_next
            c = (i+1) * ring_segs + j_next
            d = (i+1) * ring_segs + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces = np.array(faces, dtype=np.uint32)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(verts), 1))
    return verts, faces, colors


# ── GLB buffer helpers ───────────────────────────────────

def append_primitive(verts, faces, colors, bin_chunks, buffer_views, accessors, bo):
    """Append POSITION + indices + COLOR_0 data. Returns accessor indices and new bo."""
    # Positions
    pos_raw = verts.astype(np.float32).tobytes()
    pos_bv = {'buffer': 0, 'byteOffset': bo, 'byteLength': len(pos_raw)}
    pos_acc = {
        'bufferView': len(buffer_views), 'componentType': 5126,
        'count': len(verts), 'type': 'VEC3',
        'min': [float(verts[:,0].min()), float(verts[:,1].min()), float(verts[:,2].min())],
        'max': [float(verts[:,0].max()), float(verts[:,1].max()), float(verts[:,2].max())],
    }
    buffer_views.append(pos_bv)
    accessors.append(pos_acc)
    bin_chunks.append(pos_raw)
    bo += len(pos_raw)
    pos_idx = len(accessors) - 1

    # Indices
    idx_raw = faces.astype(np.uint32).flatten().tobytes()
    idx_bv = {'buffer': 0, 'byteOffset': bo, 'byteLength': len(idx_raw)}
    idx_acc = {'bufferView': len(buffer_views), 'componentType': 5125,
               'count': len(faces.flatten()), 'type': 'SCALAR'}
    buffer_views.append(idx_bv)
    accessors.append(idx_acc)
    bin_chunks.append(idx_raw)
    bo += len(idx_raw)
    idx_idx = len(accessors) - 1

    # Colors (RGBA, normalized UNSIGNED_BYTE)
    col_raw = colors.astype(np.uint8).tobytes()
    col_bv = {'buffer': 0, 'byteOffset': bo, 'byteLength': len(col_raw)}
    col_acc = {'bufferView': len(buffer_views), 'componentType': 5121,
               'count': len(colors), 'type': 'VEC4', 'normalized': True}
    buffer_views.append(col_bv)
    accessors.append(col_acc)
    bin_chunks.append(col_raw)
    bo += len(col_raw)

    return pos_idx, idx_idx, len(accessors) - 1, bo


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("HKUST Red Bird Square — Accurate Sundial Model")
    print("=" * 60)

    # ── Load existing GLB ──────────────────────────────────
    print(f"\nLoading: {INPUT_GLB.name}")
    gltf, bin_data = parse_glb(INPUT_GLB)

    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']
    meshes = gltf['meshes']

    # Keep only the first (textured terrain) primitive; discard old markers
    prim0 = meshes[0]['primitives'][0]
    v = read_accessor(prim0['attributes']['POSITION'], accessors, buffer_views, bin_data)
    print(f"  Textured terrain: {len(v):,} vertices")

    # Find square center
    z_vals = v[:, 2]
    plateau = v[(z_vals > 50) & (z_vals < 70)]
    cx, cy = np.median(plateau[:, 0]), np.median(plateau[:, 1])
    for _ in range(3):
        d = np.sqrt((plateau[:,0]-cx)**2 + (plateau[:,1]-cy)**2)
        close = d < 20
        if close.sum() < 50:
            break
        cx, cy = plateau[close, 0].mean(), plateau[close, 1].mean()
        plateau = plateau[close]

    nearby = np.sqrt((v[:,0]-cx)**2 + (v[:,1]-cy)**2) < 15
    ground_z = v[nearby, 2].min()
    print(f"  Square center: ({cx:.1f}, {cy:.1f}, {ground_z:.1f})")

    # Shift existing vertices so square center is at XY origin
    sx, sy = cx, cy
    existing_pos_acc = prim0['attributes']['POSITION']
    pos_bv_idx = accessors[existing_pos_acc]['bufferView']
    pos_bv = buffer_views[pos_bv_idx]
    pos_offset = pos_bv['byteOffset']
    pos_len = pos_bv['byteLength']

    existing_v = read_accessor(existing_pos_acc, accessors, buffer_views, bin_data)
    existing_v[:, 0] -= sx
    existing_v[:, 1] -= sy

    bin_data = bytearray(bin_data)
    bin_data[pos_offset:pos_offset+pos_len] = existing_v.astype(np.float32).tobytes()
    bin_data = bytes(bin_data)

    # Update min/max
    accessors[existing_pos_acc]['min'] = [float(existing_v[:,0].min()),
                                           float(existing_v[:,1].min()),
                                           float(existing_v[:,2].min())]
    accessors[existing_pos_acc]['max'] = [float(existing_v[:,0].max()),
                                           float(existing_v[:,1].max()),
                                           float(existing_v[:,2].max())]

    # ── Strip old marker primitives ────────────────────────
    meshes[0]['primitives'] = [prim0]
    # Remove old marker materials (keep only the textured material)
    if 'materials' in gltf and len(gltf['materials']) > 1:
        gltf['materials'] = gltf['materials'][:1]

    # ── Remove old marker accessors & bufferViews ──────────
    # Find which accessors are referenced by prim0 (keep those + all earlier ones)
    def all_refs(prim):
        refs = set()
        for attr in prim['attributes'].values():
            refs.add(attr)
        if 'indices' in prim:
            refs.add(prim['indices'])
        if 'material' in prim:
            mat_idx = prim['material']
            mat = gltf['materials'][mat_idx] if 'materials' in gltf else {}
            for tex in mat.get('pbrMetallicRoughness', {}).values():
                if isinstance(tex, dict) and 'index' in tex:
                    refs.add(tex['index'])
        return refs

    kept_acc_indices = all_refs(prim0)
    # Also track accessors referenced by images/textures/samplers
    for img in gltf.get('images', []):
        if 'bufferView' in img:
            kept_acc_indices.add(img['bufferView'])  # actually bufferView, not accessor
    # Expand: keep bufferViews referenced by kept accessors
    kept_bv_indices = set()
    for ai in kept_acc_indices:
        if ai < len(accessors):
            bvi = accessors[ai].get('bufferView')
            if bvi is not None:
                kept_bv_indices.add(bvi)
    # Also keep bufferViews for images
    for img in gltf.get('images', []):
        if 'bufferView' in img:
            kept_bv_indices.add(img['bufferView'])

    # Now trim bin_data to only what's needed
    # Find the max byte extent of kept bufferViews
    max_byte = 0
    for bvi in kept_bv_indices:
        if bvi < len(buffer_views):
            bv = buffer_views[bvi]
            end = bv.get('byteOffset', 0) + bv.get('byteLength', 0)
            max_byte = max(max_byte, end)

    # We'll rebuild cleanly: extract kept data, then build new buffer
    # This is complex — let's take a simpler approach:
    # Just keep ALL existing bufferViews/accessors/bin_data, and add new ones.
    # The old marker data stays in the binary but is unreferenced. It's tiny (~3KB)
    # so this is fine. We just need to reset primitives and materials.

    # Reset primitives to just the terrain
    meshes[0]['primitives'] = [prim0]
    # Remove old marker materials
    while len(gltf.get('materials', [])) > 1:
        gltf['materials'].pop()

    # ── Build sundial geometry ─────────────────────────────
    print("\nBuilding accurate equatorial ring sundial...")

    base_z = ground_z + 0.3  # slightly above ground

    # Color palette
    CAST_STEEL = [180, 55, 40, 255]     # dark red-brown (rusted cast steel)
    STEEL_DARK = [140, 40, 30, 255]
    STEEL_LIGHT = [200, 80, 50, 255]
    STONE_DARK = [100, 95, 90, 255]     # dark gray stone base
    STONE_MID = [130, 125, 120, 255]
    STONE_LIGHT = [160, 155, 150, 255]
    GOLD_BRONZE = [200, 150, 60, 255]   # bronze/gold accents
    BRICK_RED = [180, 70, 40, 200]      # semi-transparent brick disc
    RING_ORANGE = [220, 140, 40, 255]   # square boundary ring

    sundial_primitives = []

    # 1. Stepped circular base (3 tiers)
    tiers = [
        (2.2, 0.4, STONE_DARK),     # bottom: r=2.2m, h=0.4m
        (1.6, 0.4, STONE_MID),      # middle: r=1.6m, h=0.4m
        (1.0, 0.9, STONE_LIGHT),    # top: r=1.0m, h=0.9m
    ]
    tier_z = base_z
    for radius, height, color in tiers:
        cv, cf, cc = make_cylinder_mesh([0, 0, tier_z], radius, height, color, sections=32)
        sundial_primitives.append((cv, cf, cc))
        tier_z += height

    base_top_z = tier_z  # top of stepped base

    # 2. Central pillar (cast steel)
    pillar_r = 0.35
    pillar_h = 3.8
    pv, pf, pc = make_cylinder_mesh([0, 0, base_top_z], pillar_r, pillar_h, CAST_STEEL, sections=24)
    sundial_primitives.append((pv, pf, pc))
    pillar_top_z = base_top_z + pillar_h

    # 3. "Flying bird" support arms — two curved struts from pillar top
    # sweeping outward and upward to support the ring
    ring_center_z = pillar_top_z + 0.5  # ring sits slightly above pillar
    support_start = np.array([0., 0., pillar_top_z])
    # Two arms: one sweeping east, one west
    for direction in [-1, 1]:
        arm_end = np.array([direction * 1.2, 0., ring_center_z - 0.3])
        control = np.array([direction * 0.8, 0., pillar_top_z + 0.8])
        av, af, ac = make_curved_strut(support_start, arm_end, control - support_start,
                                        0.06, STEEL_DARK, segments=16, ring_segs=8)
        sundial_primitives.append((av, af, ac))

    # 4. Small decorative ring at pillar top (where arms branch out)
    prv, prf, prc = make_torus([0, 0, pillar_top_z], 0.45, 0.06, GOLD_BRONZE, major_segs=48, tube_segs=8)
    sundial_primitives.append((prv, prf, prc))

    # 5. Equatorial ring — THE defining feature
    # Torus in XY plane, then tilted ~67.7° around X axis
    ring_major_r = 1.5
    ring_tube_r = 0.10
    ring_center = np.array([0., 0., ring_center_z])

    # Generate torus at origin
    erv, erf, erc = make_torus([0, 0, 0], ring_major_r, ring_tube_r, CAST_STEEL,
                                 major_segs=72, tube_segs=10)

    # Rotate around X axis: ring normal tilts from Z toward Y (north)
    # Rotation angle = -(90° - latitude) = -RING_INCLINATION
    R_ring = rot_x(-RING_INCLINATION)
    erv_rot = rotate_verts(erv, R_ring) + ring_center
    sundial_primitives.append((erv_rot, erf, erc))

    # 6. Inner ring (slightly smaller, for the hour scale markings)
    inner_ring_r = 1.35
    irv, irf, irc = make_torus([0, 0, 0], inner_ring_r, 0.04, GOLD_BRONZE,
                                 major_segs=72, tube_segs=8)
    irv_rot = rotate_verts(irv, R_ring) + ring_center
    sundial_primitives.append((irv_rot, irf, irc))

    # 7. Gnomon — rod through ring center, parallel to Earth's axis
    # Polar axis direction: (0, cos(lat), sin(lat)) in (X,Y,Z) = (east, north, up)
    lat_rad = np.radians(HK_LATITUDE)
    polar_dir = np.array([0., np.cos(lat_rad), np.sin(lat_rad)])  # points N and up
    gnomon_length = 3.0
    gv, gf, gc = make_rod(ring_center, polar_dir, gnomon_length, 0.05, STEEL_LIGHT, sections=10)
    sundial_primitives.append((gv, gf, gc))

    # 8. Cross-brace ring at gnomon midpoints (decorative)
    # Small ring perpendicular to polar axis at the gnomon center
    # This is the "equatorial ring" proper — we already have it.
    # Add small sphere/ring at top of gnomon as finial
    finial_v, finial_f, finial_c = make_torus(
        ring_center + polar_dir * gnomon_length * 0.48, 0.08, 0.03, GOLD_BRONZE,
        major_segs=16, tube_segs=8)
    sundial_primitives.append((finial_v, finial_f, finial_c))

    # 9. Plaza ground — layered concentric discs (red brick pavers)
    # The real Red Bird Square has concentric circular brick paving patterns.
    # Use multiple thin overlapping discs to create visible rings.
    BRICK_DARK = [160, 55, 30, 220]
    BRICK_MID = [185, 75, 40, 220]
    BRICK_LIGHT = [200, 90, 50, 220]
    BRICK_ACCENT = [170, 60, 35, 230]

    plaza_rings = [
        (28.0, BRICK_DARK),     # full square base
        (22.0, BRICK_MID),      # inner zone
        (16.0, BRICK_LIGHT),    # middle zone
        (10.0, BRICK_ACCENT),   # inner courtyard
        (5.0, BRICK_MID),       # sundial surround
    ]
    for radius, color in plaza_rings:
        dv, df, dc = make_cylinder_mesh([0, 0, base_z - 0.05], radius, 0.12, color, sections=64)
        sundial_primitives.append((dv, df, dc))

    # Concentric paving rings (thin torus lines marking the circular brick pattern)
    TERRACOTTA = [210, 100, 55, 200]
    for ring_r in [7.0, 13.0, 19.0, 25.0]:
        rv, rf, rc = make_torus([0, 0, base_z + 0.02], ring_r, 0.12, TERRACOTTA,
                                  major_segs=96, tube_segs=8)
        sundial_primitives.append((rv, rf, rc))

    # Radial lines (spokes) from center outward — simulate radial brick pattern
    SPOKE_COLOR = [190, 85, 45, 180]
    num_spokes = 16
    spoke_length = 27.0
    for i in range(num_spokes):
        angle = 2 * np.pi * i / num_spokes
        dx = spoke_length * np.cos(angle)
        dy = spoke_length * np.sin(angle)
        # Thin flat bar from near-center to outer ring
        sv, sf, sc = make_rod([dx/2, dy/2, base_z + 0.01],
                                [dx/spoke_length, dy/spoke_length, 0.],
                                spoke_length, 0.08, SPOKE_COLOR, sections=6)
        sundial_primitives.append((sv, sf, sc))

    # 10. Square boundary ring — prominent outer ring
    rv, rf, rc = make_torus([0, 0, base_z + 0.25], 28, 0.35, RING_ORANGE,
                              major_segs=96, tube_segs=10)
    sundial_primitives.append((rv, rf, rc))

    # Inner decorative ring at the edge of the main paved area
    rv2, rf2, rc2 = make_torus([0, 0, base_z + 0.15], 20, 0.18, [200, 120, 50, 255],
                                 major_segs=80, tube_segs=8)
    sundial_primitives.append((rv2, rf2, rc2))

    print(f"  Sundial components: {len(sundial_primitives)} primitives")

    # ── Append to GLB ──────────────────────────────────────
    print("Appending to GLB...")

    # Find current binary end
    existing_bin_end = 0
    for bv in buffer_views:
        end = bv.get('byteOffset', 0) + bv.get('byteLength', 0)
        existing_bin_end = max(existing_bin_end, end)

    bo = existing_bin_end
    bin_chunks = [bin_data]

    # Pad to 4-byte alignment
    while bo % 4 != 0:
        bin_chunks.append(b'\x00')
        bo += 1

    # Add marker material
    marker_mat_idx = len(gltf.get('materials', []))
    gltf.setdefault('materials', []).append({
        'pbrMetallicRoughness': {
            'baseColorFactor': [1.0, 1.0, 1.0, 1.0],
            'metallicFactor': 0.0,
            'roughnessFactor': 0.5,
        },
        'doubleSided': True,
    })

    new_primitives = []
    for verts, faces, cols in sundial_primitives:
        pi, ii, ci, bo = append_primitive(verts, faces, cols,
                                            bin_chunks, buffer_views, accessors, bo)
        new_primitives.append({
            'attributes': {'POSITION': pi, 'COLOR_0': ci},
            'indices': ii,
            'material': marker_mat_idx,
        })

    meshes[0]['primitives'].extend(new_primitives)

    # Update buffer
    gltf['buffers'][0]['byteLength'] = bo

    bin_data_out = b''.join(bin_chunks)
    write_glb(gltf, bin_data_out, OUTPUT_GLB)

    # ── Report ─────────────────────────────────────────────
    mb = OUTPUT_GLB.stat().st_size / 1e6
    idx_acc = accessors[prim0['indices']]
    total_v = len(existing_v) + sum(p[0].shape[0] for p in sundial_primitives)
    total_f = idx_acc['count'] // 3 + sum(p[1].shape[0] for p in sundial_primitives)
    print(f"\n{'='*60}")
    print(f"✓ {OUTPUT_GLB} ({mb:.1f} MB)")
    print(f"  Terrain: {len(existing_v):,}v textured + normals")
    print(f"  Sundial base: 3-tier stepped stone + pillar")
    print(f"  Equatorial ring: inclined {RING_INCLINATION:.1f}° (HK latitude {HK_LATITUDE}°)")
    print(f"  Gnomon: rod parallel to Earth's axis")
    print(f"  Flying bird arms: curved supports")
    print(f"  Total primitives: {len(meshes[0]['primitives'])}")
    print(f"\nOpen https://gltf-viewer.donmccurdy.com/ and drag in:")
    print(f"  {OUTPUT_GLB}")
    print(f"{'='*60}")

    # Verify
    vgltf, _ = parse_glb(OUTPUT_GLB)
    vm = vgltf['meshes'][0]
    print(f"\nVerification: {len(vm['primitives'])} primitives")
    for i, p in enumerate(vm['primitives']):
        attrs = list(p['attributes'].keys())
        vc = vgltf['accessors'][p['attributes']['POSITION']]['count']
        print(f"  Prim {i}: {vc:,}v, attrs={attrs}")


if __name__ == '__main__':
    main()

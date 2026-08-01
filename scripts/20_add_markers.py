#!/usr/bin/env python3
"""
Add visual markers to the textured piazza GLB by directly manipulating
the GLB binary — preserves textures while adding vertex-colored primitives.

Markers:
  1. Red cylinder = sundial pillar
  2. Gold cone = sundial flame
  3. Dark semi-transparent disc = plaza ground
  4. Orange ring = square boundary
"""
import json, struct, io, math
from pathlib import Path
import numpy as np

PROJECT = Path("/home/zliki/HKUST_3D")
INPUT_GLB = PROJECT / "output/demo/hkust_piazza.glb"   # from script 19
OUTPUT_GLB = PROJECT / "output/demo/hkust_piazza.glb"


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


def make_cylinder_mesh(base_center, radius, height, color, sections=16):
    """Create cylinder vertices+faces+colors, axis along Z, base at base_center."""
    cx, cy, cz = base_center
    verts = []
    # Bottom center
    verts.append([cx, cy, cz])
    # Top center
    verts.append([cx, cy, cz + height])
    # Bottom ring
    for i in range(sections):
        angle = 2 * math.pi * i / sections
        verts.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle), cz])
    # Top ring
    for i in range(sections):
        angle = 2 * math.pi * i / sections
        verts.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle), cz + height])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    # Bottom fan
    for i in range(sections):
        b0 = 2 + i
        b1 = 2 + (i + 1) % sections
        faces.append([0, b1, b0])
    # Top fan
    for i in range(sections):
        t0 = 2 + sections + i
        t1 = 2 + sections + (i + 1) % sections
        faces.append([1, t0, t1])
    # Side walls
    for i in range(sections):
        b0 = 2 + i
        b1 = 2 + (i + 1) % sections
        t0 = 2 + sections + i
        t1 = 2 + sections + (i + 1) % sections
        faces.append([b0, t0, t1])
        faces.append([b0, t1, b1])
    faces = np.array(faces, dtype=np.uint32)

    colors = np.tile(color, (len(verts), 1)).astype(np.uint8)
    return verts, faces, colors


def make_cone_mesh(base_center, radius, height, color, sections=16):
    """Create cone vertices+faces+colors, axis along Z, base at base_center."""
    cx, cy, cz = base_center
    verts = [[cx, cy, cz + height]]  # tip at index 0
    # Base ring
    for i in range(sections):
        angle = 2 * math.pi * i / sections
        verts.append([cx + radius * math.cos(angle), cy + radius * math.sin(angle), cz])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    # Base fan
    for i in range(sections):
        b0 = 1 + i
        b1 = 1 + (i + 1) % sections
        faces.append([b0, b1, 0])
    # Side triangles
    for i in range(sections):
        b0 = 1 + i
        b1 = 1 + (i + 1) % sections
        faces.append([b0, 0, b1])  # both sides
    faces = np.array(faces, dtype=np.uint32)

    colors = np.tile(color, (len(verts), 1)).astype(np.uint8)
    return verts, faces, colors


def make_disc_mesh(center, radius, color, sections=48):
    """Create flat disc on ground (very thin cylinder)."""
    cx, cy, cz = center
    h = 0.15  # very thin
    return make_cylinder_mesh(center, radius, h, color, sections)


def make_ring_mesh(center, radius, tube_radius, color, sections=64, tube_sections=8):
    """Create a torus ring approximating a square boundary."""
    cx, cy, cz = center
    verts = []
    # Create a torus
    for i in range(sections):
        angle = 2 * math.pi * i / sections
        ring_cx = cx + radius * math.cos(angle)
        ring_cy = cy + radius * math.sin(angle)
        for j in range(tube_sections):
            t_angle = 2 * math.pi * j / tube_sections
            verts.append([
                ring_cx + tube_radius * math.cos(t_angle) * math.cos(angle),
                ring_cy + tube_radius * math.cos(t_angle) * math.sin(angle),
                cz + tube_radius * math.sin(t_angle),
            ])
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for i in range(sections):
        i_next = (i + 1) % sections
        for j in range(tube_sections):
            j_next = (j + 1) % tube_sections
            a = i * tube_sections + j
            b = i * tube_sections + j_next
            c = i_next * tube_sections + j_next
            d = i_next * tube_sections + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    faces = np.array(faces, dtype=np.uint32)

    colors = np.tile(color, (len(verts), 1)).astype(np.uint8)
    return verts, faces, colors


def append_primitive_data(verts, faces, colors, bin_chunks, buffer_views, accessors, bo):
    """Add position, index, and color data to GLB binary buffers. Returns accessor indices."""
    # Positions (FLOAT, VEC3)
    pos_raw = verts.astype(np.float32).tobytes()
    pos_bv = {'buffer': 0, 'byteOffset': bo, 'byteLength': len(pos_raw)}
    pos_acc = {
        'bufferView': len(buffer_views),
        'componentType': 5126, 'count': len(verts), 'type': 'VEC3',
        'min': [float(verts[:,0].min()), float(verts[:,1].min()), float(verts[:,2].min())],
        'max': [float(verts[:,0].max()), float(verts[:,1].max()), float(verts[:,2].max())],
    }
    buffer_views.append(pos_bv)
    accessors.append(pos_acc)
    bin_chunks.append(pos_raw)
    bo += len(pos_raw)
    pos_idx = len(accessors) - 1

    # Indices (UNSIGNED_INT, SCALAR)
    idx_raw = faces.astype(np.uint32).flatten().tobytes()
    idx_bv = {'buffer': 0, 'byteOffset': bo, 'byteLength': len(idx_raw)}
    idx_acc = {'bufferView': len(buffer_views), 'componentType': 5125, 'count': len(faces.flatten()), 'type': 'SCALAR'}
    buffer_views.append(idx_bv)
    accessors.append(idx_acc)
    bin_chunks.append(idx_raw)
    bo += len(idx_raw)
    idx_acc_idx = len(accessors) - 1

    # Colors (UNSIGNED_BYTE, VEC4, normalized)
    col_raw = colors.astype(np.uint8).tobytes()
    col_bv = {'buffer': 0, 'byteOffset': bo, 'byteLength': len(col_raw)}
    col_acc = {'bufferView': len(buffer_views), 'componentType': 5121, 'count': len(colors), 'type': 'VEC4', 'normalized': True}
    buffer_views.append(col_bv)
    accessors.append(col_acc)
    bin_chunks.append(col_raw)
    bo += len(col_raw)
    col_idx = len(accessors) - 1

    return pos_idx, idx_acc_idx, col_idx, bo


def main():
    print("=" * 60)
    print("HKUST Red Bird Square — Adding Visual Markers")
    print("=" * 60)

    # Load existing textured GLB
    print(f"\nLoading textured piazza model...")
    gltf, bin_data = parse_glb(INPUT_GLB)

    # Find the existing mesh vertices to locate the square center
    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']
    meshes = gltf['meshes']

    prim = meshes[0]['primitives'][0]
    v = read_accessor(prim['attributes']['POSITION'], accessors, buffer_views, bin_data)
    print(f"  Existing: {len(v):,} vertices, textured")

    # Find square center (same algorithm as script 19)
    z = v[:, 2]
    plateau_mask = (z > 50) & (z < 70)
    pv = v[plateau_mask]
    pmed_x, pmed_y = np.median(pv[:,0]), np.median(pv[:,1])
    for _ in range(3):
        pdist = np.sqrt((pv[:,0]-pmed_x)**2 + (pv[:,1]-pmed_y)**2)
        close = pdist < 20
        if close.sum() < 50:
            break
        pmed_x, pmed_y = pv[close,0].mean(), pv[close,1].mean()
        pv = pv[close]

    nearby = np.sqrt((v[:,0]-pmed_x)**2 + (v[:,1]-pmed_y)**2) < 15
    ground_z = v[nearby, 2].min()
    square_center = np.array([pmed_x, pmed_y, ground_z])
    print(f"  Square center: ({pmed_x:.1f}, {pmed_y:.1f}, {ground_z:.1f})")

    # ── Build marker geometry ───────────────────────────
    print("\nBuilding marker geometry...")

    # 1. Sundial pillar: red cylinder, 8.5m tall, at square center
    sv, sf, sc = make_cylinder_mesh(
        square_center + [0, 0, 0.3],  # on top of the platform
        0.5, 8.5, [220, 40, 30, 255])

    # 2. Flame: gold cone on top of pillar
    cone_base = square_center + [0, 0, 0.3 + 8.5]
    cv, cf, cc = make_cone_mesh(cone_base, 1.2, 2.5, [255, 180, 30, 255])

    # 3. Plaza base disc
    dv, df, dc = make_disc_mesh(square_center, 8, [180, 70, 40, 200])

    # 4. Square boundary ring
    rv, rf, rc = make_ring_mesh(square_center + [0, 0, 0.3], 25, 0.3, [220, 140, 40, 255])

    # ── Append to GLB ───────────────────────────────────
    print("Appending markers to GLB...")

    # Rebuild the binary buffer, appending new data
    # First, determine current binary buffer size
    existing_bin_len = sum(bv.get('byteLength', 0) for bv in buffer_views)
    bo = existing_bin_len

    bin_chunks = [bin_data]  # start with existing binary data
    # Pad existing bin to 4-byte alignment
    while len(bin_data) % 4 != 0:
        bin_chunks.append(b'\x00')
        bo += 1

    # Use existing buffer_views and accessors lists (append to them)
    new_primitives = []

    # Sundial pillar primitive
    sp, si, sc_idx, bo = append_primitive_data(sv, sf, sc, bin_chunks, buffer_views, accessors, bo)
    new_primitives.append({
        'attributes': {'POSITION': sp, 'COLOR_0': sc_idx},
        'indices': si,
        'material': 1,  # use marker material
    })

    # Flame cone primitive
    cp, ci, cc_idx, bo = append_primitive_data(cv, cf, cc, bin_chunks, buffer_views, accessors, bo)
    new_primitives.append({
        'attributes': {'POSITION': cp, 'COLOR_0': cc_idx},
        'indices': ci,
        'material': 1,
    })

    # Plaza disc primitive
    dp, di, dc_idx, bo = append_primitive_data(dv, df, dc, bin_chunks, buffer_views, accessors, bo)
    new_primitives.append({
        'attributes': {'POSITION': dp, 'COLOR_0': dc_idx},
        'indices': di,
        'material': 1,
    })

    # Square ring primitive
    rp, ri, rc_idx, bo = append_primitive_data(rv, rf, rc, bin_chunks, buffer_views, accessors, bo)
    new_primitives.append({
        'attributes': {'POSITION': rp, 'COLOR_0': rc_idx},
        'indices': ri,
        'material': 1,
    })

    # Update the mesh with new primitives
    meshes[0]['primitives'].extend(new_primitives)

    # Add a marker material (unlit, vertex color)
    existing_mats = gltf.get('materials', [])
    marker_mat_idx = len(existing_mats)
    existing_mats.append({
        'pbrMetallicRoughness': {
            'baseColorFactor': [1.0, 1.0, 1.0, 1.0],
            'metallicFactor': 0.0,
            'roughnessFactor': 0.5,
        },
        'doubleSided': True,
    })
    gltf['materials'] = existing_mats

    # Update buffer byteLength
    gltf['buffers'][0]['byteLength'] = bo

    # ── Center the model on the square ──────────────────
    # Shift all vertices so square center is at XY origin
    # We need to update the existing vertex positions in the binary buffer
    # This is tricky — let's just apply the shift to all position accessors
    sx, sy = square_center[0], square_center[1]

    # Get existing position accessor index
    existing_pos_acc = prim['attributes']['POSITION']

    # Read existing positions, shift, and update in the binary
    existing_v = read_accessor(existing_pos_acc, accessors, buffer_views, bin_data)
    existing_v[:, 0] -= sx
    existing_v[:, 1] -= sy

    # Update the binary data: replace the position bytes
    pos_bv_idx = accessors[existing_pos_acc]['bufferView']
    pos_bv = buffer_views[pos_bv_idx]
    pos_offset = pos_bv['byteOffset']
    pos_len = pos_bv['byteLength']
    new_pos_bytes = existing_v.astype(np.float32).tobytes()

    # Reconstruct bin_data with shifted positions
    bin_data = bytearray(bin_data)
    bin_data[pos_offset:pos_offset+pos_len] = new_pos_bytes
    bin_data = bytes(bin_data)

    # Update min/max in accessor
    accessors[existing_pos_acc]['min'] = [float(existing_v[:,0].min()), float(existing_v[:,1].min()), float(existing_v[:,2].min())]
    accessors[existing_pos_acc]['max'] = [float(existing_v[:,0].max()), float(existing_v[:,1].max()), float(existing_v[:,2].max())]

    # Shift marker vertices too
    for marker_v in [sv, cv, dv, rv]:
        marker_v[:, 0] -= sx
        marker_v[:, 1] -= sy

    # Rebuild marker binary chunks (they're at the end)
    # Need to rebuild from the existing bin + new marker data
    marker_bin_parts = []
    mbo = existing_bin_len
    while mbo % 4 != 0:
        marker_bin_parts.append(b'\x00')
        mbo += 1

    for marker_verts, marker_faces, marker_cols in [(sv, sf, sc), (cv, cf, cc), (dv, df, dc), (rv, rf, rc)]:
        # positions
        raw = marker_verts.astype(np.float32).tobytes()
        marker_bin_parts.append(raw)
        mbo += len(raw)
        # indices
        raw = marker_faces.astype(np.uint32).flatten().tobytes()
        marker_bin_parts.append(raw)
        mbo += len(raw)
        # colors
        raw = marker_cols.astype(np.uint8).tobytes()
        marker_bin_parts.append(raw)
        mbo += len(raw)

    bin_data_out = bin_data + b''.join(marker_bin_parts)

    # Write
    write_glb(gltf, bin_data_out, OUTPUT_GLB)

    mb = OUTPUT_GLB.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"✓ {OUTPUT_GLB} ({mb:.1f} MB)")
    print(f"  Textured terrain+buildings: 1 primitive")
    print(f"  Marker primitives: {len(new_primitives)} (sundial, flame, disc, ring)")
    print(f"  Total primitives: {len(meshes[0]['primitives'])}")
    print(f"\n  Square center at XY origin")
    print(f"  🔴 Red pillar + 🟡 gold flame = Red Bird Sundial (红鸟日晷)")
    print(f"  🟤 Dark disc = plaza base")
    print(f"  🟠 Orange ring = Red Bird Square boundary (红鸟广场)")
    print(f"\nOpen https://gltf-viewer.donmccurdy.com/ and drag in:")
    print(f"  {OUTPUT_GLB}")
    print(f"{'='*60}")

    # Quick verification
    verify = parse_glb(OUTPUT_GLB)
    v_gltf = verify[0]
    vm = v_gltf['meshes'][0]
    print(f"\nVerification: {len(vm['primitives'])} primitives")
    for i, p in enumerate(vm['primitives']):
        attrs = list(p['attributes'].keys())
        vc = v_gltf['accessors'][p['attributes']['POSITION']]['count']
        print(f"  Prim {i}: {vc:,}v, attrs={attrs}")


if __name__ == '__main__':
    main()

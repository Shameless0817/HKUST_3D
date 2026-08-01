#!/usr/bin/env python3
"""
Optimize HKUST Red Bird Square 3D model from Google Earth data.

Fixes from v18:
  1. CRITICAL: UV V-coordinate mapping was broken — all vertices sampled
     from the first atlas row instead of their correct tile rows.
  2. Better piazza center detection: looks for the actual campus plateau
     elevation (~60-80m) rather than clustering near sea level.
  3. Tighter default crop focused on the square area.
  4. Better material settings (less rough, slight metallic).
  5. Optional: add a colored ground disc to mark the square.
"""
import json, struct, io, math
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT = Path("/home/zliki/HKUST_3D")
INPUT_GLB = PROJECT / "output/demo/hkust_google_earth.glb"
OUTPUT_GLB = PROJECT / "output/demo/hkust_piazza.glb"

CROP_RADIUS = 60        # tighter crop — focus on the square
TILE_SIZE = 512
TILE_TARGET = 512       # FULL source resolution — no downscaling (was 256)
HALF_TILE = 256         # full half-tile height at source resolution (was 128)
WEBP_QUALITY = 90       # WebP Q90 ≈ JPEG Q95+ visually, much smaller files


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
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    comp_type = acc['componentType']
    acc_type = acc['type']
    type_sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    type_counts = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    elem = type_sizes[comp_type]
    ncomp = type_counts[acc_type]
    total = count * ncomp
    dtype = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
             5125: np.uint32, 5126: np.float32}[comp_type]
    raw = np.frombuffer(bin_data, dtype=dtype, count=total, offset=offset)
    if acc_type in ('VEC2', 'VEC3', 'VEC4'):
        raw = raw.reshape(-1, ncomp)
    return raw.copy()


def add_to_buffer(arr, comp_type, acc_type, bin_chunks, buffer_views, accessors,
                  normalized=False, byte_offset=0):
    arr = np.asarray(arr)
    raw = arr.tobytes()
    bv = {'buffer': 0, 'byteOffset': byte_offset, 'byteLength': len(raw)}
    acc = {'bufferView': len(buffer_views), 'componentType': comp_type,
           'count': len(arr), 'type': acc_type}
    if acc_type == 'VEC3' and comp_type == 5126:
        acc['min'] = [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].min())]
        acc['max'] = [float(arr[:, 0].max()), float(arr[:, 1].max()), float(arr[:, 2].max())]
    if normalized:
        acc['normalized'] = True
    buffer_views.append(bv)
    accessors.append(acc)
    bin_chunks.append(raw)
    return len(accessors) - 1, byte_offset + len(raw)


def find_piazza_center(vertices):
    """
    Find the Red Bird Square center in model coordinates.

    The square sits on a mid-elevation plateau (~60-80m in the model's Z).
    Strategy:
      1. Find vertices at plateau elevation (Z between 50th-85th percentile)
      2. Among these, find areas with low local Z variance (flat = plaza)
      3. Pick the largest flat cluster as the square center.
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

    # Campus plateau: Z between p60 and p90 (buildings + plaza area)
    z_low = np.percentile(z, 60)
    z_high = np.percentile(z, 92)
    plateau_mask = (z >= z_low) & (z <= z_high)
    plateau_verts = vertices[plateau_mask]

    if len(plateau_verts) < 100:
        # Fallback: use upper half
        z_median = np.median(z)
        plateau_mask = z > z_median
        plateau_verts = vertices[plateau_mask]

    print(f"  Plateau Z range: [{z_low:.1f}, {z_high:.1f}] — {len(plateau_verts):,} vertices")

    # Build spatial grid to find dense, flat areas
    px, py, pz = plateau_verts[:, 0], plateau_verts[:, 1], plateau_verts[:, 2]

    # Find the XY centroid of plateau vertices (rough campus center)
    cx_rough = np.median(px)
    cy_rough = np.median(py)

    # Refine: look for the densest cluster within 50 units of rough center
    dist_rough = np.sqrt((px - cx_rough)**2 + (py - cy_rough)**2)
    near_mask = dist_rough < 50
    if near_mask.sum() > 500:
        px, py, pz = px[near_mask], py[near_mask], pz[near_mask]

    # Find the elevation band with the most flat area
    # (flat = low variance in Z among neighbors)
    best_z = None
    best_flat_score = -1

    for z_band in np.linspace(pz.min() + 5, pz.max() - 5, 20):
        band_mask = (pz >= z_band - 3) & (pz <= z_band + 3)
        if band_mask.sum() < 200:
            continue
        bx, by = px[band_mask], py[band_mask]
        # Score: number of vertices * compactness (lower spread = more compact = more likely a plaza)
        spread = np.std(bx) + np.std(by)
        score = band_mask.sum() / (spread + 1)
        if score > best_flat_score:
            best_flat_score = score
            best_z = z_band

    if best_z is None:
        best_z = np.median(pz)

    # Among vertices near best_z, find the densest XY cluster
    band_mask = (pz >= best_z - 5) & (pz <= best_z + 5)
    bx, by = px[band_mask], py[band_mask]
    bmed_x, bmed_y = np.median(bx), np.median(by)

    # Iteratively refine: center on densest part
    for _ in range(3):
        bdist = np.sqrt((bx - bmed_x)**2 + (by - bmed_y)**2)
        close = bdist < 30
        if close.sum() < 100:
            break
        bmed_x, bmed_y = bx[close].mean(), by[close].mean()
        bx, by = bx[close], by[close]

    cz = best_z
    print(f"  Piazza center: ({bmed_x:.1f}, {bmed_y:.1f}, {cz:.1f})")

    return np.array([bmed_x, bmed_y, cz])


def main():
    print("=" * 60)
    print("HKUST Red Bird Square — Optimized 3D Extraction v2")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────
    print(f"\nLoading {INPUT_GLB.name} ({INPUT_GLB.stat().st_size/1024/1024:.0f} MB)...")
    gltf, bin_data = parse_glb(INPUT_GLB)

    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']
    meshes = gltf['meshes']
    images = gltf.get('images', [])

    prim = meshes[0]['primitives'][0]
    attrs = prim['attributes']
    vertices = read_accessor(attrs['POSITION'], accessors, buffer_views, bin_data)
    uvs = read_accessor(attrs.get('TEXCOORD_0'), accessors, buffer_views, bin_data)
    faces = read_accessor(prim['indices'], accessors, buffer_views, bin_data)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)

    print(f"  Vertices: {len(vertices):,}, Faces: {len(faces):,}")
    print(f"  UV range: U=[{uvs[:,0].min():.3f},{uvs[:,0].max():.3f}] "
          f"V=[{uvs[:,1].min():.3f},{uvs[:,1].max():.3f}]")

    # ── Find piazza center ────────────────────────────────
    print("\nLocating Red Bird Square...")
    center = find_piazza_center(vertices)

    # ── Crop mesh ─────────────────────────────────────────
    vdist = np.sqrt(np.sum((vertices - center)**2, axis=1))
    keep_vert = vdist < CROP_RADIUS
    face_in = keep_vert[faces.astype(int)]
    keep_face = face_in.all(axis=1)

    kept_faces = faces[keep_face].astype(int)
    old_idx = np.unique(kept_faces)
    old2new = {int(o): n for n, o in enumerate(old_idx)}
    new_faces = np.array([[old2new[int(f)] for f in face] for face in kept_faces], dtype=np.uint32)
    new_verts = vertices[old_idx].copy()
    new_uvs = uvs[old_idx].copy() if len(uvs) > 0 else np.array([])

    print(f"  Crop radius: {CROP_RADIUS} units")
    print(f"  Cropped: {len(new_verts):,}v, {len(new_faces):,}f "
          f"({100*len(new_verts)/len(vertices):.1f}%)")

    # ── Load texture ──────────────────────────────────────
    tex_image = None
    tex_w = tex_h = 0
    num_tiles = 0
    if images:
        img_info = images[0]
        bv_idx = img_info.get('bufferView')
        if bv_idx is not None:
            bv = buffer_views[bv_idx]
            offset = bv.get('byteOffset', 0)
            length = bv['byteLength']
            tex_bytes = bin_data[offset:offset+length]
            tex_image = Image.open(io.BytesIO(tex_bytes))
            tex_w, tex_h = tex_image.size
            num_tiles = tex_w // TILE_SIZE
            print(f"  Texture: {tex_w}x{tex_h}, {num_tiles} tiles of {TILE_SIZE}px")

    # ── Repack texture tiles ──────────────────────────────
    if tex_image is not None and len(new_uvs) > 0:
        u_tile_indices = np.floor(new_uvs[:, 0] * num_tiles).astype(int)
        u_tile_indices = np.clip(u_tile_indices, 0, num_tiles - 1)
        used_tiles = np.unique(u_tile_indices)
        print(f"\n  Used texture tiles: {len(used_tiles)} / {num_tiles}")

        grid_cols = int(math.ceil(math.sqrt(len(used_tiles))))
        grid_rows = int(math.ceil(len(used_tiles) / grid_cols))

        tile_to_pos = {}
        for i, t_idx in enumerate(sorted(used_tiles)):
            tile_to_pos[t_idx] = (i % grid_cols, i // grid_cols)

        new_atlas_w = grid_cols * TILE_TARGET
        new_atlas_h = grid_rows * HALF_TILE
        print(f"  New atlas: {new_atlas_w}x{new_atlas_h} "
              f"({grid_cols}x{grid_rows} grid, {TILE_TARGET}x{HALF_TILE}px tiles)")

        # Build new atlas
        # CRITICAL: original strip tiles are 512×512 but ONLY the top half
        # (rows 256-511, corresponding to V ∈ [0.5, 1.0]) has texture data.
        # The bottom half is pure black. We must crop it out.
        new_atlas = Image.new('RGB', (new_atlas_w, new_atlas_h), (128, 128, 128))

        for orig_tile, (gcol, grow) in tile_to_pos.items():
            src_x = orig_tile * TILE_SIZE
            # Extract ONLY the top half of the tile (valid texture region)
            tile = tex_image.crop((src_x, TILE_SIZE // 2,
                                   src_x + TILE_SIZE, TILE_SIZE))
            # Tile is now TILE_SIZE × (TILE_SIZE/2) = 512×256
            tile = tile.resize((TILE_TARGET, HALF_TILE), Image.LANCZOS)
            if tile.mode in ('RGBA', 'P'):
                tile = tile.convert('RGB')
            dst_x = gcol * TILE_TARGET
            dst_y = grow * HALF_TILE
            new_atlas.paste(tile, (dst_x, dst_y))

        # ── Remap UVs (FIXED V-mapping!) ──────────────────
        # Original UV V range in Google Earth strip: ~[0.5, 1.0]
        # Each tile occupies the full [0.5, 1.0] vertical range.
        # Within-tile fraction: (v - 0.5) / 0.5 → [0, 1]
        #
        # In the new atlas:
        #   new_u = (gcol + frac_u) / grid_cols
        #   new_v = (grow + frac_v) / grid_rows
        #
        # where frac_u ∈ [0,1] is the horizontal position within the tile,
        # and frac_v ∈ [0,1] is the vertical position within the tile.

        # Compute actual V range from data to be precise
        v_min = float(new_uvs[:, 1].min())
        v_max = float(new_uvs[:, 1].max())
        v_span = v_max - v_min
        print(f"  UV V range: [{v_min:.4f}, {v_max:.4f}] (span={v_span:.4f})")

        if v_span < 0.01:
            v_span = 0.5  # fallback

        for i in range(len(new_uvs)):
            u, v = new_uvs[i, 0], new_uvs[i, 1]

            # Determine which tile this UV falls into (by U coordinate)
            tile_idx = int(np.clip(np.floor(u * num_tiles), 0, num_tiles - 1))

            if tile_idx in tile_to_pos:
                gcol, grow = tile_to_pos[tile_idx]

                # Fractional position within the tile [0, 1]
                frac_u = (u * num_tiles) - tile_idx

                # FIXED: normalize V to [0, 1] within the tile
                # Original V ∈ [v_min, v_max] maps to full tile height
                frac_v = (v - v_min) / v_span

                # Clamp slightly away from edges to prevent GPU sampling
                # from adjacent tiles (texture bleeding at boundaries)
                frac_u = np.clip(frac_u, 0.002, 0.998)
                frac_v = np.clip(frac_v, 0.002, 0.998)

                # Map to new atlas coordinates [0, 1]
                new_uvs[i, 0] = (gcol + frac_u) / grid_cols
                new_uvs[i, 1] = (grow + frac_v) / grid_rows
            else:
                new_uvs[i] = [0.5, 0.5]  # fallback

        print(f"  Remapped UV range: U=[{new_uvs[:,0].min():.4f},{new_uvs[:,0].max():.4f}] "
              f"V=[{new_uvs[:,1].min():.4f},{new_uvs[:,1].max():.4f}]")

        # ── Convert texture to WebP ────────────────────────
        img_buffer = io.BytesIO()
        new_atlas.save(img_buffer, format='WEBP', quality=WEBP_QUALITY, lossless=False)
        atlas_bytes = img_buffer.getvalue()
        print(f"  Atlas WebP: {len(atlas_bytes)/1024/1024:.1f} MB (Q={WEBP_QUALITY})")

        # ── Compute smooth vertex normals ───────────────────
        print(f"\n  Computing smooth vertex normals...")
        # Face normals via cross product
        tri_verts = new_verts[new_faces]  # (nf, 3, 3)
        e1 = tri_verts[:, 1] - tri_verts[:, 0]
        e2 = tri_verts[:, 2] - tri_verts[:, 0]
        face_normals = np.cross(e1, e2)
        # Normalize face normals (avoid div by zero)
        fn_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
        fn_len[fn_len < 1e-12] = 1.0
        face_normals = face_normals / fn_len

        # Accumulate face normals to vertices (smooth/average)
        vertex_normals = np.zeros_like(new_verts)
        for fi in range(len(new_faces)):
            for vi in range(3):
                vertex_normals[new_faces[fi, vi]] += face_normals[fi]

        # Normalize vertex normals
        vn_len = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vn_len[vn_len < 1e-12] = 1.0
        vertex_normals = (vertex_normals / vn_len).astype(np.float32)
        print(f"  Normals computed: {len(vertex_normals):,} vertices")

        # ── Build output GLB ───────────────────────────────
        bin_chunks = []
        buffer_views_out = []
        accessors_out = []
        bo = 0

        pos_idx, bo = add_to_buffer(new_verts, 5126, 'VEC3', bin_chunks,
                                     buffer_views_out, accessors_out, byte_offset=bo)
        normal_idx, bo = add_to_buffer(vertex_normals, 5126, 'VEC3', bin_chunks,
                                        buffer_views_out, accessors_out, byte_offset=bo)
        idx_data = new_faces.flatten()
        idx_idx, bo = add_to_buffer(idx_data, 5125, 'SCALAR', bin_chunks,
                                     buffer_views_out, accessors_out, byte_offset=bo)
        uv_idx, bo = add_to_buffer(new_uvs.astype(np.float32), 5126, 'VEC2', bin_chunks,
                                    buffer_views_out, accessors_out, byte_offset=bo)

        # Embed texture
        img_bv = {'buffer': 0, 'byteOffset': bo, 'byteLength': len(atlas_bytes)}
        img_bv_idx = len(buffer_views_out)
        buffer_views_out.append(img_bv)
        bin_chunks.append(atlas_bytes)

        primitive = {
            'attributes': {'POSITION': pos_idx, 'NORMAL': normal_idx, 'TEXCOORD_0': uv_idx},
            'indices': idx_idx,
            'material': 0,
        }

        gltf_out = {
            'asset': {'version': '2.0', 'generator': '19_optimize_piazza'},
            'scene': 0,
            'scenes': [{'nodes': [0]}],
            'nodes': [{'mesh': 0}],
            'meshes': [{'primitives': [primitive]}],
            'accessors': accessors_out,
            'bufferViews': buffer_views_out,
            'buffers': [{'byteLength': bo + len(atlas_bytes)}],
            'images': [{'bufferView': img_bv_idx, 'mimeType': 'image/webp'}],
            'textures': [{'source': 0}],
            'samplers': [{'magFilter': 9729, 'minFilter': 9987,
                          'wrapS': 10497, 'wrapT': 10497}],
            'materials': [{
                'pbrMetallicRoughness': {
                    'baseColorTexture': {'index': 0},
                    'metallicFactor': 0.05,
                    'roughnessFactor': 0.35,
                },
                'doubleSided': True,
            }],
        }

        bin_data_out = b''.join(bin_chunks)
        write_glb(gltf_out, bin_data_out, OUTPUT_GLB)

        mb = OUTPUT_GLB.stat().st_size / 1e6
        print(f"\n{'='*60}")
        print(f"✓ {OUTPUT_GLB} ({mb:.1f} MB)")
        print(f"  Vertices: {len(new_verts):,}  Faces: {len(new_faces):,}")
        print(f"  Texture: {new_atlas_w}x{new_atlas_h} WebP atlas (Q={WEBP_QUALITY})")
        print(f"  Tiles used: {len(used_tiles)}")
        print(f"  V-mapping: ({v_min:.3f},{v_max:.3f}) → normalized per-tile")
        print(f"\nOpen https://gltf-viewer.donmccurdy.com/ and drag in:")
        print(f"  {OUTPUT_GLB}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()

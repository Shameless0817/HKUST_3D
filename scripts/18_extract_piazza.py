#!/usr/bin/env python3
"""
Extract HKUST Red Bird Square from Google Earth photorealistic 3D model.

The Google Earth model stores textures as a horizontal strip of tiles
(e.g., 524288x512 = 1024 tiles of 512x512). This script:
1. Loads the model
2. Identifies the piazza center
3. Crops the 3D mesh to the piazza area
4. Extracts only the texture tiles actually used by the cropped mesh
5. Repacks tiles into a square atlas, remaps UVs
6. Exports a compact GLB with embedded JPEG textures
"""
import json, struct, io, math
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT = Path("/home/zliki/HKUST_3D")
INPUT_GLB = PROJECT / "output/demo/hkust_google_earth.glb"
OUTPUT_GLB = PROJECT / "output/demo/hkust_piazza.glb"

CROP_RADIUS = 70  # model units
TILE_SIZE = 512   # each texture tile is 512x512 in the atlas strip
TILE_TARGET = 256 # resize tiles to this size for the output atlas


def parse_glb(path):
    """Parse a GLB file into glTF JSON and binary buffer."""
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
    """Write glTF JSON + binary buffer to .glb."""
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
    """Read data from a glTF accessor."""
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
    """Add data to GLB binary buffer, return accessor index and new byte_offset."""
    arr = np.asarray(arr)
    raw = arr.tobytes()
    bv = {'buffer': 0, 'byteOffset': byte_offset, 'byteLength': len(raw)}
    acc = {'bufferView': len(buffer_views), 'componentType': comp_type,
           'count': len(arr), 'type': acc_type}
    if acc_type == 'VEC3' and comp_type == 5126:
        acc['min'] = [float(arr[:,0].min()), float(arr[:,1].min()), float(arr[:,2].min())]
        acc['max'] = [float(arr[:,0].max()), float(arr[:,1].max()), float(arr[:,2].max())]
    if normalized:
        acc['normalized'] = True
    buffer_views.append(bv)
    accessors.append(acc)
    bin_chunks.append(raw)
    return len(accessors) - 1, byte_offset + len(raw)


def main():
    print("=" * 60)
    print("HKUST Red Bird Square — Texture-Preserving Crop")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────
    print(f"\nLoading {INPUT_GLB.name} ({INPUT_GLB.stat().st_size/1024/1024:.0f} MB)...")
    gltf, bin_data = parse_glb(INPUT_GLB)

    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']
    meshes = gltf['meshes']
    images = gltf.get('images', [])

    print(f"  Meshes: {len(meshes)}, Images: {len(images)}")

    # Extract mesh data from first primitive
    prim = meshes[0]['primitives'][0]
    attrs = prim['attributes']
    vertices = read_accessor(attrs['POSITION'], accessors, buffer_views, bin_data)
    uvs = read_accessor(attrs.get('TEXCOORD_0'), accessors, buffer_views, bin_data)
    faces = read_accessor(prim['indices'], accessors, buffer_views, bin_data)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)

    print(f"  Vertices: {len(vertices):,}, Faces: {len(faces):,}")
    if len(uvs) > 0:
        print(f"  UV range: U=[{uvs[:,0].min():.3f},{uvs[:,0].max():.3f}] "
              f"V=[{uvs[:,1].min():.3f},{uvs[:,1].max():.3f}]")

    # Load texture image
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
            print(f"  Texture: {tex_w}x{tex_h}, ~{num_tiles} tiles of {TILE_SIZE}px")

    # ── Find piazza center ────────────────────────────────
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    z_cutoff = np.percentile(z, 15)
    mask = z > z_cutoff
    med_x, med_y = np.median(x[mask]), np.median(y[mask])
    dist = np.sqrt((x[mask] - med_x)**2 + (y[mask] - med_y)**2)
    cluster = dist < 40
    cx, cy, cz = x[mask][cluster].mean(), y[mask][cluster].mean(), z[mask][cluster].mean()
    print(f"\n  Piazza center: ({cx:.1f}, {cy:.1f}, {cz:.1f})")

    # ── Crop mesh ─────────────────────────────────────────
    center = np.array([cx, cy, cz])
    vdist = np.sqrt(np.sum((vertices - center)**2, axis=1))
    keep_vert = vdist < CROP_RADIUS
    face_in = keep_vert[faces.astype(int)]
    keep_face = face_in.all(axis=1)

    kept_faces = faces[keep_face].astype(int)

    # Remap vertices
    old_idx = np.unique(kept_faces)
    old2new = {int(o): n for n, o in enumerate(old_idx)}
    new_faces = np.array([[old2new[int(f)] for f in face] for face in kept_faces], dtype=np.uint32)
    new_verts = vertices[old_idx]
    new_uvs = uvs[old_idx] if len(uvs) > 0 else np.array([])

    print(f"  Cropped: {len(new_verts):,}v, {len(new_faces):,}f "
          f"({100*len(new_verts)/len(vertices):.1f}%)")

    # ── Repack texture tiles ──────────────────────────────
    if tex_image is not None and len(new_uvs) > 0:
        # Determine which tiles are used by the cropped mesh
        u_tile_indices = np.floor(new_uvs[:, 0] * num_tiles).astype(int)
        u_tile_indices = np.clip(u_tile_indices, 0, num_tiles - 1)
        used_tiles = np.unique(u_tile_indices)
        print(f"\n  Used texture tiles: {len(used_tiles)} / {num_tiles}")

        # Compute grid layout
        grid_cols = int(math.ceil(math.sqrt(len(used_tiles))))
        grid_rows = int(math.ceil(len(used_tiles) / grid_cols))

        # Build tile index map: original tile → (col, row) in new atlas
        tile_to_pos = {}
        for i, t_idx in enumerate(sorted(used_tiles)):
            col = i % grid_cols
            row = i // grid_cols
            tile_to_pos[t_idx] = (col, row)

        new_atlas_w = grid_cols * TILE_TARGET
        new_atlas_h = grid_rows * TILE_TARGET
        print(f"  New atlas: {new_atlas_w}x{new_atlas_h} "
              f"({grid_cols}x{grid_rows} grid, {TILE_TARGET}px tiles)")

        # Build new atlas
        new_atlas = Image.new('RGB', (new_atlas_w, new_atlas_h), (128, 128, 128))

        for orig_tile, (gcol, grow) in tile_to_pos.items():
            # Extract tile from original strip
            src_x = orig_tile * TILE_SIZE
            tile = tex_image.crop((src_x, 0, src_x + TILE_SIZE, TILE_SIZE))
            if TILE_SIZE != TILE_TARGET:
                tile = tile.resize((TILE_TARGET, TILE_TARGET), Image.LANCZOS)
            if tile.mode in ('RGBA', 'P'):
                tile = tile.convert('RGB')
            dst_x = gcol * TILE_TARGET
            dst_y = grow * TILE_TARGET
            new_atlas.paste(tile, (dst_x, dst_y))

        # Remap UVs
        new_uvs = new_uvs.copy()
        for i in range(len(new_uvs)):
            u = new_uvs[i, 0]
            tile_idx = int(np.clip(u * num_tiles, 0, num_tiles - 1))
            if tile_idx in tile_to_pos:
                gcol, grow = tile_to_pos[tile_idx]
                # Fractional position within the tile
                frac_u = (u * num_tiles) - tile_idx
                # Map to new atlas coordinates [0,1]
                new_u = (gcol + frac_u) / grid_cols
                new_v = new_uvs[i, 1] / grid_rows  # V: each row is 1 tile high
                new_uvs[i, 0] = new_u
                new_uvs[i, 1] = new_v
            else:
                new_uvs[i] = [0.5, 0.5]  # fallback

        # Convert atlas to JPEG
        img_buffer = io.BytesIO()
        new_atlas.save(img_buffer, format='JPEG', quality=85)
        atlas_bytes = img_buffer.getvalue()
        print(f"  Atlas JPEG: {len(atlas_bytes)/1024/1024:.1f} MB")

        # ── Build output GLB ───────────────────────────────
        bin_chunks = []
        buffer_views_out = []
        accessors_out = []
        bo = 0

        pos_idx, bo = add_to_buffer(new_verts, 5126, 'VEC3', bin_chunks,
                                     buffer_views_out, accessors_out, byte_offset=bo)
        idx_data = new_faces.flatten()
        idx_idx, bo = add_to_buffer(idx_data, 5125, 'SCALAR', bin_chunks,
                                     buffer_views_out, accessors_out, byte_offset=bo)
        uv_idx, bo = add_to_buffer(new_uvs.astype(np.float32), 5126, 'VEC2', bin_chunks,
                                    buffer_views_out, accessors_out, byte_offset=bo)

        # Add texture
        img_bv = {'buffer': 0, 'byteOffset': bo, 'byteLength': len(atlas_bytes)}
        img_bv_idx = len(buffer_views_out)
        buffer_views_out.append(img_bv)
        bin_chunks.append(atlas_bytes)

        attributes = {'POSITION': pos_idx, 'TEXCOORD_0': uv_idx}
        primitive = {'attributes': attributes, 'indices': idx_idx, 'material': 0}

        gltf_out = {
            'asset': {'version': '2.0', 'generator': '18_extract_piazza'},
            'scene': 0,
            'scenes': [{'nodes': [0]}],
            'nodes': [{'mesh': 0}],
            'meshes': [{'primitives': [primitive]}],
            'accessors': accessors_out,
            'bufferViews': buffer_views_out,
            'buffers': [{'byteLength': bo + len(atlas_bytes)}],
            'images': [{'bufferView': img_bv_idx, 'mimeType': 'image/jpeg'}],
            'textures': [{'source': 0}],
            'samplers': [{'magFilter': 9729, 'minFilter': 9987,
                          'wrapS': 10497, 'wrapT': 10497}],
            'materials': [{
                'pbrMetallicRoughness': {
                    'baseColorTexture': {'index': 0},
                    'metallicFactor': 0.0,
                    'roughnessFactor': 1.0,
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
        print(f"  Texture: {new_atlas_w}x{new_atlas_h} JPEG atlas")
        print(f"  Tiles used: {len(used_tiles)}")
        print(f"\nOpen https://gltf-viewer.donmccurdy.com/ and drag in:")
        print(f"  {OUTPUT_GLB}")
        print(f"{'='*60}")
    else:
        print("  No texture — exporting vertex-only mesh")
        # ... vertex-only export (simplified)
        bin_chunks = []
        buffer_views_out = []
        accessors_out = []
        bo = 0
        pos_idx, bo = add_to_buffer(new_verts, 5126, 'VEC3', bin_chunks,
                                     buffer_views_out, accessors_out, byte_offset=bo)
        idx_data = new_faces.flatten()
        idx_idx, bo = add_to_buffer(idx_data, 5125, 'SCALAR', bin_chunks,
                                     buffer_views_out, accessors_out, byte_offset=bo)
        gltf_out = {
            'asset': {'version': '2.0'},
            'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
            'meshes': [{'primitives': [{'attributes': {'POSITION': pos_idx},
                                        'indices': idx_idx}]}],
            'accessors': accessors_out, 'bufferViews': buffer_views_out,
            'buffers': [{'byteLength': bo}],
        }
        write_glb(gltf_out, b''.join(bin_chunks), OUTPUT_GLB)
        mb = OUTPUT_GLB.stat().st_size / 1e6
        print(f"\n✓ {OUTPUT_GLB} ({mb:.1f} MB) — no textures")


if __name__ == '__main__':
    main()

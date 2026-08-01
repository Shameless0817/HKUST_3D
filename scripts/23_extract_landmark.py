#!/usr/bin/env python3
"""
Parameterized campus landmark extractor.
Extracts a region from the Google Earth GLB, repacks textures,
adds smooth normals, and exports an optimized GLB.

Usage:
  python3 scripts/23_extract_landmark.py --name academic --cx 5 --cy -35 --cz 68 --radius 85
  python3 scripts/23_extract_landmark.py --name seaside --cx 35 --cy 22 --cz 8 --radius 80
  python3 scripts/23_extract_landmark.py --name atrium --cx -2 --cy -18 --cz 45 --radius 55
"""
import json, struct, io, math, argparse
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT = Path("/home/zliki/HKUST_3D")
INPUT_GLB = PROJECT / "output/demo/hkust_google_earth.glb"
OUTPUT_DIR = PROJECT / "output/demo"

TILE_SIZE = 512
TILE_TARGET = 512
HALF_TILE = 256
WEBP_QUALITY = 90


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
                  byte_offset=0):
    arr = np.asarray(arr)
    raw = arr.tobytes()
    bv = {'buffer': 0, 'byteOffset': byte_offset, 'byteLength': len(raw)}
    acc = {'bufferView': len(buffer_views), 'componentType': comp_type,
           'count': len(arr), 'type': acc_type}
    if acc_type == 'VEC3' and comp_type == 5126:
        acc['min'] = [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 2].min())]
        acc['max'] = [float(arr[:, 0].max()), float(arr[:, 1].max()), float(arr[:, 2].max())]
    buffer_views.append(bv)
    accessors.append(acc)
    bin_chunks.append(raw)
    return len(accessors) - 1, byte_offset + len(raw)


def extract_landmark(name, center, radius):
    output_path = OUTPUT_DIR / f"hkust_{name}.glb"

    print(f"\n{'='*60}")
    print(f"Extracting: {name}")
    print(f"  Center: ({center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f})")
    print(f"  Radius: {radius}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────
    print(f"\nLoading source model...")
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

    # ── Crop ──────────────────────────────────────────────
    vdist = np.sqrt(np.sum((vertices - center)**2, axis=1))
    keep_vert = vdist < radius
    face_in = keep_vert[faces.astype(int)]
    keep_face = face_in.all(axis=1)

    kept_faces = faces[keep_face].astype(int)
    old_idx = np.unique(kept_faces)
    old2new = {int(o): n for n, o in enumerate(old_idx)}
    new_faces = np.array([[old2new[int(f)] for f in face] for face in kept_faces], dtype=np.uint32)
    new_verts = vertices[old_idx].copy()
    new_uvs = uvs[old_idx].copy() if len(uvs) > 0 else np.array([])

    print(f"  Cropped: {len(new_verts):,}v, {len(new_faces):,}f "
          f"({100*len(new_verts)/len(vertices):.1f}%)")

    if len(new_verts) < 100:
        print("  ⚠ Too few vertices! Check center/radius.")
        return

    # ── Load texture ──────────────────────────────────────
    tex_image = None
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
            print(f"  Texture: {tex_w}x{tex_h}, {num_tiles} tiles")

    # ── Repack texture ────────────────────────────────────
    if tex_image is not None and len(new_uvs) > 0:
        u_tile_indices = np.floor(new_uvs[:, 0] * num_tiles).astype(int)
        u_tile_indices = np.clip(u_tile_indices, 0, num_tiles - 1)
        used_tiles = np.unique(u_tile_indices)
        print(f"  Used tiles: {len(used_tiles)} / {num_tiles}")

        grid_cols = int(math.ceil(math.sqrt(len(used_tiles))))
        grid_rows = int(math.ceil(len(used_tiles) / grid_cols))

        tile_to_pos = {}
        for i, t_idx in enumerate(sorted(used_tiles)):
            tile_to_pos[t_idx] = (i % grid_cols, i // grid_cols)

        new_atlas_w = grid_cols * TILE_TARGET
        new_atlas_h = grid_rows * HALF_TILE
        new_atlas = Image.new('RGB', (new_atlas_w, new_atlas_h), (128, 128, 128))

        for orig_tile, (gcol, grow) in tile_to_pos.items():
            src_x = orig_tile * TILE_SIZE
            tile = tex_image.crop((src_x, TILE_SIZE // 2,
                                   src_x + TILE_SIZE, TILE_SIZE))
            tile = tile.resize((TILE_TARGET, HALF_TILE), Image.LANCZOS)
            if tile.mode in ('RGBA', 'P'):
                tile = tile.convert('RGB')
            dst_x = gcol * TILE_TARGET
            dst_y = grow * HALF_TILE
            new_atlas.paste(tile, (dst_x, dst_y))

        # ── Remap UVs ──────────────────────────────────────
        v_min = float(new_uvs[:, 1].min())
        v_max = float(new_uvs[:, 1].max())
        v_span = v_max - v_min
        if v_span < 0.01:
            v_span = 0.5

        for i in range(len(new_uvs)):
            u, v = new_uvs[i, 0], new_uvs[i, 1]
            tile_idx = int(np.clip(np.floor(u * num_tiles), 0, num_tiles - 1))
            if tile_idx in tile_to_pos:
                gcol, grow = tile_to_pos[tile_idx]
                frac_u = (u * num_tiles) - tile_idx
                frac_v = (v - v_min) / v_span
                frac_u = np.clip(frac_u, 0.002, 0.998)
                frac_v = np.clip(frac_v, 0.002, 0.998)
                new_uvs[i, 0] = (gcol + frac_u) / grid_cols
                new_uvs[i, 1] = (grow + frac_v) / grid_rows
            else:
                new_uvs[i] = [0.5, 0.5]

        # ── Texture to WebP ────────────────────────────────
        img_buffer = io.BytesIO()
        new_atlas.save(img_buffer, format='WEBP', quality=WEBP_QUALITY, lossless=False)
        atlas_bytes = img_buffer.getvalue()
        print(f"  Atlas: {new_atlas_w}x{new_atlas_h} WebP Q{WEBP_QUALITY}, {len(atlas_bytes)/1024:.0f}KB")

        # ── Compute smooth normals ─────────────────────────
        tri_verts = new_verts[new_faces]
        e1 = tri_verts[:, 1] - tri_verts[:, 0]
        e2 = tri_verts[:, 2] - tri_verts[:, 0]
        face_normals = np.cross(e1, e2)
        fn_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
        fn_len[fn_len < 1e-12] = 1.0
        face_normals = face_normals / fn_len

        vertex_normals = np.zeros_like(new_verts)
        for fi in range(len(new_faces)):
            for vi in range(3):
                vertex_normals[new_faces[fi, vi]] += face_normals[fi]
        vn_len = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vn_len[vn_len < 1e-12] = 1.0
        vertex_normals = (vertex_normals / vn_len).astype(np.float32)

        # Center model at origin
        model_center = new_verts.mean(axis=0)
        model_center[2] = new_verts[:, 2].min()  # Z=0 at ground
        new_verts[:, 0] -= model_center[0]
        new_verts[:, 1] -= model_center[1]
        new_verts[:, 2] -= model_center[2]

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
            'asset': {'version': '2.0', 'generator': '23_extract_landmark'},
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
        write_glb(gltf_out, bin_data_out, output_path)

        mb = output_path.stat().st_size / 1e6
        extent = new_verts.max(axis=0) - new_verts.min(axis=0)
        print(f"\n  ✓ {output_path.name} ({mb:.1f} MB)")
        print(f"    {len(new_verts):,}v, {len(new_faces):,}f")
        print(f"    Extent: {extent[0]:.0f}x{extent[1]:.0f}x{extent[2]:.0f}")
        print(f"    Atlas: {new_atlas_w}x{new_atlas_h}, {len(used_tiles)} tiles")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract campus landmark from Google Earth GLB")
    parser.add_argument('--name', required=True, help='Output name (hkust_<name>.glb)')
    parser.add_argument('--cx', type=float, required=True, help='Center X')
    parser.add_argument('--cy', type=float, required=True, help='Center Y')
    parser.add_argument('--cz', type=float, required=True, help='Center Z')
    parser.add_argument('--radius', type=float, default=60, help='Crop radius (default: 60)')
    args = parser.parse_args()

    extract_landmark(args.name, np.array([args.cx, args.cy, args.cz]), args.radius)


if __name__ == '__main__':
    main()

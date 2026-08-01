#!/usr/bin/env python3
"""
Assemble individual CSDI GLB tiles into a single Scene GLB
preserving KTX2 textures — pure binary GLB manipulation.
"""
import struct, json
from pathlib import Path
import glob

TILES_DIR = Path("/home/zliki/HKUST_3D/output/csdi/f2/glb")
OUT_DIR = Path("/home/zliki/HKUST_3D/output/csdi/f2")
DEMO_DIR = Path("/home/zliki/HKUST_3D/output/demo")

def parse_glb(path):
    """Parse a GLB file into JSON + binary buffer."""
    if not isinstance(path, Path):
        path = Path(path)
    data = path.read_bytes()
    assert data[:4] == b"glTF", f"Bad magic in {path}"
    version = struct.unpack('<I', data[4:8])[0]
    total_len = struct.unpack('<I', data[8:12])[0]

    pos = 12
    json_data = None
    bin_data = b""

    while pos < total_len:
        chunk_len = struct.unpack('<I', data[pos:pos+4])[0]
        chunk_type = data[pos+4:pos+8].decode('ascii', errors='replace')
        chunk_data = data[pos+8:pos+8+chunk_len]
        if chunk_type == 'JSON':
            json_data = json.loads(chunk_data.decode('utf-8'))
        elif chunk_type == 'BIN\0':
            bin_data = chunk_data
        pos += 8 + chunk_len

    return json_data, bin_data

def build_glb(gltf_json, bin_data):
    """Build a GLB file from JSON and binary buffer."""
    json_str = json.dumps(gltf_json, separators=(',', ':'))
    # Pad JSON to 4-byte alignment with spaces (0x20)
    while len(json_str) % 4 != 0:
        json_str += ' '
    json_bytes = json_str.encode('utf-8')

    # Pad binary to 4-byte alignment with zeros
    while len(bin_data) % 4 != 0:
        bin_data += b'\x00'

    # GLB header
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_data)
    header = struct.pack('<III', 0x46546C67, 2, total_len)  # magic=glTF, version=2

    # JSON chunk
    json_chunk = struct.pack('<II', len(json_bytes), 0x4E4F534A) + json_bytes  # 'JSON'

    # BIN chunk
    bin_chunk = struct.pack('<II', len(bin_data), 0x004E4942) + bin_data  # 'BIN\0'

    return header + json_chunk + bin_chunk


def main():
    tiles = sorted(glob.glob(str(TILES_DIR / "*.glb")))
    print(f"Found {len(tiles)} tiles")

    # Step 1: Parse all tiles
    all_tile_data = []
    for tp in tiles:
        try:
            js, bin_chunk = parse_glb(tp)
            all_tile_data.append((Path(tp).name, js, bin_chunk))
        except Exception as e:
            print(f"  Skip {Path(tp).name}: {e}")

    print(f"Parsed {len(all_tile_data)} tiles")

    # Step 2: Build combined glTF JSON
    # These are the top-level arrays we need to merge
    collections = {
        'buffers': [],
        'bufferViews': [],
        'accessors': [],
        'meshes': [],
        'nodes': [],
        'materials': [],
        'textures': [],
        'images': [],
        'samplers': [],
        'scenes': [{'nodes': []}],  # single scene
    }

    combined_bin = b""
    # Track mappings: (tile_idx, old_idx) -> new_idx for each collection
    index_maps = {k: {} for k in ['bufferViews', 'accessors', 'meshes', 'nodes',
                                    'materials', 'textures', 'images', 'samplers']}

    for tile_idx, (name, js, bin_chunk) in enumerate(all_tile_data):
        # Figure out which mesh to use
        # The tile may have a scene with nodes referencing meshes
        tile_scene = js.get('scene', 0)
        tile_root_nodes = []
        if 'scenes' in js and js['scenes']:
            s = js['scenes'][tile_scene] if tile_scene < len(js['scenes']) else js['scenes'][0]
            tile_root_nodes = s.get('nodes', [])

        # Collect all reachable nodes from root
        def collect_nodes(node_idx, visited):
            if node_idx in visited:
                return
            visited.add(node_idx)
            node = js['nodes'][node_idx]
            for child in node.get('children', []):
                collect_nodes(child, visited)

        all_node_indices = set()
        for rn in tile_root_nodes:
            collect_nodes(rn, all_node_indices)
        # Also include any nodes not in the hierarchy
        for i in range(len(js.get('nodes', []))):
            if i not in all_node_indices:
                all_node_indices.add(i)

        # --- Copy buffer data ---
        if js.get('buffers') and js['buffers'][0].get('byteLength', 0) > 0:
            old_buf_len = js['buffers'][0]['byteLength']
            actual_bin = bin_chunk[:old_buf_len]
            # Pad to 4-byte alignment
            while len(actual_bin) % 4 != 0:
                actual_bin += b'\x00'
            buf_offset = len(combined_bin)
            combined_bin += actual_bin
            # The new buffer will have byteLength = len(combined_bin), but each
            # bufferView will have its own byteOffset within the single buffer.

        # --- Copy bufferViews ---
        bv_map = {}  # old_idx -> new_idx
        if 'bufferViews' in js:
            for bv_idx, bv in enumerate(js['bufferViews']):
                new_bv = dict(bv)
                new_bv['byteOffset'] = bv.get('byteOffset', 0) + buf_offset
                collections['bufferViews'].append(new_bv)
                bv_map[bv_idx] = len(collections['bufferViews']) - 1
        index_maps['bufferViews'][tile_idx] = bv_map

        # --- Copy accessors ---
        acc_map = {}
        if 'accessors' in js:
            for acc_idx, acc in enumerate(js['accessors']):
                new_acc = dict(acc)
                if 'bufferView' in new_acc:
                    old_bv = new_acc['bufferView']
                    new_acc['bufferView'] = bv_map[old_bv]
                collections['accessors'].append(new_acc)
                acc_map[acc_idx] = len(collections['accessors']) - 1
        index_maps['accessors'][tile_idx] = acc_map

        # --- Copy images ---
        img_map = {}
        if 'images' in js:
            for img_idx, img in enumerate(js['images']):
                new_img = dict(img)
                if 'bufferView' in new_img:
                    new_img['bufferView'] = bv_map[new_img['bufferView']]
                collections['images'].append(new_img)
                img_map[img_idx] = len(collections['images']) - 1
        index_maps['images'][tile_idx] = img_map

        # --- Copy samplers ---
        sam_map = {}
        if 'samplers' in js:
            for sam_idx, sam in enumerate(js['samplers']):
                collections['samplers'].append(dict(sam))
                sam_map[sam_idx] = len(collections['samplers']) - 1
        index_maps['samplers'][tile_idx] = sam_map

        # --- Copy textures ---
        tex_map = {}
        if 'textures' in js:
            for tex_idx, tex in enumerate(js['textures']):
                new_tex = dict(tex)
                if 'source' in new_tex:
                    new_tex['source'] = img_map[new_tex['source']]
                if 'sampler' in new_tex and new_tex['sampler'] is not None:
                    new_tex['sampler'] = sam_map.get(new_tex['sampler'], new_tex['sampler'])
                collections['textures'].append(new_tex)
                tex_map[tex_idx] = len(collections['textures']) - 1
        index_maps['textures'][tile_idx] = tex_map

        # --- Copy materials ---
        mat_map = {}
        if 'materials' in js:
            for mat_idx, mat in enumerate(js['materials']):
                new_mat = {}
                for k, v in mat.items():
                    if k == 'pbrMetallicRoughness' and isinstance(v, dict):
                        new_pbr = dict(v)
                        if 'baseColorTexture' in new_pbr:
                            old_tex = new_pbr['baseColorTexture']['index']
                            new_pbr['baseColorTexture'] = dict(new_pbr['baseColorTexture'])
                            new_pbr['baseColorTexture']['index'] = tex_map[old_tex]
                        new_mat[k] = new_pbr
                    elif k == 'emissiveTexture' and isinstance(v, dict):
                        new_et = dict(v)
                        if 'index' in new_et:
                            new_et['index'] = tex_map[new_et['index']]
                        new_mat[k] = new_et
                    elif k == 'normalTexture' and isinstance(v, dict):
                        new_nt = dict(v)
                        if 'index' in new_nt:
                            new_nt['index'] = tex_map[new_nt['index']]
                        new_mat[k] = new_nt
                    elif k == 'occlusionTexture' and isinstance(v, dict):
                        new_ot = dict(v)
                        if 'index' in new_ot:
                            new_ot['index'] = tex_map[new_ot['index']]
                        new_mat[k] = new_ot
                    else:
                        new_mat[k] = v
                collections['materials'].append(new_mat)
                mat_map[mat_idx] = len(collections['materials']) - 1
        index_maps['materials'][tile_idx] = mat_map

        # --- Copy meshes ---
        mesh_map = {}
        if 'meshes' in js:
            for mesh_idx, mesh in enumerate(js['meshes']):
                new_mesh = {'name': f'{name}_{mesh_idx}', 'primitives': []}
                for prim in mesh.get('primitives', []):
                    new_prim = {'attributes': {}}
                    for attr_name, acc_idx in prim['attributes'].items():
                        new_prim['attributes'][attr_name] = acc_map[acc_idx]
                    if 'indices' in prim:
                        new_prim['indices'] = acc_map[prim['indices']]
                    if 'material' in prim and prim['material'] is not None:
                        new_prim['material'] = mat_map[prim['material']]
                    if 'mode' in prim:
                        new_prim['mode'] = prim['mode']
                    new_mesh['primitives'].append(new_prim)
                collections['meshes'].append(new_mesh)
                mesh_map[mesh_idx] = len(collections['meshes']) - 1
        index_maps['meshes'][tile_idx] = mesh_map

        # --- Copy nodes ---
        node_map = {}
        if 'nodes' in js:
            for node_idx in sorted(all_node_indices):
                if node_idx >= len(js['nodes']):
                    continue
                node = js['nodes'][node_idx]
                new_node = {'name': f'{name}_{node_idx}'}
                if 'mesh' in node and node['mesh'] is not None:
                    new_node['mesh'] = mesh_map[node['mesh']]
                if 'children' in node:
                    # Children will be remapped later
                    new_node['children'] = list(node['children'])
                if 'translation' in node:
                    new_node['translation'] = list(node['translation'])
                if 'rotation' in node:
                    new_node['rotation'] = list(node['rotation'])
                if 'scale' in node:
                    new_node['scale'] = list(node['scale'])
                if 'matrix' in node:
                    new_node['matrix'] = list(node['matrix'])
                collections['nodes'].append(new_node)
                node_map[node_idx] = len(collections['nodes']) - 1

            # Remap children references
            for node_idx in sorted(all_node_indices):
                if node_idx >= len(js['nodes']):
                    continue
                new_idx = node_map[node_idx]
                node = collections['nodes'][new_idx]
                if 'children' in node:
                    node['children'] = [node_map[c] for c in node['children'] if c in node_map]

            # Add root nodes to scene
            for rn in tile_root_nodes:
                if rn in node_map:
                    collections['scenes'][0]['nodes'].append(node_map[rn])

    # --- Final assembly ---
    # Check if samplers are actually referenced; remove if not
    if 'samplers' in collections:
        any_sampler_ref = False
        for tex in collections.get('textures', []):
            if tex.get('sampler') is not None:
                any_sampler_ref = True
                break
        if not any_sampler_ref:
            del collections['samplers']

    # Buffer
    combined_bin = combined_bin  # already padded

    out_json = {
        'asset': {'version': '2.0', 'generator': 'CSDI-HKUST-assembler'},
        'scene': 0,
        'scenes': collections['scenes'],
        'nodes': collections['nodes'],
        'meshes': collections['meshes'],
        'accessors': collections['accessors'],
        'bufferViews': collections['bufferViews'],
        'buffers': [{'byteLength': len(combined_bin)}],
        'materials': collections['materials'],
        'textures': collections['textures'],
        'images': collections['images'],
    }
    if 'samplers' in collections:
        out_json['samplers'] = collections['samplers']

    # Build GLB
    glb_data = build_glb(out_json, combined_bin)

    # Save
    out_path = OUT_DIR / "hkust_csdi_scene.glb"
    out_path.write_bytes(glb_data)
    print(f"\n✓ Scene GLB: {out_path} ({len(glb_data)/1e6:.1f} MB)")
    print(f"  {len(collections['nodes'])} nodes, {len(collections['meshes'])} meshes, "
          f"{len(collections['images'])} images, {len(collections['textures'])} textures")

    # Copy to demo
    demo_path = DEMO_DIR / "hkust_csdi_scene.glb"
    demo_path.write_bytes(glb_data)
    print(f"✓ Demo copy: {demo_path}")

    # --- Also save as merged (single mesh, needs vertex colors) ---
    # For now, let's also try a direct binary merge approach
    # where we extract the mesh data per-tile and create one big mesh
    # This requires decoding KTX2 → vertex colors

    # Verify the scene GLB
    js_check, _ = parse_glb(out_path)
    imgs = js_check.get('images', [])
    texs = js_check.get('textures', [])
    meshes = js_check.get('meshes', [])
    print(f"  Verify: {len(meshes)} meshes, {len(imgs)} images, {len(texs)} textures")
    for img in imgs[:2]:
        print(f"    image mimeType={img.get('mimeType', 'N/A')}, bufferView={img.get('bufferView', 'N/A')}")


if __name__ == "__main__":
    main()

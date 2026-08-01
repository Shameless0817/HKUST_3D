#!/usr/bin/env python3
"""For each POI, find the actual model surface Z at that XY position."""
import json, struct, io
from pathlib import Path
import numpy as np

PROJECT = Path("/home/zliki/HKUST_3D")

# Current POI positions from the HTML
POIS = {
    'piazza': [
        ('sundial', 0, 0),
        ('plaza', 0, 0),
        ('academic-wing', -18, -10),
    ],
    'academic': [
        ('arcade', 10, 10),
        ('main-entrance', -12, -5),
        ('lecture', 22, -12),
    ],
    'seaside': [
        ('track', 6, 25),
        ('promenade', 28, 15),
        ('pool', -20, -5),
    ],
    'atrium': [
        ('atrium-hall', -20, 8),
        ('entrance', -10, -5),
    ],
}

FILES = {
    'piazza': PROJECT / "output/demo/hkust_piazza.glb",
    'academic': PROJECT / "output/demo/hkust_academic.glb",
    'seaside': PROJECT / "output/demo/hkust_seaside.glb",
    'atrium': PROJECT / "output/demo/hkust_atrium.glb",
}

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

def read_acc(acc_idx, accessors, buffer_views, bin_data):
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

for name, path in FILES.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    gltf, bin_data = parse_glb(path)
    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']
    prim = gltf['meshes'][0]['primitives'][0]
    v = read_acc(prim['attributes']['POSITION'], accessors, buffer_views, bin_data)

    for poi_name, px, py in POIS[name]:
        # Find vertices near this XY
        dist = np.sqrt((v[:,0] - px)**2 + (v[:,1] - py)**2)
        nearby = dist < 5  # within 5 units
        if nearby.sum() < 10:
            nearby = dist < 10
        if nearby.sum() < 10:
            nearby = dist < 15

        if nearby.sum() > 0:
            nv = v[nearby]
            z_min = nv[:,2].min()
            z_max = nv[:,2].max()
            z_mean = nv[:,2].mean()
            z_p90 = np.percentile(nv[:,2], 90)
            z_p95 = np.percentile(nv[:,2], 95)

            # Find the surface: for the closest vertices (< 2 units), get max Z
            close = dist < 3
            if close.sum() > 5:
                surface_z = v[close, 2].max()
            else:
                surface_z = z_p90

            print(f"  {poi_name}: ({px},{py})")
            print(f"    Vertices nearby: {nearby.sum()} (within 5u)")
            print(f"    Z range: [{z_min:.1f}, {z_max:.1f}] mean={z_mean:.1f}")
            print(f"    Surface Z (nearby max): {surface_z:.1f}")
            print(f"    Suggested marker Z: {surface_z + 3:.1f} (surface + 3m)")
        else:
            print(f"  {poi_name}: ({px},{py}) — NO VERTICES NEARBY! Need different XY")

#!/usr/bin/env python3
"""Analyze each extracted GLB to find correct POI marker positions."""
import json, struct, io
from pathlib import Path
import numpy as np

PROJECT = Path("/home/zliki/HKUST_3D")
MODELS = {
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


for name, path in MODELS.items():
    print(f"\n{'='*70}")
    print(f"  {name}: {path.name} ({path.stat().st_size/1e6:.1f} MB)")
    print(f"{'='*70}")

    gltf, bin_data = parse_glb(path)
    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']
    meshes = gltf['meshes']

    # Get terrain primitive (prim 0, largest vertex count)
    prim = meshes[0]['primitives'][0]
    v = read_acc(prim['attributes']['POSITION'], accessors, buffer_views, bin_data)
    faces_acc = accessors[prim['indices']]
    fc = faces_acc['count'] // 3

    print(f"  Vertices: {len(v):,}  Faces: {fc:,}")
    print(f"  X: [{v[:,0].min():.1f}, {v[:,0].max():.1f}] span={v[:,0].max()-v[:,0].min():.0f}")
    print(f"  Y: [{v[:,1].min():.1f}, {v[:,1].max():.1f}] span={v[:,1].max()-v[:,1].min():.0f}")
    print(f"  Z: [{v[:,2].min():.1f}, {v[:,2].max():.1f}] span={v[:,2].max()-v[:,2].min():.0f}")

    z = v[:, 2]
    print(f"  Z percentiles: p5={np.percentile(z,5):.1f} p25={np.percentile(z,25):.1f} "
          f"p50={np.percentile(z,50):.1f} p75={np.percentile(z,75):.1f} "
          f"p90={np.percentile(z,90):.1f} p95={np.percentile(z,95):.1f}")

    # Find high-density XY clusters across elevation bands
    for lo, hi, label in [
        (0, np.percentile(z, 20), "LOW (ground)"),
        (np.percentile(z, 20), np.percentile(z, 60), "MID"),
        (np.percentile(z, 60), np.percentile(z, 100), "HIGH (rooftops)"),
    ]:
        band = v[(z >= lo) & (z < hi)]
        if len(band) < 100:
            continue

        # Grid density
        step = 8
        grid = {}
        bx, by = band[:, 0], band[:, 1]
        for i in range(len(band)):
            key = (int(bx[i] // step), int(by[i] // step))
            grid[key] = grid.get(key, 0) + 1

        top = sorted(grid.items(), key=lambda x: -x[1])[:6]
        print(f"\n  {label} (Z {lo:.0f}-{hi:.0f}):")
        for (gx, gy), cnt in top:
            wx, wy = gx*step+step/2, gy*step+step/2
            cell_v = band[(bx >= gx*step) & (bx < (gx+1)*step) & (by >= gy*step) & (by < (gy+1)*step)]
            avg_z = cell_v[:, 2].mean()
            print(f"    ({wx:6.0f}, {wy:6.0f}) Z={avg_z:.1f}: {cnt:>6,}v")

    # Find the highest point (potential rooftop / tower)
    z_max_idx = np.argmax(z)
    print(f"\n  Highest point: ({v[z_max_idx,0]:.1f}, {v[z_max_idx,1]:.1f}, {z[z_max_idx]:.1f})")

    # Find XY center of mass
    cx, cy = v[:, 0].mean(), v[:, 1].mean()
    print(f"  Center of mass XY: ({cx:.1f}, {cy:.1f})")
    print(f"  Center (median) XY: ({np.median(v[:,0]):.1f}, {np.median(v[:,1]):.1f})")

    # For each POI the user cares about, suggest coordinates
    # based on spatial analysis of landmarks
    print(f"\n  ── Suggested POI positions ──")
    if name == 'piazza':
        # The sundial is at the center (0,0) after centering in script 21
        # Ground Z is ~0 (min Z)
        ground_z = np.percentile(z, 2)
        print(f"  Ground Z ≈ {ground_z:.1f}")
        print(f"  Sundial center: (0, 0, {ground_z+8.5:.1f})  ← use this!")
        print(f"  Plaza center: (0, 0, {ground_z+0.5:.1f})")
        # Find a point on the academic building direction
        # Look at vertices at Z ~15-30 (building height)
        bldg = v[(z > 12) & (z < 30)]
        if len(bldg) > 100:
            bldg_center = bldg.mean(axis=0)
            print(f"  Building mass center: ({bldg_center[0]:.1f}, {bldg_center[1]:.1f}, {bldg_center[2]:.1f})")

    elif name == 'academic':
        # Academic building: find the arc structure
        # Look for elongated clusters at mid-high elevation
        ground_z = np.percentile(z, 2)
        print(f"  Ground Z ≈ {ground_z:.1f}")
        # Find top 3 dense areas as potential POIs
        mid_high = v[(z > np.percentile(z, 30)) & (z < np.percentile(z, 80))]
        if len(mid_high) > 100:
            print(f"  Mid-high building center: ({mid_high[:,0].mean():.1f}, {mid_high[:,1].mean():.1f}, {mid_high[:,2].mean():.1f})")

    elif name == 'seaside':
        ground_z = np.percentile(z, 2)
        print(f"  Ground Z ≈ {ground_z:.1f}")
        # Track/sports field: large flat area
        flat = v[(z > ground_z - 0.5) & (z < ground_z + 3)]
        if len(flat) > 100:
            print(f"  Flat ground center: ({flat[:,0].mean():.1f}, {flat[:,1].mean():.1f}, {flat[:,2].mean():.1f})")

    elif name == 'atrium':
        ground_z = np.percentile(z, 2)
        print(f"  Ground Z ≈ {ground_z:.1f}")
        # Atrium: mid-height dense area
        mid = v[(z > 5) & (z < 20)]
        if len(mid) > 100:
            print(f"  Mid-level center: ({mid[:,0].mean():.1f}, {mid[:,1].mean():.1f}, {mid[:,2].mean():.1f})")

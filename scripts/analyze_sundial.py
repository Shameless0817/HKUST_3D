#!/usr/bin/env python3
"""
Enhance piazza model: locate the Red Bird Sundial and add visual markers.
Also crops the model to center it for better default viewing.
"""
import json, struct, io
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PIAZZA = Path("/home/zliki/HKUST_3D/output/demo/hkust_piazza.glb")
OUTPUT = Path("/home/zliki/HKUST_3D/output/demo/hkust_piazza.glb")  # overwrite

pd = PIAZZA.read_bytes()
offset = 12
json_len = struct.unpack_from('<I', pd, offset)[0]
offset += 8
gltf = json.loads(pd[offset:offset+json_len])
offset += json_len
bin_len = struct.unpack_from('<I', pd, offset)[0]
offset += 8
bin_data = pd[offset:offset+bin_len]

accessors = gltf['accessors']
buffer_views = gltf['bufferViews']

def read_acc(acc_idx):
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

prim = gltf['meshes'][0]['primitives'][0]
attrs = prim['attributes']
v = read_acc(attrs['POSITION'])
uv = read_acc(attrs.get('TEXCOORD_0'))
faces = read_acc(prim['indices'])
if faces.ndim == 1:
    faces = faces.reshape(-1, 3)

# Find the sundial: tall, thin vertical structure at campus plateau elevation
# Look at face analysis — find faces that have significant Z range (vertical)
face_v = v[faces.astype(int)]
face_z_range = face_v[:, :, 2].max(axis=1) - face_v[:, :, 2].min(axis=1)
face_z_center = face_v[:, :, 2].mean(axis=1)
face_xy_center = face_v[:, :, :2].mean(axis=1)
face_area_xy = np.zeros(len(faces))
for i in range(len(faces)):
    tri_xy = face_v[i, :, :2]
    a = tri_xy[1] - tri_xy[0]
    b = tri_xy[2] - tri_xy[0]
    face_area_xy[i] = abs(np.cross(a, b)) * 0.5

# Sundial: tall (>3m Z range), small XY footprint (<2 m²), at plateau elevation (Z 50-80)
sundial_candidates = (face_z_range > 2) & (face_area_xy < 5) & (face_z_center > 45) & (face_z_center < 85)
print(f"Sundial-like faces (tall+thin+plateau): {sundial_candidates.sum()}")

if sundial_candidates.sum() > 0:
    sc_xy = face_xy_center[sundial_candidates]
    sc_z = face_z_center[sundial_candidates]
    sc_range = face_z_range[sundial_candidates]

    # Cluster them spatially
    from collections import Counter
    grid = np.round(sc_xy).astype(int)
    grid_tuples = [tuple(g) for g in grid]
    clusters = Counter(grid_tuples)

    print("\nTop sundial candidate clusters:")
    for (gx, gy), count in clusters.most_common(15):
        mask = (grid[:,0]==gx) & (grid[:,1]==gy)
        avg_z = sc_z[mask].mean()
        max_range = sc_range[mask].max()
        print(f"  ({gx:4d},{gy:4d}) n={count:3d} z={avg_z:.1f} max_range={max_range:.1f}")

# Also check: what's at the center of the model? The center should be near the square
cx, cy, cz = v[:,0].mean(), v[:,1].mean(), v[:,2].mean()
med_x, med_y, med_z = np.median(v[:,0]), np.median(v[:,1]), np.median(v[:,2])
print(f"\nModel center (mean): ({cx:.1f}, {cy:.1f}, {cz:.1f})")
print(f"Model center (median): ({med_x:.1f}, {med_y:.1f}, {med_z:.1f})")

# Find the densest flat area at plateau Z (55-65m) — this IS the square
plateau_mask = (v[:,2] > 50) & (v[:,2] < 70)
pv = v[plateau_mask]
if len(pv) > 100:
    # Find the densest XY cluster
    pmed_x, pmed_y = np.median(pv[:,0]), np.median(pv[:,1])
    for _ in range(3):
        pdist = np.sqrt((pv[:,0]-pmed_x)**2 + (pv[:,1]-pmed_y)**2)
        close = pdist < 20
        if close.sum() < 50:
            break
        pmed_x, pmed_y = pv[close, 0].mean(), pv[close, 1].mean()
        pv = pv[close]
    print(f"\nSquare center estimate: ({pmed_x:.1f}, {pmed_y:.1f})")
    print(f"  Vertices in cluster: {len(pv)}")

    # Look for low-variance areas (flat = plaza surface)
    nearby_mask = np.sqrt((v[:,0]-pmed_x)**2 + (v[:,1]-pmed_y)**2) < 30
    nearby = v[nearby_mask]
    # Check Z variance in small grid cells
    print(f"\n  Flatness analysis near square center:")
    for grid_x in np.arange(pmed_x-20, pmed_x+21, 10):
        for grid_y in np.arange(pmed_y-20, pmed_y+21, 10):
            cell = (abs(nearby[:,0] - grid_x) < 5) & (abs(nearby[:,1] - grid_y) < 5)
            if cell.sum() > 10:
                z_std = nearby[cell, 2].std()
                z_avg = nearby[cell, 2].mean()
                if z_std < 2.0:  # flat!
                    print(f"    FLAT at ({grid_x:.0f},{grid_y:.0f}): {cell.sum():4d}v z={z_avg:.1f}±{z_std:.1f}")

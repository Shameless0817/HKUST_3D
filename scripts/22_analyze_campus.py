#!/usr/bin/env python3
"""Analyze HKUST Google Earth model to locate key landmarks."""
import json, struct, io
from pathlib import Path
import numpy as np

PROJECT = Path("/home/zliki/HKUST_3D")
INPUT_GLB = PROJECT / "output/demo/hkust_google_earth.glb"

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

print("Loading model...")
gltf, bin_data = parse_glb(INPUT_GLB)
accessors = gltf['accessors']
buffer_views = gltf['bufferViews']
prim = gltf['meshes'][0]['primitives'][0]
v = read_acc(prim['attributes']['POSITION'], accessors, buffer_views, bin_data)
faces = read_acc(prim['indices'], accessors, buffer_views, bin_data)
if faces.ndim == 1:
    faces = faces.reshape(-1, 3)

print(f"Total: {len(v):,} vertices, {len(faces):,} faces")
print(f"X: [{v[:,0].min():.0f}, {v[:,0].max():.0f}]")
print(f"Y: [{v[:,1].min():.0f}, {v[:,1].max():.0f}]")
print(f"Z: [{v[:,2].min():.0f}, {v[:,2].max():.0f}]")

# Known: Red Bird Square center ~ (20.8, -19.4, 58.0)
# The square is in the campus plateau at Z ~55-65m
# Let's explore the full campus layout

# ── Elevation bands ──
z = v[:, 2]
print("\n=== Elevation Profile ===")
for lo, hi, label in [
    (0, 15, "Sea level (coastline)"),
    (15, 35, "Lower slope"),
    (35, 55, "Mid campus"),
    (55, 80, "Upper campus plateau"),
    (80, 110, "Mountain top"),
]:
    mask = (z >= lo) & (z < hi)
    count = mask.sum()
    print(f"  Z {lo:3d}-{hi:3d} ({label}): {count:>8,} vertices ({100*count/len(v):5.1f}%)")

# ── Spatial grid analysis at different elevation bands ──
def find_clusters(verts, grid_size=20, min_density=0.01):
    """Grid-based cluster detection in XY space."""
    x, y = verts[:, 0], verts[:, 1]
    x_bins = np.arange(x.min(), x.max() + grid_size, grid_size)
    y_bins = np.arange(y.min(), y.max() + grid_size, grid_size)

    cells = {}
    x_idx = np.digitize(x, x_bins) - 1
    y_idx = np.digitize(y, y_bins) - 1

    for i in range(len(verts)):
        key = (x_idx[i], y_idx[i])
        cells[key] = cells.get(key, 0) + 1

    # Find high-density cells
    total = len(verts)
    clusters = []
    for (xi, yi), count in cells.items():
        density = count / total
        if density > min_density:
            cx = x_bins[xi] + grid_size/2
            cy = y_bins[yi] + grid_size/2
            z_vals = verts[(x_idx == xi) & (y_idx == yi), 2]
            clusters.append((cx, cy, z_vals.mean(), count, density))

    clusters.sort(key=lambda c: -c[3])
    return clusters

# ── Plateau area (Z 50-85): Main campus buildings ──
print("\n=== Plateau Campus (Z 50-85): Key clusters ===")
plateau = v[(z >= 50) & (z < 85)]
clusters = find_clusters(plateau, grid_size=25)
for i, (cx, cy, cz, count, dens) in enumerate(clusters[:12]):
    print(f"  ({cx:7.0f}, {cy:7.0f}, Z={cz:.0f}): {count:>6,}v ({dens*100:.1f}%)")

# ── Sea level (Z 0-20): Coastline, sports fields ──
print("\n=== Sea Level (Z 0-20): Key clusters ===")
coastal = v[(z >= 0) & (z < 20)]
clusters = find_clusters(coastal, grid_size=20)
for i, (cx, cy, cz, count, dens) in enumerate(clusters[:12]):
    print(f"  ({cx:7.0f}, {cy:7.0f}, Z={cz:.0f}): {count:>6,}v ({dens*100:.1f}%)")

# ── Mid level (Z 20-50): Transition areas ──
print("\n=== Mid Level (Z 20-50): Key clusters ===")
mid = v[(z >= 20) & (z < 50)]
clusters = find_clusters(mid, grid_size=25)
for i, (cx, cy, cz, count, dens) in enumerate(clusters[:12]):
    print(f"  ({cx:7.0f}, {cy:7.0f}, Z={cz:.0f}): {count:>6,}v ({dens*100:.1f}%)")

# ── Face area analysis: find large flat areas ──
print("\n=== Large Structures (by face area & height) ===")
face_v = v[faces.astype(int)]
face_center = face_v.mean(axis=1)
face_z = face_center[:, 2]
face_xy = face_center[:, :2]

# For each elevation band, find the densest XY clusters of faces
for lo, hi, label in [(0, 25, "Sea"), (25, 50, "Mid"), (50, 85, "Plateau")]:
    band_mask = (face_z >= lo) & (face_z < hi)
    band_faces = face_xy[band_mask]
    if len(band_faces) < 100:
        continue

    # Simple grid-based density
    bx, by = band_faces[:, 0], band_faces[:, 1]

    # Find peaks in density — use simple grid
    grid = {}
    step = 15
    for i in range(len(band_faces)):
        key = (int(bx[i] // step), int(by[i] // step))
        grid[key] = grid.get(key, 0) + 1

    top = sorted(grid.items(), key=lambda x: -x[1])[:8]
    print(f"\n  {label} ({lo}-{hi}):")
    for (gx, gy), cnt in top:
        world_x = gx * step + step/2
        world_y = gy * step + step/2
        print(f"    ({world_x:6.0f}, {world_y:6.0f}): {cnt:>6,} faces")

# ── Also find the Academic Building: the long curved structure ──
# The Academic Building is a ~300m long arc on the plateau
# Look for elongated structures (high variance in one XY direction)
print("\n=== Elongated Structures (potential Academic Building arc) ===")
for lo, hi in [(45, 85)]:
    band = v[(z >= lo) & (z < hi)]
    if len(band) < 1000:
        continue

    # Sliding window to find long thin clusters
    step = 30
    for cx in np.arange(band[:,0].min(), band[:,0].max(), step):
        for cy in np.arange(band[:,1].min(), band[:,1].max(), step):
            window = (abs(band[:,0] - cx) < step) & (abs(band[:,1] - cy) < step)
            if window.sum() > 500:
                wv = band[window]
                # Check if elongated
                xy_std = np.std(wv[:, :2], axis=0)
                elongation = max(xy_std) / (min(xy_std) + 1)
                if elongation > 2.0:
                    pass  # just collecting data via grid analysis already done

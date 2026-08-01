#!/usr/bin/env python3
"""Verify the optimized piazza model."""
import json, struct, io
from pathlib import Path
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PIAZZA = Path("/home/zliki/HKUST_3D/output/demo/hkust_piazza.glb")
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
    type_sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    type_counts = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
    elem = type_sizes[comp_type]
    ncomp = type_counts[acc_type]
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

print("=== Optimized Piazza Model ===")
print(f"Vertices: {len(v):,}")
print(f"X: [{v[:,0].min():.2f}, {v[:,0].max():.2f}]  range={v[:,0].max()-v[:,0].min():.1f}")
print(f"Y: [{v[:,1].min():.2f}, {v[:,1].max():.2f}]  range={v[:,1].max()-v[:,1].min():.1f}")
print(f"Z: [{v[:,2].min():.2f}, {v[:,2].max():.2f}]  range={v[:,2].max()-v[:,2].min():.1f}")

print(f"\nUV: U=[{uv[:,0].min():.4f},{uv[:,0].max():.4f}] V=[{uv[:,1].min():.4f},{uv[:,1].max():.4f}]")

# Check texture
images = gltf.get('images', [])
if images:
    img_info = images[0]
    bv_idx = img_info.get('bufferView')
    bv = buffer_views[bv_idx]
    off = bv.get('byteOffset', 0)
    length = bv['byteLength']
    tex = Image.open(io.BytesIO(bin_data[off:off+length]))
    print(f"\nEmbedded texture: {tex.size[0]}x{tex.size[1]} ({tex.mode})")

# Material
mats = gltf.get('materials', [])
if mats:
    print(f"\nMaterial: {json.dumps(mats[0], indent=2)}")

# UV distribution
print(f"\nUV V distribution:")
for lo, hi, label in [(0, 0.1, "0-10%"), (0.1, 0.3, "10-30%"), (0.3, 0.5, "30-50%"),
                        (0.5, 0.7, "50-70%"), (0.7, 0.9, "70-90%"), (0.9, 1.0, "90-100%")]:
    count = ((uv[:,1] >= lo) & (uv[:,1] < hi)).sum()
    print(f"  {label}: {count:,} vertices ({100*count/len(uv):.1f}%)")

# Key landmarks — Check vertex density at various elevations
z = v[:, 2]
print(f"\nElevation profile:")
for pct in [5, 25, 50, 75, 90, 95]:
    print(f"  Z p{pct}: {np.percentile(z, pct):.1f}")

# Size estimate
print(f"\nApproximate extent:")
print(f"  Horizontal: {v[:,0].max()-v[:,0].min():.0f} x {v[:,1].max()-v[:,1].min():.0f} units")
print(f"  Vertical: {v[:,2].max()-v[:,2].min():.0f} units")

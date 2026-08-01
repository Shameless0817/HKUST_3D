#!/usr/bin/env python3
"""
增强 HKUST 3D 模型：建筑高亮 + 地形对比
让每栋建筑清晰可见
"""
import struct, json
from pathlib import Path
import numpy as np
import trimesh

INPUT = Path("/home/zliki/HKUST_3D/output/demo/hkust_optimized.glb")
OUTPUT = Path("/home/zliki/HKUST_3D/output/demo/hkust_enhanced.glb")

def main():
    print("Loading model...")
    m = trimesh.load(str(INPUT), force="mesh")
    v = m.vertices
    f = m.faces
    vc = m.visual.vertex_colors.copy() if hasattr(m.visual, 'vertex_colors') else None

    print(f"Vertices: {len(v):,}, Faces: {len(f):,}")

    # --- Step 1: Classify each face as building or terrain ---
    face_z = v[f][:, :, 2]  # (n_faces, 3)
    face_z_min = face_z.min(axis=1)
    face_z_max = face_z.max(axis=1)
    face_z_mean = face_z.mean(axis=1)
    face_z_range = face_z_max - face_z_min
    face_z_std = face_z.std(axis=1)

    # Ground level: lowest 20% of vertices
    ground_z = np.percentile(v[:, 2], 20)

    # Building score: high Z + steep faces + small area variation
    building_score = (
        np.clip((face_z_min - ground_z) / 20.0, 0, 1) * 0.4 +  # height above ground
        np.clip(face_z_range / 5.0, 0, 1) * 0.4 +                # verticality
        np.clip(face_z_std / 2.0, 0, 1) * 0.2                     # surface variation
    )

    is_building = building_score > 0.3
    is_terrain = ~is_building

    print(f"Building faces: {is_building.sum():,} ({100*is_building.sum()/len(f):.1f}%)")
    print(f"Terrain faces: {is_terrain.sum():,} ({100*is_terrain.sum()/len(f):.1f}%)")

    # --- Step 2: Create enhanced vertex colors ---
    new_colors = np.zeros((len(v), 4), dtype=np.uint8)

    # Per-vertex: count how many building vs terrain faces it belongs to
    vertex_building_count = np.bincount(f[is_building].flatten(), minlength=len(v))
    vertex_terrain_count = np.bincount(f[is_terrain].flatten(), minlength=len(v))
    vertex_total = vertex_building_count + vertex_terrain_count

    for i in range(len(v)):
        b_count = vertex_building_count[i]
        t_count = vertex_terrain_count[i]
        total = b_count + t_count

        if total == 0:
            new_colors[i] = [128, 128, 128, 255]
            continue

        b_ratio = b_count / total

        if b_ratio > 0.5:
            # Building: warm beige/gray
            z_norm = np.clip((v[i, 2] - ground_z) / 80.0, 0, 1)
            r = int(220 + z_norm * 35)
            g = int(200 + z_norm * 40)
            b = int(180 + z_norm * 50)
            new_colors[i] = [min(r, 255), min(g, 255), min(b, 255), 255]
        else:
            # Terrain: green-brown gradient based on height
            z_norm = np.clip((v[i, 2] - v[:, 2].min()) / (v[:, 2].max() - v[:, 2].min()), 0, 1)
            if z_norm < 0.3:
                # Low: dark green (vegetation)
                r, g, b_val = 60, 100 + int(z_norm * 200), 50
            elif z_norm < 0.6:
                # Mid: brown-green
                r, g, b_val = 120 + int(z_norm * 100), 130, 80
            else:
                # High: gray-brown (bare ground/rock)
                r, g, b_val = 170, 160, 140
            new_colors[i] = [min(r, 255), min(g, 255), min(b_val, 255), 255]

    # --- Step 3: Also mix in original texture colors for buildings ---
    if vc is not None:
        for i in range(len(v)):
            b_count = vertex_building_count[i]
            t_count = vertex_terrain_count[i]
            total = b_count + t_count
            if total == 0:
                continue
            b_ratio = b_count / total
            # Blend: buildings use 30% original + 70% new, terrain uses 70% original + 30% new
            if b_ratio > 0.5:
                blend = 0.3  # 30% original texture for buildings
            else:
                blend = 0.7  # 70% original texture for terrain
            new_colors[i] = np.clip(
                vc[i].astype(float) * blend + new_colors[i].astype(float) * (1 - blend),
                0, 255
            ).astype(np.uint8)

    # --- Step 4: Create output mesh ---
    visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=new_colors)
    enhanced = trimesh.Trimesh(
        vertices=v.copy(),
        faces=f.copy(),
        visual=visual,
        process=False,
    )

    # Clean up
    enhanced.update_faces(enhanced.nondegenerate_faces())
    enhanced.remove_unreferenced_vertices()

    print(f"Output: {len(enhanced.vertices):,}v, {len(enhanced.faces):,}f")

    # Save
    enhanced.export(str(OUTPUT))
    mb = OUTPUT.stat().st_size / 1e6
    print(f"\n✓ {OUTPUT.name} ({mb:.1f} MB)")

    # Verify
    with open(OUTPUT, 'rb') as f:
        data = f.read()
    json_len = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20+json_len])
    attrs = gltf['meshes'][0]['primitives'][0]['attributes']
    print(f"  COLOR_0: {'COLOR_0' in attrs}")

    print(f"\n打开 https://gltf-viewer.donmccurdy.com/ 拖入:")
    print(f"  {OUTPUT}")

    # Instructions for best viewing
    print(f"\n💡 查看技巧:")
    print(f"  1. 用鼠标右键旋转到侧面角度")
    print(f"  2. 滚轮缩放")
    print(f"  3. 米色/暖灰 = 建筑, 绿色 = 植被地面")


if __name__ == "__main__":
    main()

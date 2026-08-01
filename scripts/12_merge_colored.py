#!/usr/bin/env python3
"""
Create merged GLB with height-based vertex colors.
KTX2 textures can't be decoded without basisu, so we use
z-height coloring as a universally-compatible fallback.
"""
import struct, json
from pathlib import Path
import numpy as np
import trimesh
import glob

GLB_DIR = Path("/home/zliki/HKUST_3D/output/csdi/f2/glb")
OUT_DIR = Path("/home/zliki/HKUST_3D/output/csdi/f2")
DEMO_DIR = Path("/home/zliki/HKUST_3D/output/demo")

def height_color(z, z_min, z_max):
    """Map z-height to a terrain-like color (blue→green→gray→white)."""
    if z_max - z_min < 0.001:
        return np.array([200, 200, 200, 255], dtype=np.uint8)

    t = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)

    # Color stops (position, r, g, b)
    stops = [
        (0.00, np.array([ 60,  80, 140])),  # blue (water/shadow)
        (0.15, np.array([ 80, 120,  80])),  # dark green (vegetation)
        (0.30, np.array([140, 150, 130])),  # light gray-green
        (0.50, np.array([180, 175, 170])),  # warm gray (buildings mid)
        (0.70, np.array([210, 205, 200])),  # light gray (building tops)
        (0.90, np.array([235, 230, 225])),  # off-white (roofs)
        (1.00, np.array([250, 245, 240])),  # warm white (highest)
    ]

    # Find surrounding stops
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0
            rgb = (c0 * (1 - alpha) + c1 * alpha).astype(np.uint8)
            return np.array([rgb[0], rgb[1], rgb[2], 255], dtype=np.uint8)

    return np.array([200, 200, 200, 255], dtype=np.uint8)


def main():
    tiles = sorted(glob.glob(str(GLB_DIR / "*.glb")))
    print(f"Loading {len(tiles)} tiles...")

    all_meshes = []
    all_z = []

    for i, tp in enumerate(tiles):
        try:
            m = trimesh.load(tp, force="mesh")
            if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0:
                all_meshes.append(m)
                all_z.extend(m.vertices[:, 2].tolist())
            elif isinstance(m, trimesh.Scene):
                for _, geom in m.geometry.items():
                    if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                        all_meshes.append(geom)
                        all_z.extend(geom.vertices[:, 2].tolist())
        except Exception as e:
            print(f"  Skip {Path(tp).name}: {e}")

        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(tiles)}]")

    print(f"\nLoaded {len(all_meshes)} meshes")

    if not all_meshes:
        print("No valid meshes!")
        return

    # Compute global z range
    all_z = np.array(all_z)
    z_min, z_max = float(all_z.min()), float(all_z.max())
    print(f"Z range: {z_min:.1f} → {z_max:.1f}")

    # Apply height colors to each mesh
    print("Applying height-based vertex colors...")
    colored_meshes = []
    total_v, total_f = 0, 0

    for i, mesh in enumerate(all_meshes):
        zs = mesh.vertices[:, 2]
        colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
        for j, z in enumerate(zs):
            colors[j] = height_color(z, z_min, z_max)

        visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=colors,
        )
        colored = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            visual=visual,
            process=False,  # Don't merge vertices (would lose colors)
        )
        colored_meshes.append(colored)
        total_v += len(mesh.vertices)
        total_f += len(mesh.faces)

        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(all_meshes)}] colored")

    # Merge
    print(f"\nMerging {len(colored_meshes)} meshes ({total_v:,}v, {total_f:,}f)...")
    merged = trimesh.util.concatenate(colored_meshes)

    # Clean up without destroying vertex colors
    merged.update_faces(merged.nondegenerate_faces())
    merged.remove_unreferenced_vertices()

    print(f"Merged: {len(merged.vertices):,}v, {len(merged.faces):,}f")

    # Check vertex colors preserved
    if hasattr(merged.visual, 'vertex_colors'):
        vc = merged.visual.vertex_colors
        print(f"Vertex colors: shape={vc.shape}, range=[{vc.min()},{vc.max()}]")

    # Center and scale
    merged.vertices -= merged.centroid
    extent = merged.vertices.max(axis=0) - merged.vertices.min(axis=0)
    merged.vertices *= 200.0 / max(extent)

    # Save merged
    merged_path = OUT_DIR / "hkust_csdi_merged.glb"
    merged.export(str(merged_path))
    mb = merged_path.stat().st_size / 1e6
    print(f"\n✓ Merged GLB: {merged_path} ({mb:.1f} MB)")

    # Demo copy
    demo_path = DEMO_DIR / "hkust_csdi_merged.glb"
    merged.export(str(demo_path))
    print(f"✓ Demo copy: {demo_path}")

    # Verify
    with open(merged_path, 'rb') as f:
        data = f.read()
    json_len = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20+json_len])
    meshes = gltf.get('meshes', [])
    if meshes:
        attrs = meshes[0]['primitives'][0]['attributes']
        print(f"Verify: attrs={list(attrs.keys())}, COLOR_0={'COLOR_0' in attrs}")

    print("\nDone!")


if __name__ == "__main__":
    main()

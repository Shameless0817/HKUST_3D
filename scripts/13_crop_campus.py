#!/usr/bin/env python3
"""
修复 CSDI 模型：裁剪到 HKUST 校园区域 + 更好的着色
"""
import struct, json
from pathlib import Path
import numpy as np
import trimesh
import glob

GLB_DIR = Path("/home/zliki/HKUST_3D/output/csdi/f2/glb")
OUT_DIR = Path("/home/zliki/HKUST_3D/output/csdi/f2")
DEMO_DIR = Path("/home/zliki/HKUST_3D/output/demo")

def main():
    tiles = sorted(glob.glob(str(GLB_DIR / "*.glb")))
    print(f"Loading {len(tiles)} tiles...")

    # Step 1: Load all meshes and collect vertex stats
    all_meshes = []
    mesh_meta = []  # (name, centroid, vertex_count, z_range)

    for tp in tiles:
        try:
            m = trimesh.load(tp, force="mesh")
            if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
                continue
            name = Path(tp).name
            v = m.vertices
            c = m.centroid
            zr = v[:, 2].max() - v[:, 2].min()
            all_meshes.append(m)
            mesh_meta.append((name, c, len(v), zr))
        except Exception as e:
            pass

    print(f"Loaded {len(all_meshes)} valid meshes")
    if not all_meshes:
        print("No meshes!")
        return

    # Step 2: Find the best HKUST cluster
    # Strategy: use weighted vertex centroid, then find dense cluster
    all_verts = np.vstack([m.vertices for m in all_meshes])
    print(f"Total vertices: {len(all_verts):,}")

    # Use a histogram approach to find the densest X-Y region
    # HKUST should be in a relatively small area with many vertices (buildings)
    x, y = all_verts[:, 0], all_verts[:, 1]

    # Compute 2D histogram (100m bins)
    bin_size = 100.0  # 100m bins
    x_bins = np.arange(x.min(), x.max(), bin_size)
    y_bins = np.arange(y.min(), y.max(), bin_size)

    if len(x_bins) < 2 or len(y_bins) < 2:
        print("Not enough data for histogram")
        return

    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
    print(f"Histogram: {hist.shape}, max density = {hist.max():.0f} verts/bin")

    # Find the densest bin
    max_idx = np.unravel_index(hist.argmax(), hist.shape)
    dense_x = (x_edges[max_idx[0]] + x_edges[max_idx[0] + 1]) / 2
    dense_y = (y_edges[max_idx[1]] + y_edges[max_idx[1] + 1]) / 2
    print(f"Densest point: X={dense_x:.0f}, Y={dense_y:.0f}")

    # Top 5 dense areas
    flat_hist = hist.flatten()
    top_indices = np.argsort(flat_hist)[-10:][::-1]
    print("\nTop 10 dense clusters:")
    for rank, idx in enumerate(top_indices[:10]):
        ix, iy = np.unravel_index(idx, hist.shape)
        cx = (x_edges[ix] + x_edges[ix + 1]) / 2
        cy = (y_edges[iy] + y_edges[iy + 1]) / 2
        count = hist[ix, iy]
        # Find the tile that covers this area
        nearby_tiles = []
        for name, c, vc, zr in mesh_meta:
            dist = np.sqrt((c[0] - cx)**2 + (c[1] - cy)**2)
            if dist < 5000:
                nearby_tiles.append((name, dist))
        nearby_tiles.sort(key=lambda x: x[1])
        tile_names = ", ".join([t[0] for t in nearby_tiles[:3]])
        print(f"  #{rank+1}: X={cx:.0f}, Y={cy:.0f}, count={count:.0f} verts → {tile_names}")

    # Step 3: Crop to a reasonable campus-sized area around densest point
    # HKUST campus is ~1km across
    CROP_SIZE = 600  # meters - campus-size
    crop_bounds = [
        [dense_x - CROP_SIZE, dense_y - CROP_SIZE],
        [dense_x + CROP_SIZE, dense_y + CROP_SIZE],
    ]
    print(f"\nCropping to {CROP_SIZE*2}m × {CROP_SIZE*2}m around ({dense_x:.0f}, {dense_y:.0f})")

    # Crop all meshes
    cropped_meshes = []
    for mesh in all_meshes:
        v = mesh.vertices
        # Check if any vertex is in crop bounds
        in_x = (v[:, 0] >= crop_bounds[0][0]) & (v[:, 0] <= crop_bounds[1][0])
        in_y = (v[:, 1] >= crop_bounds[0][1]) & (v[:, 1] <= crop_bounds[1][1])
        in_bounds = in_x & in_y

        if in_bounds.sum() < 10:
            continue  # Skip tiles outside crop area

        # Extract the cropped portion
        # Keep faces where at least one vertex is in bounds
        faces_in = in_bounds[mesh.faces]
        keep_face = faces_in.any(axis=1)
        if keep_face.sum() < 5:
            continue

        # Sub-mesh with only kept faces
        kept_faces = mesh.faces[keep_face]
        # Get unique vertices used by kept faces
        used_verts = np.unique(kept_faces)
        old_to_new = {old: new for new, old in enumerate(used_verts)}
        new_faces = np.array([[old_to_new[f] for f in face] for face in kept_faces])
        new_verts = v[used_verts]

        if len(new_verts) < 10:
            continue

        # Check for KTX2 texture → try to bake colors if present
        cropped = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
        cropped_meshes.append(cropped)

    print(f"Cropped to {len(cropped_meshes)} mesh pieces")

    if not cropped_meshes:
        print("❌ Nothing in crop area! Trying larger crop...")
        # Fallback: use a larger crop (2000m)
        CROP_SIZE = 2000
        crop_bounds = [
            [dense_x - CROP_SIZE, dense_y - CROP_SIZE],
            [dense_x + CROP_SIZE, dense_y + CROP_SIZE],
        ]
        for mesh in all_meshes:
            v = mesh.vertices
            in_x = (v[:, 0] >= crop_bounds[0][0]) & (v[:, 0] <= crop_bounds[1][0])
            in_y = (v[:, 1] >= crop_bounds[0][1]) & (v[:, 1] <= crop_bounds[1][1])
            in_bounds = in_x & in_y
            if in_bounds.sum() < 10:
                continue
            faces_in = in_bounds[mesh.faces]
            keep_face = faces_in.any(axis=1)
            if keep_face.sum() < 5:
                continue
            kept_faces = mesh.faces[keep_face]
            used_verts = np.unique(kept_faces)
            old_to_new = {old: new for new, old in enumerate(used_verts)}
            new_faces = np.array([[old_to_new[f] for f in face] for face in kept_faces])
            new_verts = v[used_verts]
            if len(new_verts) < 10:
                continue
            cropped = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
            cropped_meshes.append(cropped)
        print(f"Larger crop: {len(cropped_meshes)} mesh pieces")

    if not cropped_meshes:
        print("Still nothing! Aborting.")
        return

    # Step 4: Apply height-based colors (buildings have more Z variance)
    all_cropped_v = np.vstack([m.vertices for m in cropped_meshes])
    z_min, z_max = float(all_cropped_v[:, 2].min()), float(all_cropped_v[:, 2].max())
    print(f"Cropped Z range: {z_min:.0f} → {z_max:.0f} (range={z_max-z_min:.0f}m)")

    # Better color scheme: gray buildings, green ground
    colored_meshes = []
    for mesh in cropped_meshes:
        zs = mesh.vertices[:, 2]
        z_norm = np.clip((zs - z_min) / (z_max - z_min + 0.01), 0, 1)

        colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
        for j in range(len(zs)):
            t = z_norm[j]
            if t < 0.1:
                # Ground/water: dark blue-green
                rgb = np.array([50 + int(t * 500), 80 + int(t * 200), 100 + int(t * 500)])
            elif t < 0.3:
                # Low vegetation / ground: green-brown
                rgb = np.array([140 + int(t * 200), 150 + int(t * 100), 100])
            elif t < 0.7:
                # Buildings: warm gray
                rgb = np.array([180 + int(t * 40), 175 + int(t * 40), 170 + int(t * 40)])
            else:
                # Roofs / tall structures: lighter
                rgb = np.array([220 + int(t * 30), 215 + int(t * 30), 210 + int(t * 30)])

            rgb = np.clip(rgb, 0, 255)
            colors[j] = [rgb[0], rgb[1], rgb[2], 255]

        visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        colored = trimesh.Trimesh(
            vertices=mesh.vertices.copy(),
            faces=mesh.faces.copy(),
            visual=visual,
            process=False,
        )
        colored_meshes.append(colored)

    # Step 5: Merge and export
    print("Merging...")
    merged = trimesh.util.concatenate(colored_meshes)
    merged.update_faces(merged.nondegenerate_faces())
    merged.remove_unreferenced_vertices()

    # Center and scale nicely
    merged.vertices -= merged.centroid
    extent = merged.vertices.max(axis=0) - merged.vertices.min(axis=0)
    scale = 200.0 / max(extent)
    merged.vertices *= scale

    print(f"Merged: {len(merged.vertices):,}v, {len(merged.faces):,}f")

    # Verify colors
    if hasattr(merged.visual, 'vertex_colors'):
        vc = merged.visual.vertex_colors
        print(f"Vertex colors: shape={vc.shape}, range=[{vc.min()},{vc.max()}]")

    # Save
    merged_path = OUT_DIR / "hkust_csdi_cropped.glb"
    merged.export(str(merged_path))
    mb = merged_path.stat().st_size / 1e6
    print(f"\n✓ Cropped: {merged_path} ({mb:.1f} MB)")

    # Demo copy - also update the main merged
    demo_merged = DEMO_DIR / "hkust_csdi_merged.glb"
    merged.export(str(demo_merged))
    print(f"✓ Demo: {demo_merged} ({mb:.1f} MB)")

    # Verify
    with open(merged_path, 'rb') as f:
        data = f.read()
    json_len = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20+json_len])
    attrs = gltf['meshes'][0]['primitives'][0]['attributes']
    pos_count = gltf['accessors'][attrs['POSITION']]['count']
    print(f"Verify: {pos_count:,}v, attrs={list(attrs.keys())}, COLOR_0={'COLOR_0' in attrs}")

    print("\n✅ 完成！现在打开 https://gltf-viewer.donmccurdy.com/ 拖入")
    print(f"   output/demo/hkust_csdi_merged.glb ({mb:.1f} MB)")


if __name__ == "__main__":
    main()

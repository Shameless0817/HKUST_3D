#!/usr/bin/env python3
"""
HKUST 完整校园 3D 模型 v2：
- Google Earth 地形 + 海岸线
- GeoJSON 建筑足迹挤出 (在地形之上)
- 层次化着色
"""
import json, struct
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import triangulate

GEOJSON = Path("/home/zliki/HKUST_3D/output/demo/hkust_buildings.geojson")
TERRAIN = Path("/home/zliki/HKUST_3D/output/demo/hkust_optimized.glb")
OUTPUT = Path("/home/zliki/HKUST_3D/output/demo/hkust_full_campus.glb")

HKUST_LON, HKUST_LAT = 114.263, 22.337
CAMPUS_RADIUS_M = 700

def extrude_polygon(coords_xy, z_base, height, color):
    """Extrude a 2D polygon into a colored 3D mesh."""
    pts = np.array(coords_xy)
    if len(pts) < 3:
        return None

    poly = ShapelyPolygon(pts)
    if not poly.is_valid or poly.area < 0.5:
        poly = poly.buffer(0)
        if poly.is_empty:
            return None
        if hasattr(poly, 'geoms'):
            geoms = [g for g in poly.geoms if hasattr(g, 'exterior')]
            if not geoms:
                return None
            poly = max(geoms, key=lambda g: g.area)
        pts = np.array(poly.exterior.coords)

    # Triangulate
    try:
        tris = triangulate(poly)
    except Exception:
        return None
    if not tris:
        return None

    all_tri_pts = []
    for tri in tris:
        coords = np.array(tri.exterior.coords)[:3]
        all_tri_pts.append(coords)
    if not all_tri_pts:
        return None

    top_tris = np.vstack(all_tri_pts)
    n_tri_verts = len(top_tris)

    top_3d = np.column_stack([top_tris, np.full(n_tri_verts, z_base + height)])
    bottom_3d = np.column_stack([top_tris, np.full(n_tri_verts, z_base)])
    all_verts = np.vstack([bottom_3d, top_3d])

    bottom_faces = np.arange(0, n_tri_verts).reshape(-1, 3)
    top_faces = np.arange(n_tri_verts, 2 * n_tri_verts).reshape(-1, 3)[:, ::-1]

    # Walls from boundary
    boundary = np.array(poly.exterior.coords)
    wall_bot = np.column_stack([boundary, np.full(len(boundary), z_base)])
    wall_top = np.column_stack([boundary, np.full(len(boundary), z_base + height)])
    wall_all = np.vstack([wall_bot, wall_top])
    n_wall = len(boundary)
    wall_offset = n_tri_verts * 2
    wall_faces = []
    for i in range(n_wall - 1):
        b0, b1 = i, i + 1
        t0, t1 = n_wall + i, n_wall + i + 1
        wall_faces.append([wall_offset + b0, wall_offset + t0, wall_offset + t1])
        wall_faces.append([wall_offset + b0, wall_offset + t1, wall_offset + b1])

    if wall_faces:
        wall_faces = np.array(wall_faces)
        all_verts = np.vstack([all_verts, wall_all])
    else:
        wall_faces = np.zeros((0, 3), dtype=int)

    all_faces = np.vstack([bottom_faces, top_faces, wall_faces])
    all_colors = np.tile(color, (len(all_verts), 1))

    visual = trimesh.visual.ColorVisuals(vertex_colors=all_colors.astype(np.uint8))
    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, visual=visual, process=False)
    return mesh


def main():
    print("Loading terrain...")
    terrain = trimesh.load(str(TERRAIN), force="mesh")
    tv = terrain.vertices
    tf = terrain.faces

    # Terrain coordinate ranges
    tx_min, tx_max = tv[:, 0].min(), tv[:, 0].max()
    ty_min, ty_max = tv[:, 1].min(), tv[:, 1].max()
    tz_min, tz_max = tv[:, 2].min(), tv[:, 2].max()

    print(f"Terrain: X=[{tx_min:.0f},{tx_max:.0f}] Y=[{ty_min:.0f},{ty_max:.0f}] Z=[{tz_min:.0f},{tz_max:.0f}]")

    # Load buildings
    print("Loading GeoJSON...")
    with open(GEOJSON) as f:
        geojson = json.load(f)

    lat_to_m = 111000.0
    lon_to_m = 111000.0 * np.cos(np.radians(HKUST_LAT))

    # Filter and convert buildings to meters
    raw_buildings = []
    for feat in geojson['features']:
        geom = feat.get('geometry', {})
        if geom.get('type') != 'Polygon':
            continue
        coords = geom['coordinates'][0]
        if len(coords) < 3:
            continue
        xy = np.array([[(c[0] - HKUST_LON) * lon_to_m,
                        (c[1] - HKUST_LAT) * lat_to_m] for c in coords])
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        dist = np.sqrt(cx**2 + cy**2)
        if dist < CAMPUS_RADIUS_M:
            floors = (feat.get('properties', {}) or {}).get('num_floors', 0) or 0
            raw_buildings.append({'xy': xy, 'floors': floors, 'dist': dist})

    print(f"Campus buildings: {len(raw_buildings)}")

    if not raw_buildings:
        print("No buildings!")
        return

    # Compute building XY ranges for mapping
    all_bx = np.concatenate([b['xy'][:, 0] for b in raw_buildings])
    all_by = np.concatenate([b['xy'][:, 1] for b in raw_buildings])
    bx_min, bx_max = all_bx.min(), all_bx.max()
    by_min, by_max = all_by.min(), all_by.max()

    # Map building XY → terrain XY
    def bx_to_tx(bx):
        return tx_min + (bx - bx_min) / (bx_max - bx_min) * (tx_max - tx_min)

    def by_to_ty(by):
        return ty_min + (by - by_min) / (by_max - by_min) * (ty_max - ty_min)

    # Build a terrain Z lookup (KD-tree for nearest-neighbor query)
    from scipy.spatial import cKDTree
    terrain_xy = tv[:, :2]
    tree = cKDTree(terrain_xy)

    def get_terrain_z(tx, ty):
        """Query terrain Z at given terrain coordinates."""
        dist, idx = tree.query([tx, ty], k=3)
        if hasattr(dist, '__len__'):
            weights = 1.0 / (dist + 0.01)
            weights /= weights.sum()
            z = np.sum(tv[idx, 2] * weights)
        else:
            z = tv[idx, 2]
        return float(z)

    # --- Create enhanced terrain with water coloring ---
    print("Processing terrain...")

    # Classify terrain vertices: water (low), ground (mid), building-base (high)
    terrain_z = tv[:, 2]
    z_05 = np.percentile(terrain_z, 5)
    z_20 = np.percentile(terrain_z, 20)
    z_60 = np.percentile(terrain_z, 60)

    terrain_colors = np.zeros((len(tv), 4), dtype=np.uint8)
    for i in range(len(tv)):
        z = terrain_z[i]
        if z < z_05 + 2:
            # Water: blue
            terrain_colors[i] = [40, 80, 160, 255]
        elif z < z_20:
            # Low ground: green
            t = (z - z_05) / (z_20 - z_05)
            terrain_colors[i] = [int(60 + t * 30), int(120 + t * 30), int(50 + t * 30), 255]
        elif z < z_60:
            # Mid ground: brown-green
            terrain_colors[i] = [150, 140, 100, 255]
        else:
            # High ground: gray-brown
            terrain_colors[i] = [180, 170, 150, 255]

    # Water level: set all Z below water threshold to water_level
    water_z = z_05
    tv_flat = tv.copy()
    below_water = terrain_z < water_z
    tv_flat[below_water, 2] = water_z  # flatten water surface

    terrain_vis = trimesh.visual.ColorVisuals(vertex_colors=terrain_colors)
    terrain_mesh = trimesh.Trimesh(vertices=tv_flat, faces=tf, visual=terrain_vis, process=False)

    # --- Extrude buildings on terrain ---
    print(f"Extruding {len(raw_buildings)} buildings on terrain...")
    building_meshes = []

    def building_color(floors):
        if floors >= 8:
            return np.array([235, 220, 200, 255])
        elif floors >= 6:
            return np.array([225, 210, 190, 255])
        elif floors >= 4:
            return np.array([215, 200, 180, 255])
        elif floors >= 2:
            return np.array([205, 190, 170, 255])
        else:
            return np.array([195, 180, 160, 255])

    for i, b in enumerate(raw_buildings):
        # Map building coordinates to terrain coordinates
        xy_t = np.array([[bx_to_tx(c[0]), by_to_ty(c[1])] for c in b['xy']])
        cx_t = xy_t[:, 0].mean()
        cy_t = xy_t[:, 1].mean()

        # Get terrain Z at building center
        z_base = get_terrain_z(cx_t, cy_t)

        floors = b['floors']
        height = max(floors * 3.5, 5.0) if floors > 0 else 6.0

        # Scale building height to terrain units
        # terrain Z range is about 200 display units for ~200m real elevation
        # So 1 display unit ≈ 1m of real height
        height *= 0.8  # Slightly compress buildings

        color = building_color(floors)

        mesh = extrude_polygon(xy_t, z_base, height, color)
        if mesh is not None:
            building_meshes.append(mesh)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(raw_buildings)}]")

    print(f"Extruded: {len(building_meshes)} buildings")

    # --- Merge everything ---
    print("Merging terrain + buildings...")
    all_meshes = [terrain_mesh] + building_meshes
    final = trimesh.util.concatenate(all_meshes)
    final.update_faces(final.nondegenerate_faces())
    final.remove_unreferenced_vertices()

    print(f"Final: {len(final.vertices):,}v, {len(final.faces):,}f")

    # Center and scale for good viewing
    final.vertices -= final.centroid
    extent = final.vertices.max(axis=0) - final.vertices.min(axis=0)
    final.vertices *= 200.0 / max(extent)

    # Save
    final.export(str(OUTPUT))
    mb = OUTPUT.stat().st_size / 1e6
    print(f"\n✓ {OUTPUT} ({mb:.1f} MB)")

    # Verify
    with open(OUTPUT, 'rb') as f:
        data = f.read()
    js_len = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20+js_len])
    attrs = gltf['meshes'][0]['primitives'][0]['attributes']
    vc = gltf['accessors'][attrs['POSITION']]['count']
    print(f"Verify: {vc:,}v, COLOR_0={'COLOR_0' in attrs}")

    print(f"""
✅ 完成! 打开 https://gltf-viewer.donmccurdy.com/ 拖入:
   {OUTPUT}

🎨 配色:
   蓝色 = 海岸线水域
   绿色/棕色 = 地形
   米色建筑 = 颜色越浅=楼层越高
   建筑物在地形之上

💡 查看技巧:
   - 右键拖动旋转视角，从侧面看建筑
   - 滚轮缩放，靠近看细节
   - 找到海岸线就能确定校园朝向
""")


if __name__ == "__main__":
    main()

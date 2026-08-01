#!/usr/bin/env python3
"""
从建筑足迹 GeoJSON + 地形生成清晰的 HKUST 3D 模型
使用 Shapely 做多边形三角剖分
"""
import json, struct
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon as ShapelyPolygon, Point
from shapely.ops import triangulate

GEOJSON = Path("/home/zliki/HKUST_3D/output/demo/hkust_buildings.geojson")
TERRAIN = Path("/home/zliki/HKUST_3D/output/demo/hkust_optimized.glb")
OUTPUT = Path("/home/zliki/HKUST_3D/output/demo/hkust_campus.glb")

CAMPUS_LON, CAMPUS_LAT = 114.263, 22.337
CAMPUS_RADIUS_M = 800

def extrude_polygon(coords_xy, z_base, height, color):
    """Extrude a 2D polygon (in meters XY) into a colored 3D mesh."""
    pts = np.array(coords_xy)
    if len(pts) < 3:
        return None

    poly = ShapelyPolygon(pts)
    if not poly.is_valid or poly.area < 0.1:
        poly = poly.buffer(0)
        if poly.is_empty or (hasattr(poly, 'geoms') and len(poly.geoms) == 0):
            return None
        # Take largest polygon
        if hasattr(poly, 'geoms'):
            poly = max(poly.geoms, key=lambda g: g.area)
        pts = np.array(poly.exterior.coords)

    # Triangulate
    try:
        tris = triangulate(poly)
    except Exception:
        return None

    if not tris:
        return None

    # Collect all triangles
    all_tri_pts = []
    for tri in tris:
        coords = np.array(tri.exterior.coords)[:3]  # 3 vertices
        all_tri_pts.append(coords)

    if not all_tri_pts:
        return None

    # Build 3D mesh
    top_tris = np.vstack(all_tri_pts)
    n_pts_2d = len(pts)
    n_tri_verts = len(top_tris)

    # Top + Bottom vertices
    top_3d = np.column_stack([top_tris, np.full(n_tri_verts, z_base + height)])
    bottom_3d = np.column_stack([top_tris, np.full(n_tri_verts, z_base)])

    all_verts = np.vstack([bottom_3d, top_3d])

    # Faces: bottom (reversed winding), top
    bottom_faces = np.arange(0, n_tri_verts).reshape(-1, 3)
    top_faces = np.arange(n_tri_verts, 2 * n_tri_verts).reshape(-1, 3)[:, ::-1]

    # Wall faces: find boundary edges
    # Use the original polygon outline for walls
    boundary_pts = np.array(poly.exterior.coords)
    wall_verts_bottom = np.column_stack([boundary_pts, np.full(len(boundary_pts), z_base)])
    wall_verts_top = np.column_stack([boundary_pts, np.full(len(boundary_pts), z_base + height)])

    wall_all = np.vstack([wall_verts_bottom, wall_verts_top])
    n_wall = len(boundary_pts)
    wall_faces = []
    for i in range(n_wall - 1):
        b0, b1 = i, i + 1
        t0, t1 = n_wall + i, n_wall + i + 1
        wall_faces.append([b0, t0, t1])
        wall_faces.append([b0, t1, b1])

    if len(wall_faces) > 0:
        wall_faces = np.array(wall_faces)
        all_verts = np.vstack([all_verts, wall_all])
        wall_offset = n_tri_verts * 2
        wall_faces += wall_offset
    else:
        wall_faces = np.zeros((0, 3), dtype=int)

    all_faces = np.vstack([bottom_faces, top_faces, wall_faces])
    all_colors = np.tile(color, (len(all_verts), 1))

    visual = trimesh.visual.ColorVisuals(vertex_colors=all_colors.astype(np.uint8))
    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces, visual=visual, process=False)
    return mesh


def main():
    print("Loading GeoJSON...")
    with open(GEOJSON) as f:
        geojson = json.load(f)

    features = geojson.get('features', [])
    print(f"Total buildings: {len(features)}")

    lat_to_m = 111000.0
    lon_to_m = 111000.0 * np.cos(np.radians(CAMPUS_LAT))

    # Filter and convert to local XY
    campus = []
    for feat in features:
        geom = feat.get('geometry', {})
        if geom.get('type') != 'Polygon':
            continue
        coords = geom['coordinates'][0]
        if len(coords) < 3:
            continue
        # Convert to meters
        xy = np.array([[(c[0] - CAMPUS_LON) * lon_to_m,
                        (c[1] - CAMPUS_LAT) * lat_to_m] for c in coords])
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        dist = np.sqrt(cx**2 + cy**2)
        if dist < CAMPUS_RADIUS_M:
            props = feat.get('properties', {})
            floors = props.get('num_floors', 0) or 0
            campus.append({'xy': xy, 'floors': floors, 'dist': dist})

    print(f"Campus buildings (<{CAMPUS_RADIUS_M}m): {len(campus)}")

    # Load terrain for visual reference
    print("Loading terrain...")
    terrain = trimesh.load(str(TERRAIN), force="mesh")
    tv = terrain.vertices

    # Terrain is in different coordinate system — use flat ground Z for buildings
    # Buildings will sit at the terrain's lowest Z level
    GROUND_Z = float(np.percentile(tv[:, 2], 5))

    def get_ground_z(cx, cy):
        return GROUND_Z

    def bld_color(floors):
        if floors >= 8:
            return np.array([230, 215, 195, 255])
        elif floors >= 5:
            return np.array([220, 205, 185, 255])
        elif floors >= 2:
            return np.array([210, 195, 175, 255])
        else:
            return np.array([200, 185, 165, 255])

    # Extrude
    print("Extruding buildings...")
    meshes = []
    skipped = 0

    for i, b in enumerate(campus):
        xy = b['xy']
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        z_base = get_ground_z(cx, cy)
        floors = b['floors']
        height = max(floors * 3.5, 4.0) if floors > 0 else 6.0
        color = bld_color(floors)

        mesh = extrude_polygon(xy, z_base, height, color)
        if mesh is not None:
            meshes.append(mesh)
        else:
            skipped += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(campus)}]")

    print(f"Extruded: {len(meshes)}, Skipped: {skipped}")

    if not meshes:
        print("❌ No buildings!")
        return

    # Add a simple ground plane
    print("Adding ground plane...")
    all_x = np.concatenate([m.vertices[:, 0] for m in meshes])
    all_y = np.concatenate([m.vertices[:, 1] for m in meshes])
    pad = 20
    x_min, x_max = all_x.min() - pad, all_x.max() + pad
    y_min, y_max = all_y.min() - pad, all_y.max() + pad

    ground_v = np.array([
        [x_min, y_min, GROUND_Z],
        [x_max, y_min, GROUND_Z],
        [x_max, y_max, GROUND_Z],
        [x_min, y_max, GROUND_Z],
    ])
    ground_f = np.array([[0, 1, 2], [0, 2, 3]])
    gc = np.tile([90, 140, 70, 255], (4, 1))
    gvis = trimesh.visual.ColorVisuals(vertex_colors=gc.astype(np.uint8))
    gmesh = trimesh.Trimesh(vertices=ground_v, faces=ground_f, visual=gvis, process=False)
    meshes.append(gmesh)

    # Merge
    print("Merging...")
    final = trimesh.util.concatenate(meshes)
    final.update_faces(final.nondegenerate_faces())
    final.remove_unreferenced_vertices()

    print(f"Final: {len(final.vertices):,}v, {len(final.faces):,}f")

    # Center & scale
    final.vertices -= final.centroid
    extent = final.vertices.max(axis=0) - final.vertices.min(axis=0)
    final.vertices *= 200.0 / max(extent)

    final.export(str(OUTPUT))
    mb = OUTPUT.stat().st_size / 1e6
    print(f"\n✓ {OUTPUT} ({mb:.1f} MB)")

    with open(OUTPUT, 'rb') as f:
        data = f.read()
    js_len = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20+js_len])
    attrs = gltf['meshes'][0]['primitives'][0]['attributes']
    vc = gltf['accessors'][attrs['POSITION']]['count']
    print(f"Verify: {vc:,}v, COLOR_0={'COLOR_0' in attrs}")

    print(f"\n✅ 打开 https://gltf-viewer.donmccurdy.com/ 拖入:")
    print(f"   {OUTPUT}")
    print(f"\n暖色 = 建筑(颜色越浅=楼层越高), 绿色 = 地面")


if __name__ == "__main__":
    main()

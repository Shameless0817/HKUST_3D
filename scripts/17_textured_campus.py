#!/usr/bin/env python3
"""
HKUST 3D Campus v3 — Photorealistic Textures + Landmarks + Labels

Features:
  - Google Earth photorealistic terrain (from hkust_optimized.glb vertex colors)
  - GeoJSON building extrusion with OSM class-based coloring
  - Red Bird Sundial (Circle of Time) procedural 3D model
  - Label markers on all named buildings
  - Landmark coordinate export (hkust_landmarks.json)
  - Water/coastline detection from elevation

Output:
  output/demo/hkust_final.glb          — 3D campus model
  output/demo/hkust_landmarks.json     — named building coordinates
"""
import json, struct
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import triangulate
from scipy.spatial import cKDTree

# ── Paths ──────────────────────────────────────────────────
PROJECT = Path("/home/zliki/HKUST_3D")
GEOJSON = PROJECT / "output/demo/hkust_buildings.geojson"
TERRAIN_GLB = PROJECT / "output/demo/hkust_optimized.glb"  # vertex-colored Google Earth
OUTPUT_GLB = PROJECT / "output/demo/hkust_final.glb"
OUTPUT_LANDMARKS = PROJECT / "output/demo/hkust_landmarks.json"

HKUST_LON, HKUST_LAT = 114.263, 22.337
CAMPUS_RADIUS_M = 700

# ── Color palettes by OSM class ────────────────────────────
CLASS_COLORS = {
    "university":  np.array([230, 228, 225, 255]),  # white/light gray — Academic Building
    "dormitory":   np.array([215, 200, 180, 255]),  # warm beige — UG Halls
    "residential": np.array([220, 210, 190, 255]),  # light yellow — staff quarters
    "house":       np.array([210, 200, 175, 255]),  # cream — small houses
    "school":      np.array([210, 215, 225, 255]),  # light blue-gray
    "roof":        np.array([160, 155, 150, 255]),  # dark gray
    "parking":     np.array([140, 140, 140, 255]),  # gray
    "apartments":  np.array([215, 205, 190, 255]),  # light beige
    "commercial":  np.array([220, 215, 205, 255]),  # warm light
    "service":     np.array([180, 175, 170, 255]),  # neutral gray
    "warehouse":   np.array([170, 165, 160, 255]),  # darker gray
}
DEFAULT_BUILDING_COLOR = np.array([210, 195, 175, 255])


# ═══════════════════════════════════════════════════════════
# Part A: Terrain Loading
# ═══════════════════════════════════════════════════════════

def load_textured_terrain():
    """Load Google Earth terrain mesh with photorealistic vertex colors."""
    print("Loading terrain from hkust_optimized.glb ...")
    mesh = trimesh.load(str(TERRAIN_GLB), force="mesh")

    if not isinstance(mesh, trimesh.Trimesh):
        # Might be a Scene — grab first geometry
        if hasattr(mesh, 'geometry') and mesh.geometry:
            mesh = list(mesh.geometry.values())[0]
        else:
            raise RuntimeError("Cannot extract mesh from terrain GLB")

    print(f"  Vertices: {len(mesh.vertices):,}  Faces: {len(mesh.faces):,}")

    # Extract vertex colors if present
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vc = mesh.visual.vertex_colors.copy()
        print(f"  Vertex colors: YES  (range {vc.min()}-{vc.max()})")
    else:
        print("  Vertex colors: NO — will use height-based fallback")
        vc = None

    v = mesh.vertices.copy()
    f = mesh.faces.copy()

    # Terrain coordinate ranges
    tx_min, tx_max = v[:, 0].min(), v[:, 0].max()
    ty_min, ty_max = v[:, 1].min(), v[:, 1].max()
    tz = v[:, 2]
    tz_min, tz_max = tz.min(), tz.max()

    print(f"  Bounds: X=[{tx_min:.0f},{tx_max:.0f}] Y=[{ty_min:.0f},{ty_max:.0f}] Z=[{tz_min:.0f},{tz_max:.0f}]")

    # Water detection
    z_05 = np.percentile(tz, 5)
    z_20 = np.percentile(tz, 20)
    water_z = z_05

    # Build terrain with water flattening
    if vc is not None:
        terrain_colors = vc.copy()
        # Enhance water areas — make them more blue
        for i in range(len(v)):
            if tz[i] < z_05 + 2:
                orig = vc[i].astype(float)[:3]
                blue = np.array([40, 100, 180])
                terrain_colors[i, :3] = (orig * 0.3 + blue * 0.7).astype(np.uint8)
    else:
        # Height-based fallback
        terrain_colors = np.zeros((len(v), 4), dtype=np.uint8)
        terrain_colors[:, 3] = 255
        for i in range(len(v)):
            z = tz[i]
            if z < z_05 + 2:
                terrain_colors[i, :3] = [40, 100, 180]  # water blue
            elif z < z_20:
                t = (z - z_05) / max(z_20 - z_05, 0.01)
                terrain_colors[i, :3] = [int(60 + t * 30), int(120 + t * 30), int(50 + t * 30)]
            else:
                terrain_colors[i, :3] = [160, 150, 130]  # ground brown

    tv_flat = v.copy()
    below_water = tz < water_z
    tv_flat[below_water, 2] = water_z  # flatten water

    terrain_vis = trimesh.visual.ColorVisuals(vertex_colors=terrain_colors)
    terrain_mesh = trimesh.Trimesh(vertices=tv_flat, faces=f, visual=terrain_vis, process=False)

    return terrain_mesh, (tx_min, tx_max, ty_min, ty_max, tz_min, tz_max)


# ═══════════════════════════════════════════════════════════
# Part B: Building Polygons
# ═══════════════════════════════════════════════════════════

def extrude_polygon(coords_xy, z_base, height, color):
    """Extrude 2D polygon → colored 3D mesh (top + bottom + walls)."""
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

    # Walls from boundary edges
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


def load_buildings_with_names(terrain_bounds):
    """Load GeoJSON buildings, filter to campus, parse names & classes."""
    tx_min, tx_max, ty_min, ty_max, tz_min, tz_max = terrain_bounds

    lat_to_m = 111000.0
    lon_to_m = 111000.0 * np.cos(np.radians(HKUST_LAT))

    with open(GEOJSON) as f:
        geojson = json.load(f)

    raw_buildings = []
    named_count = 0

    for feat in geojson['features']:
        geom = feat.get('geometry', {})
        if geom.get('type') != 'Polygon':
            continue
        coords = geom['coordinates'][0]
        if len(coords) < 3:
            continue

        # Convert to meters relative to HKUST center
        xy = np.array([[(c[0] - HKUST_LON) * lon_to_m,
                        (c[1] - HKUST_LAT) * lat_to_m] for c in coords])
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        dist = np.sqrt(cx**2 + cy**2)
        if dist > CAMPUS_RADIUS_M:
            continue

        props = feat.get('properties', {}) or {}
        floors = props.get('num_floors', 0) or 0

        # Parse name
        name = None
        names_obj = props.get('names')
        if names_obj and isinstance(names_obj, dict):
            name = names_obj.get('primary')
        if not name and names_obj:
            # Try list format
            pass

        # Parse class/subtype
        bld_class = props.get('class') or None
        subtype = props.get('subtype') or None

        bld = {
            'xy': xy,
            'floors': floors,
            'dist': dist,
            'name': name,
            'class': bld_class,
            'subtype': subtype,
            'centroid_m': (cx, cy),
        }
        raw_buildings.append(bld)
        if name:
            named_count += 1

    print(f"Buildings in campus: {len(raw_buildings)}  (named: {named_count})")
    return raw_buildings


def building_color(bld):
    """Get color for a building based on its OSM class."""
    bld_class = bld.get('class')
    if bld_class and bld_class in CLASS_COLORS:
        return CLASS_COLORS[bld_class].copy()

    # Fallback: floor-based
    floors = bld['floors']
    if floors >= 8:
        return np.array([235, 220, 200, 255])
    elif floors >= 6:
        return np.array([225, 210, 190, 255])
    elif floors >= 4:
        return np.array([215, 200, 180, 255])
    elif floors >= 2:
        return np.array([205, 190, 170, 255])
    return DEFAULT_BUILDING_COLOR.copy()


# ═══════════════════════════════════════════════════════════
# Part C: Red Bird Sundial
# ═══════════════════════════════════════════════════════════

def create_sundial_model(sundial_x, sundial_y, ground_z):
    """Create a procedural Red Bird Sundial at the given terrain coordinates."""
    parts = []

    def colored_cylinder(radius, height, color, z_offset=0, sections=24):
        cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        cyl.vertices[:, 2] += height / 2 + z_offset  # lift so bottom is at z_offset
        cyl.vertices[:, 0] += sundial_x
        cyl.vertices[:, 1] += sundial_y
        vc = np.tile(np.array(color, dtype=np.uint8), (len(cyl.vertices), 1))
        cyl.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
        return cyl

    def colored_cone(radius, height, color, z_offset=0, sections=24):
        cone = trimesh.creation.cone(radius=radius, height=height, sections=sections)
        cone.vertices[:, 2] += z_offset
        cone.vertices[:, 0] += sundial_x
        cone.vertices[:, 1] += sundial_y
        vc = np.tile(np.array(color, dtype=np.uint8), (len(cone.vertices), 1))
        cone.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
        return cone

    # 1. Water pool base (large thin blue disc)
    pool = colored_cylinder(radius=8, height=0.3, color=[50, 120, 200, 255], z_offset=ground_z)
    parts.append(pool)

    # 2. Stone platform
    platform = colored_cylinder(radius=3.5, height=0.6, color=[180, 175, 165, 255], z_offset=ground_z + 0.3)
    parts.append(platform)

    # 3. Second tier platform
    tier2 = colored_cylinder(radius=2.0, height=0.4, color=[190, 185, 175, 255], z_offset=ground_z + 0.9)
    parts.append(tier2)

    # 4. Central steel pillar
    pillar_z = ground_z + 1.3
    pillar = colored_cylinder(radius=0.4, height=6.5, color=[100, 95, 90, 255], z_offset=pillar_z)
    parts.append(pillar)

    # 5. Sundial ring (gold/bronze) at mid-pillar
    ring_z = pillar_z + 2.5
    ring = colored_cylinder(radius=1.5, height=0.15, color=[200, 150, 60, 255], z_offset=ring_z)
    parts.append(ring)

    # 6. Flame/bird top — red-orange cone
    flame_z = pillar_z + 6.5
    flame = colored_cone(radius=1.2, height=2.5, color=[220, 60, 30, 255], z_offset=flame_z)
    parts.append(flame)

    # 7. Small decorative elements (smaller flames around)
    for angle in [0, 72, 144, 216, 288]:
        rad = np.radians(angle)
        dx = 0.7 * np.cos(rad)
        dy = 0.7 * np.sin(rad)
        small_flame = colored_cone(radius=0.35, height=1.2, color=[240, 100, 40, 255],
                                   z_offset=flame_z - 0.3)
        small_flame.vertices[:, 0] += dx
        small_flame.vertices[:, 1] += dy
        parts.append(small_flame)

    # Merge sundial parts
    sundial = trimesh.util.concatenate(parts)
    return sundial


# ═══════════════════════════════════════════════════════════
# Part D: Label Markers
# ═══════════════════════════════════════════════════════════

def create_label_markers(buildings, terrain_bounds, tree, tv):
    """Create small colored spheres above named buildings."""
    tx_min, tx_max, ty_min, ty_max, tz_min, tz_max = terrain_bounds

    def bx_to_tx(bx):
        return tx_min + (bx - bx_min_local) / max(bx_max_local - bx_min_local, 1) * (tx_max - tx_min)

    def by_to_ty(by):
        return ty_min + (by - by_min_local) / max(by_max_local - by_min_local, 1) * (ty_max - ty_min)

    # Compute building XY ranges for mapping
    all_bx = np.concatenate([b['xy'][:, 0] for b in buildings])
    all_by = np.concatenate([b['xy'][:, 1] for b in buildings])
    bx_min_local, bx_max_local = all_bx.min(), all_bx.max()
    by_min_local, by_max_local = all_by.min(), all_by.max()

    def get_terrain_z(tx, ty):
        dist, idx = tree.query([tx, ty], k=3)
        if hasattr(dist, '__len__'):
            weights = 1.0 / (dist + 0.01)
            weights /= weights.sum()
            z = np.sum(tv[idx, 2] * weights)
        else:
            z = tv[idx, 2]
        return float(z)

    key_landmarks = {
        'Academic Building': 'red',
        '學術大樓': 'red',
        'Library': 'gold',
        '圖書館': 'gold',
        'Shaw Auditorium': 'magenta',
        'Conference Lodge': 'gold',
    }

    markers = []
    landmarks_data = []

    for b in buildings:
        name = b.get('name')
        if not name:
            continue

        cx_m, cy_m = b['centroid_m']
        tx = bx_to_tx(cx_m)
        ty = by_to_ty(cy_m)
        tz = get_terrain_z(tx, ty)

        height = max(b['floors'] * 3.5, 5.0) if b['floors'] > 0 else 6.0
        height *= 0.8
        marker_z = tz + height + 3.0  # 3m above roof

        # Determine marker color
        marker_color = np.array([100, 150, 255, 255])  # default blue
        for keyword, color_name in key_landmarks.items():
            if keyword.lower() in name.lower():
                if color_name == 'red':
                    marker_color = np.array([255, 60, 60, 255])
                elif color_name == 'gold':
                    marker_color = np.array([255, 200, 50, 255])
                elif color_name == 'magenta':
                    marker_color = np.array([255, 60, 200, 255])
                break

        # Create marker sphere
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.2)
        sphere.vertices[:, 0] += tx
        sphere.vertices[:, 1] += ty
        sphere.vertices[:, 2] += marker_z
        vc = np.tile(marker_color.astype(np.uint8), (len(sphere.vertices), 1))
        sphere.visual = trimesh.visual.ColorVisuals(vertex_colors=vc)
        markers.append(sphere)

        # Record for JSON
        landmarks_data.append({
            'name': name,
            'class': b.get('class', ''),
            'floors': b['floors'],
            'marker_x': float(tx),
            'marker_y': float(ty),
            'marker_z': float(marker_z),
        })

    print(f"Label markers: {len(markers)}")
    return markers, landmarks_data


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    # ── Load terrain ──────────────────────────────────────
    terrain_mesh, tb = load_textured_terrain()
    tv = terrain_mesh.vertices
    tx_min, tx_max, ty_min, ty_max, tz_min, tz_max = tb

    # Build KD-tree for terrain Z lookup
    terrain_xy = tv[:, :2]
    tree = cKDTree(terrain_xy)

    def get_terrain_z(tx, ty):
        dist, idx = tree.query([tx, ty], k=3)
        if hasattr(dist, '__len__'):
            weights = 1.0 / (dist + 0.01)
            weights /= weights.sum()
            z = np.sum(tv[idx, 2] * weights)
        else:
            z = tv[idx, 2]
        return float(z)

    # ── Load buildings ────────────────────────────────────
    buildings = load_buildings_with_names(tb)
    if not buildings:
        print("No buildings found! Aborting.")
        return

    # Coordinate mapping: building XY → terrain XY
    all_bx = np.concatenate([b['xy'][:, 0] for b in buildings])
    all_by = np.concatenate([b['xy'][:, 1] for b in buildings])
    bx_min, bx_max = all_bx.min(), all_bx.max()
    by_min, by_max = all_by.min(), all_by.max()

    def bx_to_tx(bx):
        return tx_min + (bx - bx_min) / max(bx_max - bx_min, 1) * (tx_max - tx_min)

    def by_to_ty(by):
        return ty_min + (by - by_min) / max(by_max - by_min, 1) * (ty_max - ty_min)

    # ── Sundial placement ─────────────────────────────────
    # Red Bird Sundial GPS: (22.3375, 114.263)
    lat_to_m = 111000.0
    lon_to_m = 111000.0 * np.cos(np.radians(HKUST_LAT))
    sundial_mx = (114.263 - HKUST_LON) * lon_to_m
    sundial_my = (22.3375 - HKUST_LAT) * lat_to_m
    sundial_tx = bx_to_tx(sundial_mx)
    sundial_ty = by_to_ty(sundial_my)
    sundial_tz = get_terrain_z(sundial_tx, sundial_ty)

    print(f"\nRed Bird Sundial:")
    print(f"  GPS: (22.3375, 114.263)")
    print(f"  Meters from HKUST center: ({sundial_mx:.0f}, {sundial_my:.0f})")
    print(f"  Terrain coords: ({sundial_tx:.0f}, {sundial_ty:.0f}, {sundial_tz:.0f})")

    sundial = create_sundial_model(sundial_tx, sundial_ty, sundial_tz)
    print(f"  Sundial mesh: {len(sundial.vertices):,}v, {len(sundial.faces):,}f")

    # ── Extrude buildings ──────────────────────────────────
    print(f"\nExtruding {len(buildings)} buildings on terrain...")
    building_meshes = []

    for i, b in enumerate(buildings):
        xy_t = np.array([[bx_to_tx(c[0]), by_to_ty(c[1])] for c in b['xy']])
        cx_t = xy_t[:, 0].mean()
        cy_t = xy_t[:, 1].mean()
        z_base = get_terrain_z(cx_t, cy_t)

        floors = b['floors']
        height = max(floors * 3.5, 5.0) if floors > 0 else 6.0
        height *= 0.8

        color = building_color(b)
        mesh = extrude_polygon(xy_t, z_base, height, color)
        if mesh is not None:
            building_meshes.append(mesh)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(buildings)}]")

    print(f"Extruded: {len(building_meshes)} buildings")

    # ── Label markers ──────────────────────────────────────
    markers, landmarks_data = create_label_markers(buildings, tb, tree, tv)

    # ── Merge everything ──────────────────────────────────
    print("\nMerging terrain + buildings + sundial + markers...")
    all_meshes = [terrain_mesh] + building_meshes + [sundial] + markers
    final = trimesh.util.concatenate(all_meshes)
    final.update_faces(final.nondegenerate_faces())
    final.remove_unreferenced_vertices()

    # Center and scale
    final.vertices -= final.centroid
    extent = final.vertices.max(axis=0) - final.vertices.min(axis=0)
    final.vertices *= 200.0 / max(extent)

    print(f"Final: {len(final.vertices):,}v, {len(final.faces):,}f")

    # ── Export GLB ─────────────────────────────────────────
    final.export(str(OUTPUT_GLB))
    mb = OUTPUT_GLB.stat().st_size / 1e6
    print(f"\n✓ {OUTPUT_GLB} ({mb:.1f} MB)")

    # Verify
    with open(OUTPUT_GLB, 'rb') as f:
        data = f.read()
    js_len = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20+js_len])
    attrs = gltf['meshes'][0]['primitives'][0]['attributes']
    vc = gltf['accessors'][attrs['POSITION']]['count']
    has_color = 'COLOR_0' in attrs
    print(f"Verify: {vc:,}v, COLOR_0={has_color}")

    # ── Export landmarks JSON ──────────────────────────────
    with open(OUTPUT_LANDMARKS, 'w', encoding='utf-8') as f:
        json.dump(landmarks_data, f, ensure_ascii=False, indent=2)
    print(f"✓ {OUTPUT_LANDMARKS} ({len(landmarks_data)} landmarks)")

    # ── Named building summary ─────────────────────────────
    print(f"\n🏛️  Key named buildings found:")
    for lm in landmarks_data[:15]:
        class_str = f" [{lm['class']}]" if lm['class'] else ""
        floors_str = f" ({lm['floors']}F)" if lm['floors'] else ""
        print(f"  • {lm['name']}{class_str}{floors_str}")
    if len(landmarks_data) > 15:
        print(f"  ... and {len(landmarks_data) - 15} more")

    print(f"""
✅ 完成! Open https://gltf-viewer.donmccurdy.com/ and drag in:
   {OUTPUT_GLB}

🔴 Red Bird Sundial: Red flame sculpture at entrance piazza
🔵 Blue markers: Named buildings
🟡 Gold markers: Library, Conference Lodge
🔴 Red markers: Academic Building
🟣 Magenta marker: Shaw Auditorium

🏛️  Building colors:
   White = Academic/University buildings
   Beige = Dormitories/Halls
   Yellow = Residential/Staff quarters
   Gray = Parking/Roof/Service structures

📋 Landmark list: {OUTPUT_LANDMARKS}
""")


if __name__ == "__main__":
    main()

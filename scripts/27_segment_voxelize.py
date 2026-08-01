#!/usr/bin/env python3
"""
Segment a textured GLB model into Buildings / Terrain / Water,
then voxelize each category with distinct Minecraft-style coloring.

Buildings: hollow surface-only voxels, wall/roof color distinction
Terrain:  solid ground with height-based green→brown gradient
Water:    flat blue surface at water level (models with water bodies only)

Usage:
  python3 scripts/27_segment_voxelize.py --input output/demo/hkust_piazza.glb
  python3 scripts/27_segment_voxelize.py --all
"""

import argparse, json, struct, sys, time, os, io
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from scipy.spatial import KDTree
except ImportError:
    sys.exit("pip install scipy")

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    sys.exit("pip install Pillow")

import trimesh

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "output/demo"
MODELS = {
    "piazza":   OUTPUT_DIR / "hkust_piazza.glb",
    "academic": OUTPUT_DIR / "hkust_academic.glb",
    "seaside":  OUTPUT_DIR / "hkust_seaside.glb",
    "atrium":   OUTPUT_DIR / "hkust_atrium.glb",
}

# ── Tunable parameters ──────────────────────────────────────
PITCH_BUILDING = 0.5       # Building block size (finer for architectural detail)
PITCH_TERRAIN  = 1.0       # Terrain block size
PITCH_WATER    = 2.0       # Water block size (larger = fewer blocks)

WALL_NZ_THRESHOLD  = 0.35  # |normal_z| below this = vertical = wall
ROOF_NZ_THRESHOLD  = 0.75  # |normal_z| above this = horizontal = roof/ground
FLAT_NZ_THRESHOLD  = 0.85  # |normal_z| above this = very flat (water candidate)
GROUND_NZ_MIN      = 0.55  # |normal_z| above this = roughly horizontal (ground candidate)

MIN_BUILDING_FACES = 30    # Minimum faces for a valid building component
MIN_BUILDING_HEIGHT = 3.0  # Minimum Z span for a valid building

SURFACE_SAMPLES = 500_000  # Dense surface samples for color lookup
TILE_SIZE = 512

# ── Category constants ──────────────────────────────────────
LABEL_TERRAIN  = 0
LABEL_WATER    = 1
LABEL_BUILDING = 2

# ── Color schemes (RGB) ─────────────────────────────────────
COLOR_WALL   = np.array([215, 205, 190])   # Warm beige
COLOR_ROOF   = np.array([160, 150, 135])   # Darker gray
COLOR_GROUND_LOW  = np.array([90, 125, 55])    # Green
COLOR_GROUND_MID  = np.array([135, 120, 80])   # Brown-green
COLOR_GROUND_HIGH = np.array([165, 150, 135])  # Gray-brown
COLOR_WATER  = np.array([50, 140, 210])    # Blue

JITTER = 8  # Per-block color jitter range


# ═══════════════════════════════════════════════════════════════
#  GLB Parser (reused from 26_voxelize_textured.py)
# ═══════════════════════════════════════════════════════════════

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


def read_accessor(acc_idx, accessors, buffer_views, bin_data):
    if acc_idx is None:
        return np.array([])
    acc = accessors[acc_idx]
    bv_idx = acc.get("bufferView")
    if bv_idx is None:
        return np.array([])
    bv = buffer_views[bv_idx]
    off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    comp_type = acc["componentType"]
    acc_type = acc["type"]
    tc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    ncomp = tc[acc_type]
    total = count * ncomp
    dtype = {
        5120: np.int8, 5121: np.uint8, 5122: np.int16,
        5123: np.uint16, 5125: np.uint32, 5126: np.float32,
    }[comp_type]
    raw = np.frombuffer(bin_data, dtype=dtype, count=total, offset=off)
    if acc_type in ("VEC2", "VEC3", "VEC4"):
        raw = raw.reshape(-1, ncomp)
    return raw.copy()


def extract_valid_mesh(glb_path):
    """Extract vertices, faces, UVs, and texture from GLB. Skips degenerate primitives."""
    gltf, bin_data = parse_glb(glb_path)
    accessors = gltf["accessors"]
    buffer_views = gltf["bufferViews"]
    meshes = gltf["meshes"]

    all_verts = []
    all_faces = []
    all_uvs = []
    face_offset = 0

    for mesh in meshes:
        for prim in mesh.get("primitives", []):
            attrs = prim.get("attributes", {})
            v = read_accessor(attrs.get("POSITION"), accessors, buffer_views, bin_data)
            f = read_accessor(prim.get("indices"), accessors, buffer_views, bin_data)

            # Skip degenerate primitives
            if len(v) == 0 or len(f) == 0:
                continue
            if np.any(~np.isfinite(v)):
                continue
            if np.any(np.abs(v) > 1e7):
                continue

            uv_acc = attrs.get("TEXCOORD_0")
            uv = read_accessor(uv_acc, accessors, buffer_views, bin_data) if uv_acc is not None else np.array([])

            all_verts.append(v)
            if f.ndim == 2:
                # Triangulate if needed
                if f.shape[1] == 3:
                    all_faces.append(f + face_offset)
                elif f.shape[1] == 4:
                    tri_f = np.column_stack([f[:, [0,1,2]], f[:, [0,2,3]]]).reshape(-1, 3)
                    all_faces.append(tri_f + face_offset)
                elif f.shape[1] > 4:
                    # Fan triangulation
                    tris = []
                    for j in range(1, f.shape[1] - 1):
                        tris.append(np.column_stack([f[:, 0], f[:, j], f[:, j+1]]))
                    all_faces.append(np.vstack(tris) + face_offset)
            else:
                all_faces.append(f.reshape(-1, 3) + face_offset)
            if len(uv) > 0:
                all_uvs.append(uv)

            face_offset += len(v)

    if not all_verts:
        sys.exit(f"No valid geometry in {glb_path}")

    verts = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    uvs = np.vstack(all_uvs) if all_uvs else np.array([])

    # Filter invalid face indices
    valid_f = (faces >= 0) & (faces < len(verts))
    faces = faces[valid_f.all(axis=1)]

    # Load texture
    tex_image = None
    images = gltf.get("images", [])
    if images:
        for img in images:
            bv_idx = img.get("bufferView")
            mime = img.get("mimeType", "")
            if bv_idx is not None:
                bv = buffer_views[bv_idx]
                off = bv.get("byteOffset", 0)
                length = bv["byteLength"]
                img_data = bin_data[off:off+length]
                try:
                    tex_image = Image.open(io.BytesIO(img_data))
                    break
                except Exception:
                    pass

    return verts, faces, uvs, tex_image


# ═══════════════════════════════════════════════════════════════
#  Voxelization (adapted from 26_voxelize_textured.py)
# ═══════════════════════════════════════════════════════════════

def point_in_triangle_2d(pt, a, b, c, dominant_axis):
    if dominant_axis == 0:
        ax, ay = a[1], a[2]; bx, by = b[1], b[2]; cx, cy = c[1], c[2]
        px, py = pt[1], pt[2]
    elif dominant_axis == 1:
        ax, ay = a[0], a[2]; bx, by = b[0], b[2]; cx, cy = c[0], c[2]
        px, py = pt[0], pt[2]
    else:
        ax, ay = a[0], a[1]; bx, by = b[0], b[1]; cx, cy = c[0], c[1]
        px, py = pt[0], pt[1]

    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-12:
        return False
    alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    beta  = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    gamma = 1 - alpha - beta
    return (alpha >= -1e-9) and (beta >= -1e-9) and (gamma >= -1e-9)


def voxelize_numpy(verts, faces, pitch):
    t0 = time.time()
    vmin = verts.min(axis=0).astype(np.float64)
    vmax = verts.max(axis=0).astype(np.float64)
    span = vmax - vmin
    if np.any(span < 0) or np.any(~np.isfinite(span)):
        raise ValueError("Invalid vertex bounds")
    dims = np.ceil(span / pitch).astype(int) + 2
    print(f"    Bounds: [{vmin[0]:.0f},{vmax[0]:.0f}] [{vmin[1]:.0f},{vmax[1]:.0f}] "
          f"[{vmin[2]:.0f},{vmax[2]:.0f}]")
    print(f"    Grid: {dims[0]}×{dims[1]}×{dims[2]} ({dims.prod()/1e6:.1f}M cells)")
    grid = np.zeros(dims, dtype=bool)
    offset = vmin - pitch * 0.5

    tri_verts = verts[faces]
    total_faces = len(faces)
    report_every = max(1, total_faces // 5)

    for fi in range(total_faces):
        v0, v1, v2 = tri_verts[fi]
        e1 = v1 - v0; e2 = v2 - v0
        normal = np.cross(e1, e2)
        dominant = np.argmax(np.abs(normal))

        tmin = np.minimum(np.minimum(v0, v1), v2)
        tmax = np.maximum(np.maximum(v0, v1), v2)
        imin = np.floor((tmin - offset) / pitch).astype(int)
        imax = np.ceil((tmax - offset) / pitch).astype(int) + 1
        imin = np.clip(imin, 0, [d-1 for d in dims])
        imax = np.clip(imax, 0, [d-1 for d in dims])

        area = np.linalg.norm(normal) / 2.0
        cell_area = pitch * pitch
        if area < cell_area * 0.1:
            centroid = (v0 + v1 + v2) / 3
            ci = tuple(np.floor((centroid - offset) / pitch).astype(int))
            if all(0 <= ci[j] < dims[j] for j in range(3)):
                grid[ci] = True
            continue

        for ix in range(imin[0], imax[0] + 1):
            for iy in range(imin[1], imax[1] + 1):
                for iz in range(imin[2], imax[2] + 1):
                    cell_center = offset + (np.array([ix, iy, iz]) + 0.5) * pitch
                    if point_in_triangle_2d(cell_center, v0, v1, v2, dominant):
                        grid[ix, iy, iz] = True

        if (fi + 1) % report_every == 0:
            print(f"    ... {fi+1}/{total_faces} faces")

    n_occ = int(np.sum(grid))
    print(f"    Occupied: {n_occ:,} voxels ({time.time()-t0:.1f}s)")
    return grid, offset, dims


def extract_surface_voxels(grid):
    t0 = time.time()
    surface = np.zeros(grid.shape, dtype=bool)
    for axis in range(3):
        surface |= grid & ~np.roll(grid, shift=1, axis=axis)
        surface |= grid & ~np.roll(grid, shift=-1, axis=axis)
    n_surf = int(np.sum(surface))
    if n_surf == 0:
        print("    ⚠ No surface voxels, using all")
        surface = grid
        n_surf = int(np.sum(surface))
    print(f"    Surface: {n_surf:,} / {int(np.sum(grid)):,} voxels "
          f"({100*n_surf/max(1,int(np.sum(grid))):.0f}%) ({time.time()-t0:.1f}s)")
    return surface


# ═══════════════════════════════════════════════════════════════
#  Face Classification
# ═══════════════════════════════════════════════════════════════

def classify_faces(mesh):
    """Label each face as TERRAIN, WATER, or BUILDING.

    Strategy (priority order):
      1. WATER:   very low Z, nearly horizontal, large flat faces
      2. BUILDING: elevated, vertical (walls) or elevated horizontal (roofs)
      3. TERRAIN: everything else (low/mid elevation, horizontal or sloped)
    """
    centroids = mesh.triangles_center
    normals = mesh.face_normals
    verts_in_faces = mesh.vertices[mesh.faces]  # (n_faces, 3, 3)

    centroid_z = centroids[:, 2]
    abs_nz = np.abs(normals[:, 2])

    # Face Z range (max-min Z of 3 vertices)
    z_per_face = verts_in_faces[:, :, 2]
    z_range = z_per_face.max(axis=1) - z_per_face.min(axis=1)

    # Global height profile
    all_z = mesh.vertices[:, 2]
    z_05 = np.percentile(all_z, 5)
    z_15 = np.percentile(all_z, 15)
    z_85 = np.percentile(all_z, 85)

    ground_base_z = z_15
    building_min_z = ground_base_z + 3.0
    water_max_z = z_05 + 3.0

    n_faces = len(mesh.faces)
    labels = np.full(n_faces, -1, dtype=int)

    print(f"  Height profile: P5={z_05:.1f}  P15={z_15:.1f}  P85={z_85:.1f}")
    print(f"  Ground base: {ground_base_z:.1f}  Building min: {building_min_z:.1f}  "
          f"Water max: {water_max_z:.1f}")

    # ── WATER: low elevation, very flat ──
    water_mask = (
        (centroid_z < water_max_z) &
        (abs_nz > FLAT_NZ_THRESHOLD) &
        (z_range < 1.5)
    )
    labels[water_mask] = LABEL_WATER

    # ── BUILDING WALLS: elevated + vertical ──
    wall_mask = (
        (centroid_z > building_min_z) &
        (abs_nz < WALL_NZ_THRESHOLD) &
        (labels == -1)
    )
    labels[wall_mask] = LABEL_BUILDING

    # ── BUILDING ROOFS: elevated + horizontal + small-ish ──
    roof_mask = (
        (centroid_z > building_min_z) &
        (abs_nz > ROOF_NZ_THRESHOLD) &
        (labels == -1)
    )
    labels[roof_mask] = LABEL_BUILDING

    # ── TERRAIN: low/mid + horizontal ──
    terrain_mask = (
        (centroid_z < building_min_z) &
        (abs_nz > GROUND_NZ_MIN) &
        (labels == -1)
    )
    labels[terrain_mask] = LABEL_TERRAIN

    # ── Remaining = SLOPE / transitional ──
    # Assign via spatial nearest-neighbor vote from already-labeled faces
    unlabeled = labels == -1
    if unlabeled.sum() > 0:
        print(f"  Resolving {unlabeled.sum():,} unlabeled faces via spatial voting...")
        labeled_mask = ~unlabeled
        labeled_centroids = centroids[labeled_mask]
        labeled_labels = labels[labeled_mask]
        tree = KDTree(labeled_centroids)
        unlabeled_centroids = centroids[unlabeled]
        _, knn_idx = tree.query(unlabeled_centroids, k=5)
        # Majority vote among k nearest labeled neighbors
        knn_labels = labeled_labels[knn_idx]
        for i, ul_idx in enumerate(np.where(unlabeled)[0]):
            votes = np.bincount(knn_labels[i], minlength=3)
            labels[ul_idx] = np.argmax(votes)

    # ── Print summary ──
    for name, val in [("TERRAIN", LABEL_TERRAIN), ("WATER", LABEL_WATER),
                       ("BUILDING", LABEL_BUILDING)]:
        count = (labels == val).sum()
        print(f"  {name}: {count:,} faces ({100*count/n_faces:.1f}%)")

    return labels, centroid_z, abs_nz


# ═══════════════════════════════════════════════════════════════
#  Building Separation
# ═══════════════════════════════════════════════════════════════

def mesh_from_face_subset(mesh, face_indices):
    """Build a new trimesh from a subset of faces. Avoids broken submesh()."""
    if len(face_indices) == 0:
        return None
    sub_faces = mesh.faces[face_indices]
    unique_verts, new_faces = np.unique(sub_faces.ravel(), return_inverse=True)
    new_faces = new_faces.reshape(-1, 3)
    new_verts = mesh.vertices[unique_verts]
    return trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)


def separate_buildings(mesh, face_labels):
    """Split merged building faces into individual buildings by DBSCAN clustering.

    Photogrammetry meshes are continuous surfaces where classified building faces
    are interleaved with non-building faces. Connected-components on face adjacency
    fragments them. DBSCAN on face centroids in XY + Z space gives robust clusters.
    """
    building_mask = face_labels == LABEL_BUILDING
    building_faces = np.where(building_mask)[0]

    if len(building_faces) < MIN_BUILDING_FACES:
        print("  No significant building faces found")
        return [], []

    print(f"  Clustering {len(building_faces):,} building faces...")
    t0 = time.time()

    # DBSCAN on face centroid positions (XY for footprint + Z for height)
    all_centroids = mesh.triangles_center
    building_centroids = all_centroids[building_faces]

    # Normalize dimensions for clustering (Z should not dominate XY distance)
    features = building_centroids.copy()
    features[:, 2] *= 0.3  # Downweight Z so vertical stacking doesn't split buildings

    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=15.0, min_samples=MIN_BUILDING_FACES)
    cluster_ids = db.fit_predict(features)
    unique_clusters = set(cluster_ids) - {-1}  # exclude noise

    print(f"  {len(unique_clusters)} clusters + {int((cluster_ids == -1).sum())} noise faces")

    # Build a mesh for each cluster and filter
    valid_buildings = []
    building_face_ids = []

    for cid in sorted(unique_clusters):
        c_mask = cluster_ids == cid
        c_local_indices = np.where(c_mask)[0]  # indices within building_faces
        c_global_faces = building_faces[c_local_indices]

        n_f = len(c_global_faces)
        if n_f < MIN_BUILDING_FACES:
            continue

        # Build mesh for this cluster
        comp = mesh_from_face_subset(mesh, c_global_faces)
        if comp is None:
            continue

        z_span = comp.vertices[:, 2].ptp()
        if z_span < MIN_BUILDING_HEIGHT:
            continue

        valid_buildings.append(comp)
        building_face_ids.append(set(c_global_faces))

    # Sort by size (largest first)
    sorted_pairs = sorted(zip(valid_buildings, building_face_ids),
                          key=lambda x: len(x[0].faces), reverse=True)
    valid_buildings = [p[0] for p in sorted_pairs]
    building_face_ids = [p[1] for p in sorted_pairs]

    print(f"  Buildings found: {len(valid_buildings)} ({time.time()-t0:.1f}s)")
    for i, b in enumerate(valid_buildings[:5]):
        z = b.vertices[:, 2]
        print(f"    #{i}: {len(b.faces):,} faces, Z[{z.min():.0f},{z.max():.0f}] "
              f"H={z.ptp():.0f}m")

    return valid_buildings, building_face_ids


# ═══════════════════════════════════════════════════════════════
#  Surface Sampling & Color Assignment
# ═══════════════════════════════════════════════════════════════

def sample_mesh_surface(verts, faces, uvs, tex_image, n_samples):
    """Densely sample mesh surface with UV texture colors."""
    t0 = time.time()
    tri_verts = verts[faces]

    n_samples = min(n_samples, len(faces) * 50)

    v0 = tri_verts[:, 1] - tri_verts[:, 0]
    v1 = tri_verts[:, 2] - tri_verts[:, 0]
    cross = np.cross(v0, v1)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    areas = np.maximum(areas, 1e-12)
    probs = areas / areas.sum()

    rng = np.random.RandomState(42)
    chosen_faces = rng.choice(len(faces), size=n_samples, p=probs)

    r1 = np.sqrt(rng.random(n_samples))
    r2 = rng.random(n_samples)
    u_param = 1 - r1
    v_param = r2 * r1
    w_param = 1 - u_param - v_param

    fv = tri_verts[chosen_faces]
    sample_pts = (fv[:, 0] * u_param[:, None] +
                  fv[:, 1] * v_param[:, None] +
                  fv[:, 2] * w_param[:, None])

    # Sample texture
    colors = np.full((n_samples, 3), 128, dtype=np.uint8)

    if tex_image is not None and len(uvs) > 0:
        tex_w, tex_h = tex_image.size
        tex_array = np.array(tex_image)
        if tex_array.ndim == 2:
            tex_array = np.stack([tex_array]*3, axis=-1)
        num_tiles = tex_w // TILE_SIZE

        # Interpolate UV from face vertices
        f_uvs = uvs[faces[chosen_faces]]
        interp_u = (f_uvs[:, 0, 0] * u_param +
                    f_uvs[:, 1, 0] * v_param +
                    f_uvs[:, 2, 0] * w_param)
        interp_v = (f_uvs[:, 0, 1] * u_param +
                    f_uvs[:, 1, 1] * v_param +
                    f_uvs[:, 2, 1] * w_param)

        for i in range(n_samples):
            u, v = interp_u[i], interp_v[i]
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            tile_idx = int(np.clip(np.floor(u * num_tiles), 0, num_tiles - 1))
            frac_u = u * num_tiles - tile_idx
            frac_v = v * 2.0
            px = int(tile_idx * TILE_SIZE + frac_u * TILE_SIZE)
            py = int(frac_v * TILE_SIZE)
            px = np.clip(px, 0, tex_w - 1)
            py = np.clip(py, 0, tex_h - 1)
            pixel = tex_array[py, px]
            colors[i] = pixel[:3]

    print(f"  Surface samples: {n_samples:,} points ({time.time()-t0:.1f}s)")
    return sample_pts, colors


# ═══════════════════════════════════════════════════════════════
#  Per-Category Voxelization & Coloring
# ═══════════════════════════════════════════════════════════════

def voxel_positions_from_grid(grid, offset, pitch):
    """Convert boolean grid to N×3 position array."""
    idx = np.argwhere(grid)
    positions = offset + (idx + 0.5) * pitch
    return positions


def assign_colors_kdtree(voxel_positions, sample_points, sample_colors):
    """Assign colors to voxels via nearest surface sample."""
    t0 = time.time()
    tree = KDTree(sample_points)
    _, nn = tree.query(voxel_positions, k=1)
    colors = sample_colors[nn].copy()
    # Add deterministic jitter
    rng = np.random.RandomState(hash(tuple(map(int, voxel_positions[0]))) % (2**31))
    jitter = rng.randint(-JITTER, JITTER+1, size=colors.shape, dtype=np.int16)
    colors = np.clip(colors.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
    print(f"  KDTree color assignment: {len(voxel_positions):,} voxels ({time.time()-t0:.1f}s)")
    return colors


def apply_category_colors(positions, tex_colors, category):
    """Blend texture colors with category theme colors."""
    if category == "building_wall":
        theme = COLOR_WALL
        blend = 0.4
    elif category == "building_roof":
        theme = COLOR_ROOF
        blend = 0.5
    elif category == "terrain":
        # Height-based gradient
        z = positions[:, 2]
        if len(z) > 0:
            z_min, z_max = z.min(), z.max()
            if z_max > z_min:
                z_norm = np.clip((z - z_min) / (z_max - z_min), 0, 1)
                theme_low = COLOR_GROUND_LOW
                theme_high = COLOR_GROUND_HIGH
                thematic = (theme_low * (1 - z_norm[:, None]) +
                            theme_high * z_norm[:, None])
            else:
                thematic = np.tile(COLOR_GROUND_MID, (len(positions), 1))
        else:
            thematic = np.tile(COLOR_GROUND_MID, (len(positions), 1))
        blend = 0.6
        result = np.clip(tex_colors.astype(float) * (1 - blend) + thematic * blend, 0, 255).astype(np.uint8)
        return result
    elif category == "water":
        theme = COLOR_WATER
        blend = 0.9
    else:
        return tex_colors

    theme = np.tile(theme.astype(float), (len(positions), 1))
    result = np.clip(tex_colors.astype(float) * (1 - blend) + theme * blend, 0, 255).astype(np.uint8)
    return result


def voxelize_buildings(building_meshes, sample_pts, sample_colors):
    """Voxelize each building as hollow surface with wall/roof color distinction."""
    all_positions = []
    all_colors = []
    building_bboxes = []

    for bi, bld_mesh in enumerate(building_meshes):
        if len(bld_mesh.faces) < MIN_BUILDING_FACES:
            continue

        print(f"\n  Building #{bi}: {len(bld_mesh.vertices):,}v, {len(bld_mesh.faces):,}f")
        try:
            grid, offset, dims = voxelize_numpy(bld_mesh.vertices, bld_mesh.faces, PITCH_BUILDING)
        except Exception as e:
            print(f"    Voxelization failed: {e}, skipping")
            continue

        surface = extract_surface_voxels(grid)
        positions = voxel_positions_from_grid(surface, offset, PITCH_BUILDING)

        if len(positions) == 0:
            continue

        # Classify each voxel as wall or roof
        # Wall: |normal_z| at nearest face is small (vertical face nearby)
        # Roof: |normal_z| at nearest face is large (horizontal face nearby)
        centroids_nz = np.abs(bld_mesh.face_normals[:, 2])
        centroid_pos = bld_mesh.triangles_center
        tree = KDTree(centroid_pos)
        _, nn = tree.query(positions, k=1)
        nn_nz = centroids_nz[nn]
        is_roof = nn_nz > ROOF_NZ_THRESHOLD
        is_wall = ~is_roof

        # Assign texture colors first
        tex_colors = assign_colors_kdtree(positions, sample_pts, sample_colors)

        # Apply category colors
        wall_positions = positions[is_wall]
        roof_positions = positions[is_roof]
        wall_tex = tex_colors[is_wall]
        roof_tex = tex_colors[is_roof]

        wall_colors = apply_category_colors(wall_positions, wall_tex, "building_wall")
        roof_colors = apply_category_colors(roof_positions, roof_tex, "building_roof")

        if len(wall_positions) > 0:
            all_positions.extend(wall_positions.tolist())
            all_colors.extend(wall_colors.tolist())
        if len(roof_positions) > 0:
            all_positions.extend(roof_positions.tolist())
            all_colors.extend(roof_colors.tolist())

        bbox = {"min": bld_mesh.vertices.min(axis=0).tolist(),
                "max": bld_mesh.vertices.max(axis=0).tolist()}
        building_bboxes.append(bbox)
        print(f"    Voxels: {len(positions):,} (walls: {is_wall.sum():,}, roofs: {is_roof.sum():,})")

    return all_positions, all_colors, building_bboxes


def voxelize_terrain(mesh, terrain_face_indices, sample_pts, sample_colors):
    """Voxelize terrain faces as surface blocks with height-based coloring."""
    print(f"\n  Terrain: {len(terrain_face_indices):,} faces")
    t0 = time.time()

    # Extract terrain submesh
    terrain_mesh = mesh_from_face_subset(mesh, terrain_face_indices)
    if terrain_mesh is None:
        print("  Too few terrain faces, skipping")
        return [], []
    try:
        grid, offset, dims = voxelize_numpy(terrain_mesh.vertices, terrain_mesh.faces, PITCH_TERRAIN)
    except Exception as e:
        print(f"    Voxelization failed: {e}")
        return [], []

    surface = extract_surface_voxels(grid)
    positions = voxel_positions_from_grid(surface, offset, PITCH_TERRAIN)

    if len(positions) == 0:
        return [], []

    tex_colors = assign_colors_kdtree(positions, sample_pts, sample_colors)
    terrain_colors = apply_category_colors(positions, tex_colors, "terrain")

    print(f"    Voxels: {len(positions):,} ({time.time()-t0:.1f}s)")
    return positions.tolist(), terrain_colors.tolist()


def voxelize_water(mesh, water_face_indices, sample_pts, sample_colors):
    """Create flat water surface at water level."""
    if len(water_face_indices) < 10:
        print("  No significant water faces, skipping")
        return [], []

    print(f"\n  Water: {len(water_face_indices):,} faces")

    centroids = mesh.triangles_center[water_face_indices]
    water_level = np.median(centroids[:, 2])

    # Get XY bounds from water faces
    water_verts = mesh.vertices[np.unique(mesh.faces[water_face_indices].flatten())]
    x_min, y_min, _ = water_verts.min(axis=0)
    x_max, y_max, _ = water_verts.max(axis=0)

    # Generate grid at water level
    nx = int((x_max - x_min) / PITCH_WATER) + 2
    ny = int((y_max - y_min) / PITCH_WATER) + 2

    positions = []
    for ix in range(nx):
        for iy in range(ny):
            wx = x_min + (ix + 0.5) * PITCH_WATER
            wy = y_min + (iy + 0.5) * PITCH_WATER
            positions.append([wx, wy, water_level])

    positions = np.array(positions)

    if len(positions) == 0:
        return [], []

    # Blue color with jitter
    rng = np.random.RandomState(42)
    colors = np.tile(COLOR_WATER.astype(float), (len(positions), 1))
    jitter = rng.randint(-JITTER, JITTER+1, size=colors.shape, dtype=np.int16)
    colors = np.clip(colors + jitter, 0, 255).astype(np.uint8)

    print(f"    Water level: Z={water_level:.1f}, {len(positions)} blocks "
          f"[{x_min:.0f},{x_max:.0f}]×[{y_min:.0f},{y_max:.0f}]")
    return positions.tolist(), colors.tolist()


# ═══════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════

def segment_and_voxelize(glb_path, output_path=None):
    """Full pipeline: load GLB → classify faces → separate buildings → voxelize → export JSON."""
    name = Path(glb_path).stem
    print("=" * 80)
    print(f"SEGMENT & VOXELIZE: {name}")
    print("=" * 80)

    # ── Step 1: Load GLB ──
    print("\n[1] Loading GLB...")
    t0 = time.time()
    verts, faces, uvs, tex_image = extract_valid_mesh(glb_path)
    print(f"  Vertices: {len(verts):,}  Faces: {len(faces):,}  UVs: {len(uvs):,}  "
          f"Texture: {'yes' if tex_image else 'no'}  ({time.time()-t0:.1f}s)")

    # ── Step 2: Build trimesh object ──
    print("\n[2] Building trimesh mesh...")
    t0 = time.time()
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    print(f"  {len(mesh.faces):,} faces, {len(mesh.vertices):,} vertices ({time.time()-t0:.1f}s)")

    # ── Step 3: Sample surface for color lookup ──
    print("\n[3] Sampling surface for colors...")
    sample_pts, sample_colors = sample_mesh_surface(verts, faces, uvs, tex_image, SURFACE_SAMPLES)

    # ── Step 4: Classify faces ──
    print("\n[4] Classifying faces...")
    t0 = time.time()
    face_labels, centroid_z, abs_nz = classify_faces(mesh)

    # ── Step 5: Separate buildings ──
    print("\n[5] Separating buildings...")
    building_meshes, _ = separate_buildings(mesh, face_labels)
    print(f"  {len(building_meshes)} building(s) identified")

    # ── Step 6: Voxelize each category ──
    print("\n[6] Voxelizing categories...")

    # Buildings
    building_positions, building_colors, building_bboxes = [], [], []
    if building_meshes:
        building_positions, building_colors, building_bboxes = (
            voxelize_buildings(building_meshes, sample_pts, sample_colors))
    else:
        print("  No buildings to voxelize")

    # Terrain
    terrain_faces = np.where(face_labels == LABEL_TERRAIN)[0]
    terrain_positions, terrain_colors = voxelize_terrain(
        mesh, terrain_faces, sample_pts, sample_colors)

    # Water
    water_faces = np.where(face_labels == LABEL_WATER)[0]
    water_positions, water_colors = voxelize_water(
        mesh, water_faces, sample_pts, sample_colors)

    # ── Step 7: Combine & export ──
    print("\n[7] Combining and exporting...")
    all_positions = []
    all_colors = []

    # Add buildings
    for i in range(len(building_positions)):
        all_positions.append(building_positions[i])
        all_colors.append(building_colors[i])

    n_building = len(building_positions)
    n_terrain = len(terrain_positions)
    n_water = len(water_positions)

    all_positions.extend(terrain_positions)
    all_colors.extend(terrain_colors)
    all_positions.extend(water_positions)
    all_colors.extend(water_colors)

    if not all_positions:
        sys.exit("No voxels generated!")

    # Compute overall bbox
    all_pos = np.array(all_positions)
    bbox = {"min": all_pos.min(axis=0).tolist(), "max": all_pos.max(axis=0).tolist()}

    result = {
        "pitch": PITCH_BUILDING,
        "count": len(all_positions),
        "bbox": bbox,
        "positions": all_positions,
        "colors": all_colors,
        "categories": {
            "building": {"count": n_building},
            "terrain": {"count": n_terrain},
            "water": {"count": n_water},
        },
    }

    if output_path is None:
        output_path = OUTPUT_DIR / f"voxel_{name.replace('hkust_', '')}_segmented.json"

    with open(output_path, 'w') as f:
        json.dump(result, f)
    size_kb = os.path.getsize(output_path) / 1024

    print(f"\n  Total voxels: {len(all_positions):,}")
    print(f"    Buildings: {n_building:,}")
    print(f"    Terrain:   {n_terrain:,}")
    print(f"    Water:     {n_water:,}")
    print(f"  Output: {output_path} ({size_kb:.0f} KB)")

    return result


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Segment GLB into buildings/terrain/water + voxelize")
    parser.add_argument("--input", "-i", help="Input GLB file")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--all", action="store_true", help="Process all 4 landmark models")
    args = parser.parse_args()

    if args.all:
        for key, path in MODELS.items():
            if path.exists():
                result = segment_and_voxelize(str(path))
            else:
                print(f"⚠ {path} not found, skipping")
    elif args.input:
        segment_and_voxelize(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

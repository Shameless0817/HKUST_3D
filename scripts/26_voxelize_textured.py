#!/usr/bin/env python3
"""
Convert a textured GLB model (Google Earth photogrammetry) into colored Minecraft-style voxels.

Uses the project's existing GLB parser for robust multi-primitive model loading,
then Trimesh for voxelization and scipy KDTree for color mapping.

Usage:
  python3 scripts/26_voxelize_textured.py --input output/demo/hkust_piazza.glb
  python3 scripts/26_voxelize_textured.py --all
"""
import argparse, json, struct, sys, time, io
from pathlib import Path
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

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "output/demo"
MODELS = {
    "piazza":   OUTPUT_DIR / "hkust_piazza.glb",
    "academic": OUTPUT_DIR / "hkust_academic.glb",
    "seaside":  OUTPUT_DIR / "hkust_seaside.glb",
    "atrium":   OUTPUT_DIR / "hkust_atrium.glb",
}

PITCH = 0.5
N_SURFACE_SAMPLES = 2_000_000
TILE_SIZE = 512


# ═══════════════════════════════════════════════════════════════
#  GLB Parser (same as scripts 23/24/25)
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


def load_texture_images(gltf, buffer_views, bin_data):
    """Extract all embedded texture images from the GLB."""
    images = []
    for img_info in gltf.get('images', []):
        bv_idx = img_info.get('bufferView')
        if bv_idx is not None:
            bv = buffer_views[bv_idx]
            offset = bv.get('byteOffset', 0)
            length = bv['byteLength']
            img_bytes = bin_data[offset:offset+length]
            images.append(Image.open(io.BytesIO(img_bytes)))
        else:
            images.append(None)
    return images


def extract_mesh_from_glb(path):
    """Extract merged mesh with texture info from a GLB, handling multi-primitive models."""
    t0 = time.time()
    gltf, bin_data = parse_glb(path)
    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']
    meshes = gltf['meshes']
    textures = load_texture_images(gltf, buffer_views, bin_data)
    materials = gltf.get('materials', [])

    all_verts = []
    all_faces = []
    all_uvs = []
    vert_offset = 0

    for mesh in meshes:
        for pi, prim in enumerate(mesh['primitives']):
            attrs = prim['attributes']
            v = read_accessor(attrs['POSITION'], accessors, buffer_views, bin_data)
            f = read_accessor(prim['indices'], accessors, buffer_views, bin_data)
            if f.ndim == 1:
                f = f.reshape(-1, 3)
            uv_acc_idx = attrs.get('TEXCOORD_0')
            uv = read_accessor(uv_acc_idx, accessors, buffer_views, bin_data) if uv_acc_idx is not None else np.array([])

            all_verts.append(v)
            all_faces.append(f.astype(np.int64) + vert_offset)
            if len(uv) > 0:
                all_uvs.append(uv)
            elif len(all_verts) > 1 and len(all_uvs) > 0:
                # Primitive without UVs — pad with zeros to keep arrays aligned
                all_uvs.append(np.zeros((len(v), 2), dtype=np.float32))
            vert_offset += len(v)

            if pi == 0 and len(all_verts) == 1:
                print(f"  Primitive 0: {len(v):,}v, {len(f):,}f")
            elif len(all_verts) == 1:
                print(f"  Total primitives: {len(meshes[0]['primitives'])}")

    merged_verts = np.vstack(all_verts)
    merged_faces = np.vstack(all_faces)
    merged_uvs = np.vstack(all_uvs) if all_uvs else np.array([])

    # Sanitize: remove NaN/Inf/extreme vertices using robust statistics
    # Convert to float64 to avoid internal overflow
    v64 = merged_verts.astype(np.float64)
    # First pass: remove clearly invalid values
    clean_mask = (
        ~np.any(np.isnan(v64), axis=1) &
        ~np.any(np.isinf(v64), axis=1) &
        np.all(np.abs(v64) < 1e4, axis=1)  # remove obviously extreme
    )
    clean_verts = v64[clean_mask]
    if (~clean_mask).sum() > 0:
        print(f"  Removed {(~clean_mask).sum():,} obviously-bad vertices")
    # Second pass: statistical outlier filter (IQR-based per axis)
    q1 = np.percentile(clean_verts, 25, axis=0)
    q3 = np.percentile(clean_verts, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - 5 * iqr
    upper = q3 + 5 * iqr
    inlier = np.all((clean_verts >= lower) & (clean_verts <= upper), axis=1)
    bad_outliers = np.where(clean_mask)[0][~inlier] if len(clean_mask) == len(v64) else None
    # Build combined mask
    bad_mask = ~clean_mask
    bad_mask[np.where(clean_mask)[0][~inlier]] = True
    if bad_mask.any():
        print(f"  Removing {bad_mask.sum():,} bad vertices (NaN/Inf/extreme) (from {len(merged_verts):,})")
        keep_idx = np.where(~bad_mask)[0]
        old2new = {int(o): n for n, o in enumerate(keep_idx)}
        merged_verts = merged_verts[keep_idx]
        if len(merged_uvs) > 0:
            merged_uvs = merged_uvs[keep_idx]
        # Keep only faces where ALL three vertices are in keep_idx
        keep_set = set(keep_idx.tolist())
        valid_face = np.array([
            (int(merged_faces[i,0]) in keep_set and
             int(merged_faces[i,1]) in keep_set and
             int(merged_faces[i,2]) in keep_set)
            for i in range(len(merged_faces))
        ], dtype=bool)
        merged_faces = merged_faces[valid_face]
        remapped = np.array([
            [old2new[int(merged_faces[i,0])], old2new[int(merged_faces[i,1])], old2new[int(merged_faces[i,2])]]
            for i in range(len(merged_faces))
        ], dtype=np.int64)
        merged_faces = remapped
        print(f"  After sanitize: {len(merged_verts):,}v, {len(merged_faces):,}f")

    # Use the first texture image (all primitives share the same atlas in script 23 output)
    tex_image = textures[0] if textures else None

    print(f"  Merged: {len(merged_verts):,}v, {len(merged_faces):,}f  "
          f"({time.time()-t0:.1f}s)")
    return merged_verts, merged_faces, merged_uvs, tex_image


# ═══════════════════════════════════════════════════════════════
#  Color Sampling
# ═══════════════════════════════════════════════════════════════

def sample_texture_at_uv(tex_image, uvs, num_tiles, grid_cols, grid_rows, tile_to_pos):
    """Sample the texture atlas at given UV coordinates, handling tile mapping."""
    if tex_image is None:
        return np.full((len(uvs), 3), 128, dtype=np.uint8)

    tex_w, tex_h = tex_image.size
    colors = np.zeros((len(uvs), 3), dtype=np.uint8)
    v_min = float(uvs[:, 1].min())
    v_max = float(uvs[:, 1].max())
    v_span = v_max - v_min if (v_max - v_min) > 0.01 else 0.5

    for i in range(len(uvs)):
        u, v = uvs[i, 0], uvs[i, 1]
        tile_idx = int(np.clip(np.floor(u * num_tiles), 0, num_tiles - 1))

        if tile_idx in tile_to_pos:
            gcol, grow = tile_to_pos[tile_idx]
            frac_u = (u * num_tiles) - tile_idx
            frac_v = (v - v_min) / v_span
            frac_u = np.clip(frac_u, 0.001, 0.999)
            frac_v = np.clip(frac_v, 0.001, 0.999)

            px = int(gcol * TILE_SIZE + frac_u * TILE_SIZE)
            py = int(TILE_SIZE // 2 + grow * (TILE_SIZE // 2) + frac_v * (TILE_SIZE // 2))
            # Actually: the atlas in script 23 is repacked with grid_cols * TILE_SIZE width
            # and grid_rows * HALF_TILE height. But here we have the raw original texture.
            # For the raw texture: tiles are horizontal strip, each 512x256
            src_x = int(tile_idx * TILE_SIZE + frac_u * TILE_SIZE)
            src_y = int(TILE_SIZE // 2 + frac_v * (TILE_SIZE // 2))
            # Actually, Google Earth atlas is: width = num_tiles * 512, height = 512
            # Each tile is 512x256, stacked vertically as top half and bottom half
            # But for simplicity, let's just sample from the raw texture
            atlas_cols = tex_w // TILE_SIZE
            tx = tile_idx % atlas_cols
            ty = tile_idx // atlas_cols
            src_x = int(tx * TILE_SIZE + frac_u * TILE_SIZE)
            src_y = int(ty * TILE_SIZE + frac_v * TILE_SIZE)

            if 0 <= src_x < tex_w and 0 <= src_y < tex_h:
                pixel = tex_image.getpixel((src_x, src_y))
                if isinstance(pixel, int):
                    colors[i] = [pixel, pixel, pixel]
                elif len(pixel) >= 3:
                    colors[i] = list(pixel[:3])
                elif len(pixel) == 1:
                    colors[i] = [pixel[0], pixel[0], pixel[0]]
        else:
            colors[i] = [128, 128, 128]

    return colors


def build_vertex_colors(verts, faces, uvs, tex_image):
    """Assign a color to each vertex by sampling the texture at its UV coordinates."""
    t0 = time.time()

    if tex_image is None or len(uvs) == 0:
        print("  No texture/UVs — using gray vertex colors")
        return np.full((len(verts), 3), 128, dtype=np.uint8)

    tex_w, tex_h = tex_image.size
    num_tiles = tex_w // TILE_SIZE  # Google Earth: horizontal strip of 512x512 tiles

    # Map each UV to a pixel in the texture
    vcolors = np.zeros((len(verts), 3), dtype=np.uint8)
    for i in range(len(verts)):
        u, v = uvs[i, 0], uvs[i, 1]
        # UV u maps to which tile
        tile_idx = int(np.clip(np.floor(u * num_tiles), 0, num_tiles - 1))
        frac_u = u * num_tiles - tile_idx

        # UV v maps to vertical position within tile (each tile is 512x256 of a 512x512 square)
        frac_v = v * 2.0  # Google Earth uses top half of each 512x512 tile

        px = int(tile_idx * TILE_SIZE + frac_u * TILE_SIZE)
        py = int(frac_v * TILE_SIZE)
        px = np.clip(px, 0, tex_w - 1)
        py = np.clip(py, 0, tex_h - 1)

        pixel = tex_image.getpixel((px, py))
        if isinstance(pixel, int):
            vcolors[i] = [pixel, pixel, pixel]
        elif len(pixel) >= 3:
            vcolors[i] = list(pixel[:3])
        elif len(pixel) == 1:
            vcolors[i] = [pixel[0], pixel[0], pixel[0]]

    print(f"  Vertex colors from texture: {len(vcolors):,}  ({time.time()-t0:.1f}s)")
    return vcolors


# ═══════════════════════════════════════════════════════════════
#  Voxelization
# ═══════════════════════════════════════════════════════════════

def point_in_triangle_2d(pt, a, b, c, dominant_axis):
    """Check if point is inside triangle when projected onto the plane
    perpendicular to dominant_axis. Uses barycentric coordinates in 2D."""
    if dominant_axis == 0:  # project to YZ
        ax, ay = a[1], a[2]; bx, by = b[1], b[2]; cx, cy = c[1], c[2]
        px, py = pt[1], pt[2]
    elif dominant_axis == 1:  # project to XZ
        ax, ay = a[0], a[2]; bx, by = b[0], b[2]; cx, cy = c[0], c[2]
        px, py = pt[0], pt[2]
    else:  # project to XY
        ax, ay = a[0], a[1]; bx, by = b[0], b[1]; cx, cy = c[0], c[1]
        px, py = pt[0], pt[1]

    # Barycentric coordinates
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-12:
        return False
    alpha = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    beta  = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    gamma = 1 - alpha - beta
    return (alpha >= -1e-9) and (beta >= -1e-9) and (gamma >= -1e-9)


def voxelize_numpy(verts, faces, pitch):
    """Rasterize every triangle onto the voxel grid.

    For each face, finds the grid cells within its bounding box and tests
    whether the cell center projects inside the triangle. This produces a
    solid surface with no gaps — essential for recognizable Minecraft style.
    """
    t0 = time.time()
    vmin = verts.min(axis=0).astype(np.float64)
    vmax = verts.max(axis=0).astype(np.float64)
    span = vmax - vmin
    if np.any(span < 0) or np.any(~np.isfinite(span)):
        print(f"  ERROR in vertex bounds, filtering...")
        finite = np.all(np.isfinite(verts), axis=1) & np.all(np.abs(verts) < 1e6, axis=1)
        verts = verts[finite]
        vmin = verts.min(axis=0).astype(np.float64)
        vmax = verts.max(axis=0).astype(np.float64)
    dims = np.ceil(span / pitch).astype(int) + 2
    print(f"  Bounds: [{vmin[0]:.0f},{vmax[0]:.0f}] [{vmin[1]:.0f},{vmax[1]:.0f}] [{vmin[2]:.0f},{vmax[2]:.0f}]")
    print(f"  Grid dims: {dims[0]}×{dims[1]}×{dims[2]} ({dims[0]*dims[1]*dims[2]/1e6:.1f}M cells)")
    grid = np.zeros(dims, dtype=bool)
    offset = vmin - pitch * 0.5

    tri_verts = verts[faces]  # (n_faces, 3, 3)

    # Report progress periodically
    total_faces = len(faces)
    report_every = max(1, total_faces // 10)

    for fi in range(total_faces):
        v0, v1, v2 = tri_verts[fi]

        # Determine dominant axis from face normal
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        dominant = np.argmax(np.abs(normal))

        # Triangle bounding box in world space
        tmin = np.minimum(np.minimum(v0, v1), v2)
        tmax = np.maximum(np.maximum(v0, v1), v2)

        # Convert to grid indices with 1-cell padding for edge cases
        imin = np.floor((tmin - offset) / pitch).astype(int)
        imax = np.ceil((tmax - offset) / pitch).astype(int) + 1
        imin = np.clip(imin, 0, dims - 1)
        imax = np.clip(imax, 0, dims - 1)

        # Quick face area to determine sampling density
        area = np.linalg.norm(normal) / 2.0
        # For large faces, need more thorough coverage
        cell_area = pitch * pitch
        # If face area is small relative to cell, just mark centroid cell
        if area < cell_area * 0.1:
            centroid = (v0 + v1 + v2) / 3
            ci = tuple(np.floor((centroid - offset) / pitch).astype(int))
            if all(0 <= ci[j] < dims[j] for j in range(3)):
                grid[ci] = True
            continue

        # For each cell in the bbox, test if center projects inside triangle
        for ix in range(imin[0], imax[0] + 1):
            for iy in range(imin[1], imax[1] + 1):
                for iz in range(imin[2], imax[2] + 1):
                    cell_center = offset + (np.array([ix, iy, iz]) + 0.5) * pitch
                    if point_in_triangle_2d(cell_center, v0, v1, v2, dominant):
                        grid[ix, iy, iz] = True

        if (fi + 1) % report_every == 0:
            print(f"  ... {fi+1}/{total_faces} faces rasterized")

    n_occ = int(np.sum(grid))
    print(f"  Voxel grid: {dims[0]}×{dims[1]}×{dims[2]} = {n_occ:,} occupied  "
          f"({time.time()-t0:.1f}s)")
    return grid, offset, dims


def extract_surface_voxels(grid):
    """Keep only voxels that have at least one empty neighbor (surface-only)."""
    t0 = time.time()
    surface = np.zeros(grid.shape, dtype=bool)
    for axis in range(3):
        # Shift forward
        surface |= grid & ~np.roll(grid, shift=1, axis=axis)
        # Shift backward
        surface |= grid & ~np.roll(grid, shift=-1, axis=axis)
    n_surf = int(np.sum(surface))
    if n_surf == 0:
        print("  ⚠ No surface voxels found, using all occupied")
        surface = grid
        n_surf = int(np.sum(surface))
    print(f"  Surface: {n_surf:,} / {int(np.sum(grid)):,} voxels ({100*n_surf/max(1,int(np.sum(grid))):.0f}%)  "
          f"({time.time()-t0:.1f}s)")
    return surface


def sample_mesh_surface(verts, faces, uvs, tex_image, n_samples):
    """Densely sample the mesh surface with colors interpolated from UV texture.

    Returns (sample_points, sample_colors) arrays for building a KDTree.
    Much more accurate than vertex-only sampling because interpolation captures
    texture detail that sparse vertices miss.
    """
    t0 = time.time()
    tri_verts = verts[faces]  # (n_faces, 3, 3)

    # Cap samples to reasonable number for number of faces
    n_samples = min(n_samples, len(faces) * 50)

    # Compute face areas for proportional sampling
    v0 = tri_verts[:, 1] - tri_verts[:, 0]
    v1 = tri_verts[:, 2] - tri_verts[:, 0]
    cross = np.cross(v0, v1)
    areas = np.linalg.norm(cross, axis=1) / 2.0
    areas = np.maximum(areas, 1e-12)
    probs = areas / areas.sum()

    # Choose faces to sample (do in chunks to avoid memory issues)
    CHUNK = 250_000
    all_sample_points = []
    all_sample_colors = []

    # Pre-convert texture to numpy array for fast sampling
    tex_array = None
    tex_w = tex_h = 0
    if tex_image is not None:
        tex_w, tex_h = tex_image.size
        tex_array = np.array(tex_image)  # (H, W, C) or (H, W)
        if tex_array.ndim == 2:
            tex_array = np.stack([tex_array]*3, axis=-1)

    for chunk_start in range(0, n_samples, CHUNK):
        chunk_size = min(CHUNK, n_samples - chunk_start)

        # Barycentric coordinates for uniform triangle sampling
        r1 = np.random.random(chunk_size)
        r2 = np.random.random(chunk_size)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        w = 1 - r1 - r2

        # Choose faces
        fi = np.random.choice(len(faces), size=chunk_size, p=probs)

        # Compute 3D positions
        fv = tri_verts[fi]  # (chunk, 3, 3)
        pts = (fv[:, 0] * w[:, None] +
               fv[:, 1] * r1[:, None] +
               fv[:, 2] * r2[:, None])
        all_sample_points.append(pts.astype(np.float32))

        # Compute colors from texture
        colors = np.full((chunk_size, 3), 128, dtype=np.uint8)
        if tex_array is not None and len(uvs) > 0:
            face_uvs = uvs[faces[fi]]  # (chunk, 3, 2)
            interp_uvs = (face_uvs[:, 0] * w[:, None] +
                          face_uvs[:, 1] * r1[:, None] +
                          face_uvs[:, 2] * r2[:, None])
            num_tiles = tex_w // TILE_SIZE

            # Fast numpy-based texture sampling
            for i in range(chunk_size):
                u, v = interp_uvs[i, 0], interp_uvs[i, 1]
                tile_idx = int(np.clip(np.floor(u * num_tiles), 0, num_tiles - 1))
                frac_u = u * num_tiles - tile_idx
                frac_v = np.clip(v * 2.0, 0.0, 0.999)
                px = int(tile_idx * TILE_SIZE + frac_u * TILE_SIZE)
                py = int(frac_v * TILE_SIZE)
                px = np.clip(px, 0, tex_w - 1)
                py = np.clip(py, 0, tex_h - 1)
                colors[i] = tex_array[py, px, :3]
        all_sample_colors.append(colors)

    sample_points = np.vstack(all_sample_points)
    sample_colors = np.vstack(all_sample_colors)

    print(f"  Surface samples: {n_samples:,} pts  ({time.time()-t0:.1f}s)")
    return sample_points.astype(np.float32), sample_colors


def assign_voxel_colors(surface_grid, offset, pitch, sample_pts, sample_colors, bbox_voxel):
    """For each surface voxel center, find nearest sample point color via KDTree.

    Uses pre-sampled dense surface points (from sample_mesh_surface) rather than
    raw vertices, giving much better color accuracy for large triangles.
    """
    t0 = time.time()

    # Get surface voxel centers
    indices = np.argwhere(surface_grid)  # (n_surf, 3)
    centers = offset + (indices + 0.5) * pitch  # (n_surf, 3)

    # Build KDTree on dense surface sample points
    tree = KDTree(sample_pts)
    _, nn_idx = tree.query(centers, k=1)

    voxel_colors = sample_colors[nn_idx].astype(np.uint8)

    # Add subtle random jitter so adjacent blocks are distinguishable (±4 per channel)
    np.random.seed(42)
    jitter = np.random.randint(-4, 5, size=voxel_colors.shape, dtype=np.int16)
    voxel_colors = np.clip(voxel_colors.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

    positions = centers.tolist()
    colors = voxel_colors.tolist()

    # Compact output: round positions to 1 decimal
    positions = [[round(float(x), 1), round(float(y), 1), round(float(z), 1)]
                 for x, y, z in positions]

    print(f"  Color assignment: {len(positions):,} voxels  ({time.time()-t0:.1f}s)")
    return positions, colors


# ═══════════════════════════════════════════════════════════════
#  Pipeline
# ═══════════════════════════════════════════════════════════════

def voxelize_textured(path, pitch=PITCH):
    print(f"\n{'='*60}\n  {path.name} ({path.stat().st_size/1e6:.0f} MB)\n{'='*60}")

    # 1. Extract mesh from GLB
    verts, faces, uvs, tex_image = extract_mesh_from_glb(path)

    # 2. Densely sample mesh surface for color reference
    sample_pts, sample_colors = sample_mesh_surface(
        verts, faces, uvs, tex_image, N_SURFACE_SAMPLES)

    # 3. Voxelize
    grid, offset, dims = voxelize_numpy(verts, faces, pitch)

    # 4. Surface only
    surface = extract_surface_voxels(grid)

    # 5. Compute actual voxel bbox (not raw vertex bbox which includes outliers)
    voxel_indices = np.argwhere(surface)
    voxel_mins = offset + voxel_indices.min(axis=0) * pitch
    voxel_maxs = offset + (voxel_indices.max(axis=0) + 1) * pitch
    bbox_voxel = {"min": voxel_mins, "max": voxel_maxs}

    # 6. Assign colors from surface samples
    positions, colors = assign_voxel_colors(
        surface, offset, pitch, sample_pts, sample_colors, bbox_voxel)

    bbox = {
        "min": [round(float(x), 1) for x in voxel_mins],
        "max": [round(float(x), 1) for x in voxel_maxs],
    }

    result = {
        "pitch": pitch,
        "count": len(positions),
        "bbox": bbox,
        "positions": positions,
        "colors": colors,
    }

    json_str = json.dumps(result, separators=(",", ":"))
    kb = len(json_str) / 1024
    print(f"  → {len(positions):,} voxels, {kb:.0f} KB JSON")
    return result


def main():
    parser = argparse.ArgumentParser(description="Voxelize textured GLB → colored voxel JSON")
    parser.add_argument("--input", type=Path, help="Input GLB file")
    parser.add_argument("--output", type=Path, help="Output JSON file (auto-generated if omitted)")
    parser.add_argument("--pitch", type=float, default=PITCH,
                        help=f"Voxel size in model units (default: {PITCH})")
    parser.add_argument("--all", action="store_true", help="Process all 4 campus models")
    args = parser.parse_args()

    targets = []
    if args.all:
        targets = [(name, path) for name, path in MODELS.items() if path.exists()]
        if not targets:
            sys.exit("No campus models found in output/demo/")
    elif args.input:
        name = args.input.stem.replace("hkust_", "")
        targets = [(name, args.input)]
    else:
        sys.exit("Specify --input or --all")

    for name, path in targets:
        t0 = time.time()
        data = voxelize_textured(path, args.pitch)
        out_path = args.output or (OUTPUT_DIR / f"voxel_{name}.json")
        out_path.write_text(json.dumps(data, separators=(",", ":")))
        print(f"  ✓ {out_path.name}  ({out_path.stat().st_size/1024:.0f} KB)  "
              f"[{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()

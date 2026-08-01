#!/usr/bin/env python3
"""
Extract 2D building/landmark footprints from photogrammetry GLB models.

Projects ground-level vertices to XY plane, rasterizes, and finds contours.
Specifically tuned to detect the seaside track ellipse, red bird plaza circle,
and academic building arc.

Usage:
  python3 scripts/27_extract_footprints.py --input output/demo/hkust_seaside.glb --landmark track
  python3 scripts/27_extract_footprints.py --input output/demo/hkust_piazza.glb --landmark plaza
"""
import argparse, json, struct, sys, time
from pathlib import Path
import numpy as np
import cv2

PROJECT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT / "output/demo"

# ═══════════════════════════════════════════════════════════════
#  GLB Parser (from 26_voxelize_textured.py)
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


def extract_all_vertices(glb_path):
    """Extract all vertex positions from a GLB, filtering NaN/Inf."""
    gltf, bin_data = parse_glb(glb_path)
    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']
    meshes = gltf['meshes']

    all_verts = []
    for mesh in meshes:
        for prim in mesh['primitives']:
            attrs = prim['attributes']
            v = read_accessor(attrs['POSITION'], accessors, buffer_views, bin_data)
            all_verts.append(v)

    verts = np.vstack(all_verts).astype(np.float64)

    # Filter NaN/Inf/extreme
    mask = (
        ~np.any(np.isnan(verts), axis=1) &
        ~np.any(np.isinf(verts), axis=1) &
        np.all(np.abs(verts) < 1e4, axis=1)
    )
    verts = verts[mask]

    # IQR filter per axis
    q1 = np.percentile(verts, 25, axis=0)
    q3 = np.percentile(verts, 75, axis=0)
    iqr = q3 - q1
    lower = q1 - 5 * iqr
    upper = q3 + 5 * iqr
    inlier = np.all((verts >= lower) & (verts <= upper), axis=1)
    verts = verts[inlier]

    print(f"  Extracted {len(verts):,} clean vertices")
    print(f"  Bounds: X[{verts[:,0].min():.1f},{verts[:,0].max():.1f}] "
          f"Y[{verts[:,1].min():.1f},{verts[:,1].max():.1f}] "
          f"Z[{verts[:,2].min():.1f},{verts[:,2].max():.1f}]")
    return verts


# ═══════════════════════════════════════════════════════════════
#  2D Footprint Extraction
# ═══════════════════════════════════════════════════════════════

def rasterize_ground(verts, resolution=0.5, z_percentile=15):
    """Rasterize ground-level vertices to a 2D binary image.

    Ground = vertices with Z below z_percentile threshold.
    Returns (binary_grid, origin_x, origin_y, resolution).
    """
    z_thresh = np.percentile(verts[:, 2], z_percentile)
    ground = verts[verts[:, 2] <= z_thresh]
    print(f"  Ground threshold: Z <= {z_thresh:.1f} ({len(ground):,} vertices)")

    x_min, y_min = ground[:, 0].min(), ground[:, 1].min()
    x_max, y_max = ground[:, 0].max(), ground[:, 1].max()

    width = int(np.ceil((x_max - x_min) / resolution)) + 3
    height = int(np.ceil((y_max - y_min) / resolution)) + 3
    print(f"  Grid: {width}×{height} @ {resolution}m/px "
          f"({(x_max-x_min):.0f}m × {(y_max-y_min):.0f}m)")

    grid = np.zeros((height, width), dtype=np.uint8)
    col = np.floor((ground[:, 0] - x_min) / resolution + 0.5).astype(int) + 1
    row = np.floor((ground[:, 1] - y_min) / resolution + 0.5).astype(int) + 1
    valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
    grid[row[valid], col[valid]] = 255

    print(f"  Rasterized {valid.sum():,} points onto grid")

    return grid, x_min - resolution, y_min - resolution, resolution


def find_all_contours(grid, min_area=10):
    """Find contours in binary grid, sorted by area (largest first)."""
    # Morphological close to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Also try dilate to connect nearby points
    dilated = cv2.dilate(closed, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    valid = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area]
    valid.sort(key=lambda x: -x[1])  # largest first

    print(f"  Found {len(valid)} contours (area >= {min_area})")
    for i, (c, area) in enumerate(valid[:10]):
        x, y, w, h = cv2.boundingRect(c)
        print(f"    #{i}: area={area:.0f} bbox=({x},{y} {w}×{h})")
    return [c for c, _ in valid], dilated


def detect_ellipse_from_contours(contours, binary_img):
    """Try to fit an ellipse to the largest contour and return params."""
    if not contours:
        return None

    # Try the largest contour first
    for contour in contours[:5]:
        if len(contour) < 5:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (major, minor), angle = ellipse
            area = np.pi * major * minor / 4
            hull_area = cv2.contourArea(contour)

            # Check if contour is roughly elliptical (area ratio)
            if hull_area > 0:
                ratio = area / hull_area
                if 0.3 < ratio < 3.0:  # reasonable ellipse fit
                    print(f"  ✓ Ellipse fit: center=({cx:.0f},{cy:.0f}) "
                          f"axes=({major:.0f},{minor:.0f}) angle={angle:.0f}° "
                          f"ratio={ratio:.2f}")
                    return {
                        'center': (float(cx), float(cy)),
                        'axes': (float(major), float(minor)),
                        'angle': float(angle),
                        'area': float(area),
                    }
        except cv2.error:
            continue

    print("  ⚠ No good ellipse fit found")
    return None


def detect_plaza_circle(contours, binary_img):
    """Try to find a circular feature (the red bird plaza).

    Returns circle params if found.
    """
    if not contours:
        return None

    for contour in contours[:3]:
        if len(contour) < 5:
            continue
        try:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            area_circle = np.pi * radius * radius
            area_contour = cv2.contourArea(contour)
            if area_contour > 0:
                fill_ratio = area_contour / area_circle
                if fill_ratio > 0.4:  # reasonably circular
                    print(f"  ✓ Circle: center=({cx:.0f},{cy:.0f}) r={radius:.0f} "
                          f"fill={fill_ratio:.2f}")
                    return {
                        'center': (float(cx), float(cy)),
                        'radius': float(radius),
                        'area': float(area_circle),
                    }
        except cv2.error:
            continue

    print("  ⚠ No circle found")
    return None


def grid_to_world(px, py, origin_x, origin_y, resolution):
    """Convert grid (pixel) coordinates to world (meter) coordinates."""
    wx = origin_x + px * resolution
    wy = origin_y + py * resolution
    return wx, wy


def world_to_grid(wx, wy, origin_x, origin_y, resolution):
    """Convert world (meter) coordinates to grid (pixel) coordinates."""
    px = (wx - origin_x) / resolution
    py = (wy - origin_y) / resolution
    return px, py


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def extract_footprint(glb_path, landmark_type, resolution=0.5):
    """Main entry: extract footprint from GLB and identify landmark features."""
    print(f"\n{'='*60}")
    print(f"  Extract footprint: {glb_path.name} ({landmark_type})")
    print(f"{'='*60}")

    verts = extract_all_vertices(glb_path)

    # Rasterize ground layer
    grid, origin_x, origin_y, res = rasterize_ground(verts, resolution=resolution)

    # Find contours
    contours, binary = find_all_contours(grid, min_area=20)

    result = {
        'file': str(glb_path.name),
        'landmark': landmark_type,
        'origin': (origin_x, origin_y),
        'resolution': resolution,
        'grid_shape': grid.shape,
        'num_contours': len(contours),
    }

    if landmark_type == 'track':
        ellipse = detect_ellipse_from_contours(contours, binary)
        if ellipse:
            # Convert to world coordinates
            wx, wy = grid_to_world(ellipse['center'][0], ellipse['center'][1],
                                   origin_x, origin_y, resolution)
            result['ellipse'] = {
                'center_world': (wx, wy),
                'axes_world': (ellipse['axes'][0] * resolution,
                              ellipse['axes'][1] * resolution),
                'angle_deg': ellipse['angle'],
            }
            print(f"  → Track ellipse (world): center=({wx:.1f},{wy:.1f}) "
                  f"axes=({ellipse['axes'][0]*resolution:.1f},"
                  f"{ellipse['axes'][1]*resolution:.1f})m")

            # The "track" is the RING, not the filled ellipse.
            # Standard 400m track: ring width ~10m
            # We'll use this ellipse as the centerline

    elif landmark_type == 'plaza':
        circle = detect_plaza_circle(contours, binary)
        if circle:
            wx, wy = grid_to_world(circle['center'][0], circle['center'][1],
                                   origin_x, origin_y, resolution)
            result['circle'] = {
                'center_world': (wx, wy),
                'radius_world': circle['radius'] * resolution,
            }
            print(f"  → Plaza circle (world): center=({wx:.1f},{wy:.1f}) "
                  f"r={circle['radius']*resolution:.1f}m")

    # Save contour visualization for debugging
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(contours[:5]):
        color = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255)][i % 5]
        cv2.drawContours(vis, [c], -1, color, 2)

    if 'ellipse' in result:
        e = result['ellipse']
        px, py = world_to_grid(e['center_world'][0], e['center_world'][1],
                               origin_x, origin_y, resolution)
        ctr = (int(px), int(py))
        axes = (int(e['axes_world'][0] / resolution / 2),
                int(e['axes_world'][1] / resolution / 2))
        cv2.ellipse(vis, ctr, axes, e['angle_deg'], 0, 360, (0, 255, 255), 2)

    if 'circle' in result:
        c = result['circle']
        px, py = world_to_grid(c['center_world'][0], c['center_world'][1],
                               origin_x, origin_y, resolution)
        cv2.circle(vis, (int(px), int(py)), int(c['radius_world'] / resolution), (0, 255, 255), 2)

    vis_path = OUTPUT_DIR / f"footprint_{landmark_type}.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"  → Visualization saved: {vis_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract 2D footprints from GLB models")
    parser.add_argument("--input", type=Path, required=True, help="Input GLB file")
    parser.add_argument("--landmark", choices=['track', 'plaza', 'academic', 'atrium'],
                        required=True, help="Landmark type to detect")
    parser.add_argument("--resolution", type=float, default=0.5,
                        help="Grid resolution in meters (default: 0.5)")
    args = parser.parse_args()

    result = extract_footprint(args.input, args.landmark, args.resolution)

    # Save result JSON
    out_path = OUTPUT_DIR / f"footprint_{args.landmark}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"  → Saved: {out_path}")


if __name__ == "__main__":
    main()

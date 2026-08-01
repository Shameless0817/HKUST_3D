#!/usr/bin/env python3
"""
HKUST 红鸟广场 → LEGO / Minecraft Voxel 导出

Usage:
  python scripts/09_voxel_export.py crop --bbox <x1,x2,y1,y2,z1,z2>
  python scripts/09_voxel_export.py minecraft [--resolution 1.0] [--bbox ...]
  python scripts/09_voxel_export.py lego [--scale 1:150] [--bbox ...]
"""

import sys
import math
import argparse
import struct
import gzip
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

PROJECT = Path("/home/zliki/HKUST_3D")
DEFAULT_INPUT = PROJECT / "output/demo/hkust_optimized.glb"
OUTPUT_DIR = PROJECT / "output/voxel"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Default bbox in model coordinates (Red Bird Square + Academic Building)
# Estimated from vertex analysis: center-south of campus model
# ============================================================
DEFAULT_BBOX = (-20, 40, -40, 10, -86, 30)  # x1,x2,y1,y2,z1,z2


# ============================================================
# Section 1: Voxelizer — Mesh → 3D occupancy + color grid
# ============================================================
@dataclass
class VoxelGrid:
    """3D voxel representation of a mesh."""
    grid: np.ndarray       # 3D boolean (occupied or not)
    colors: np.ndarray     # 3D uint8[...4] (RGBA per voxel)
    pitch: float           # voxel size in model units
    offset: np.ndarray     # origin of voxel [0,0,0] in model coords
    model_bounds: np.ndarray  # original mesh bounds (for reference)


class Voxelizer:
    """Convert trimesh to voxel grid with color sampling."""

    def __init__(self, mesh, pitch: float = 1.0):
        """
        Args:
            mesh: trimesh.Trimesh with vertex colors
            pitch: voxel size in model units (1.0 = 1 unit per voxel)
        """
        self.mesh = mesh
        self.pitch = pitch

    def voxelize(self, bbox: Optional[Tuple[float, ...]] = None) -> VoxelGrid:
        """
        Voxelize mesh into 3D grid.

        Args:
            bbox: (x1, x2, y1, y2, z1, z2) crop bounds in model coords.
                  If None, uses entire mesh.

        Returns:
            VoxelGrid with boolean occupancy and sampled vertex colors.
        """
        import trimesh as tm

        # Crop mesh if bbox specified
        if bbox is not None:
            mesh = self._crop_mesh(bbox)
        else:
            mesh = self.mesh

        if mesh is None or len(mesh.vertices) == 0:
            raise ValueError("Empty mesh after cropping")

        bounds = np.array([
            [mesh.vertices[:, 0].min(), mesh.vertices[:, 0].max()],
            [mesh.vertices[:, 1].min(), mesh.vertices[:, 1].max()],
            [mesh.vertices[:, 2].min(), mesh.vertices[:, 2].max()],
        ])

        # Calculate grid dimensions
        dims = np.ceil((bounds[:, 1] - bounds[:, 0]) / self.pitch).astype(int) + 1
        dims = np.maximum(dims, 1)

        print(f"  Voxelizing: {dims[0]}×{dims[1]}×{dims[2]} grid ({dims.prod():,} cells)")
        print(f"  Pitch: {self.pitch} units/voxel")
        print(f"  Bounds: X[{bounds[0,0]:.0f}..{bounds[0,1]:.0f}] Y[{bounds[1,0]:.0f}..{bounds[1,1]:.0f}] Z[{bounds[2,0]:.0f}..{bounds[2,1]:.0f}]")

        # Try trimesh voxelization first
        # Note: our mesh is non-watertight (97+ bodies from Google Earth),
        # so fill() won't work. We use subdivide for better surface coverage.
        vox = None
        for method in ['subdivide', 'ray']:
            try:
                vox = tm.voxel.creation.voxelize(mesh, pitch=self.pitch, method=method)
                grid = vox.matrix.astype(bool)
                if grid.sum() > 0:
                    print(f"  {method} method: {grid.sum():,} occupied voxels")
                    break
            except Exception as e:
                print(f"  {method} method failed: {e}")
                continue

        if vox is None:
            print(f"  Falling back to manual voxelization...")
            grid = self._manual_voxelize(mesh, bounds, dims)

        # Sample colors at occupied voxel positions
        colors = self._sample_colors(mesh, grid, bounds)

        return VoxelGrid(
            grid=grid,
            colors=colors,
            pitch=self.pitch,
            offset=bounds[:, 0].copy(),
            model_bounds=bounds,
        )

    def _crop_mesh(self, bbox):
        """Crop mesh to bounding box."""
        import trimesh as tm

        x1, x2, y1, y2, z1, z2 = bbox
        # Find vertices inside bbox
        mask = (
            (self.mesh.vertices[:, 0] >= x1) & (self.mesh.vertices[:, 0] <= x2) &
            (self.mesh.vertices[:, 1] >= y1) & (self.mesh.vertices[:, 1] <= y2) &
            (self.mesh.vertices[:, 2] >= z1) & (self.mesh.vertices[:, 2] <= z2)
        )
        if mask.sum() == 0:
            print(f"  ⚠ No vertices in bbox! Check bounds.")
            return None

        # Keep faces where at least one vertex is inside
        face_mask = mask[self.mesh.faces].any(axis=1)

        if face_mask.sum() == 0:
            print(f"  ⚠ No faces in bbox!")
            return None

        cropped = self.mesh.submesh([np.where(face_mask)[0]], append=True)
        if isinstance(cropped, list):
            cropped = cropped[0]

        print(f"  Cropped: {face_mask.sum():,} faces, {len(cropped.vertices):,} vertices "
              f"(from {len(self.mesh.vertices):,})")
        return cropped

    def _manual_voxelize(self, mesh, bounds, dims):
        """Manual voxelization using face-voxel intersection."""
        grid = np.zeros(dims, dtype=bool)

        # Compute voxel index for each face centroid
        centroids = mesh.triangles_center
        indices = np.floor((centroids - bounds[:, 0]) / self.pitch).astype(int)

        # Clamp to grid
        for i in range(3):
            indices[:, i] = np.clip(indices[:, i], 0, dims[i] - 1)

        grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

        # Also add voxels along face edges for better fill
        for face in mesh.faces:
            verts = mesh.vertices[face]
            for v1, v2 in [(0, 1), (0, 2), (1, 2)]:
                p1, p2 = verts[v1], verts[v2]
                steps = max(1, int(np.linalg.norm(p2 - p1) / (self.pitch * 0.5)))
                for t in np.linspace(0, 1, steps + 1):
                    pt = p1 + t * (p2 - p1)
                    idx = np.floor((pt - bounds[:, 0]) / self.pitch).astype(int)
                    idx = np.clip(idx, 0, dims - 1)
                    grid[tuple(idx)] = True

        print(f"    Manual voxelization: {grid.sum():,} occupied cells")
        return grid

    def _sample_colors(self, mesh, grid, bounds) -> np.ndarray:
        """Sample mesh vertex colors at occupied voxel positions."""
        if not hasattr(mesh.visual, 'vertex_colors') or mesh.visual.vertex_colors is None:
            return np.full(grid.shape + (4,), [128, 128, 128, 255], dtype=np.uint8)

        colors = np.zeros(grid.shape + (4,), dtype=np.uint8)
        colors[..., 3] = 255  # alpha

        # Get occupied voxel indices
        occ = np.argwhere(grid)
        if len(occ) == 0:
            return colors

        # Compute world positions of occupied voxels
        world_pos = occ * self.pitch + bounds[:, 0] + self.pitch / 2

        # Find nearest vertex color for each voxel
        from scipy.spatial import cKDTree
        tree = cKDTree(mesh.vertices)

        # Process in batches to avoid memory issues
        batch_size = 10000
        src_colors = mesh.visual.vertex_colors[:, :4]

        for start in range(0, len(world_pos), batch_size):
            end = min(start + batch_size, len(world_pos))
            batch = world_pos[start:end]
            _, indices = tree.query(batch, k=1)
            batch_colors = src_colors[indices]
            for j, idx in enumerate(occ[start:end]):
                colors[tuple(idx)] = batch_colors[j]

        return colors


# ============================================================
# Section 2: Minecraft .schematic Exporter
# ============================================================

# Map RGB color → Minecraft block type
# We quantize colors to major categories
def rgb_to_minecraft_block(r, g, b, a=255):
    """Map an RGB color to a Minecraft block ID."""
    if a < 128:
        return "minecraft:air"

    # Compute luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Green detection (vegetation)
    if g > r + 30 and g > b + 30:
        if lum < 80:
            return "minecraft:green_concrete"
        return "minecraft:lime_concrete"

    # Blue detection (water)
    if b > r + 30 and b > g + 30:
        if lum > 150:
            return "minecraft:light_blue_concrete"
        return "minecraft:blue_concrete"

    # Red/brown (roofs, red sculpture!)
    if r > g + 40 and r > b + 40:
        if r > 180:
            return "minecraft:red_concrete"
        return "minecraft:brown_concrete"

    # Gray/white (buildings, plaza)
    gray_range = max(r, g, b) - min(r, g, b)
    if gray_range < 30:
        if lum > 200:
            return "minecraft:white_concrete"
        elif lum > 160:
            return "minecraft:light_gray_concrete"
        elif lum > 120:
            return "minecraft:stone"
        elif lum > 80:
            return "minecraft:gray_concrete"
        else:
            return "minecraft:black_concrete"

    # Warm tones (terracotta, ground)
    if r > b + 20 and g > b + 20:
        if lum > 140:
            return "minecraft:smooth_sandstone"
        return "minecraft:brown_terracotta"

    # Default: stone
    return "minecraft:stone"


class MinecraftExporter:
    """Export VoxelGrid as Minecraft .schematic (Sponge v2 format)."""

    # NBT tag type constants (from nbtlib)
    TAG_END = 0
    TAG_BYTE = 1
    TAG_SHORT = 2
    TAG_INT = 3
    TAG_LONG = 4
    TAG_FLOAT = 5
    TAG_DOUBLE = 6
    TAG_BYTE_ARRAY = 7
    TAG_STRING = 8
    TAG_LIST = 9
    TAG_COMPOUND = 10
    TAG_INT_ARRAY = 11
    TAG_LONG_ARRAY = 12

    def __init__(self, voxel_grid: VoxelGrid, name: str = "hkust_piazza"):
        self.vox = voxel_grid
        self.name = name

    def export(self, path: Path):
        """Write .schem file using Sponge Schematic v2 format."""
        print(f"\n  === Minecraft Export ===")

        # Build block palette
        palette, block_data = self._build_palette()
        print(f"  Palette: {len(palette)} unique block types")
        for block_id in palette:
            print(f"    - {block_id}")

        # Write .schem using nbtlib
        try:
            import nbtlib
            self._write_nbtlib(path, palette, block_data)
        except ImportError:
            print("  nbtlib not installed, writing raw NBT...")
            self._write_raw_nbt(path, palette, block_data)

        size = path.stat().st_size
        print(f"  ✓ Written: {path} ({size/1024:.1f} KB)")

    def _build_palette(self):
        """Build block palette from voxel colors.

        Returns:
            palette: list of block ID strings (indexed by palette index)
            block_data: 3D int array of palette indices
        """
        grid = self.vox.grid
        colors = self.vox.colors

        # Build mapping from block ID → palette index
        palette_dict = {"minecraft:air": 0}
        block_data = np.zeros(grid.shape, dtype=np.int32)

        occ = np.argwhere(grid)
        for idx in occ:
            r, g, b, a = colors[tuple(idx)]
            block_id = rgb_to_minecraft_block(int(r), int(g), int(b), int(a))
            if block_id not in palette_dict:
                palette_dict[block_id] = len(palette_dict)
            block_data[tuple(idx)] = palette_dict[block_id]

        palette = [""] * len(palette_dict)
        for block_id, pal_idx in palette_dict.items():
            palette[pal_idx] = block_id

        return palette, block_data

    def _write_nbtlib(self, path, palette, block_data):
        """Write .schem using nbtlib library."""
        import nbtlib
        from nbtlib.tag import (
            Byte, Short, Int, Long, String, IntArray,
            ByteArray, Compound, List as NbtList
        )

        w, h, d = block_data.shape

        # Sponge Schematic v2 format
        schematic = Compound({
            "Version": Int(2),
            "DataVersion": Int(2865),  # Minecraft 1.20
            "Metadata": Compound({
                "Name": String(self.name),
                "Author": String("Claude Code"),
                "Date": Long(0),
                "RequiredMods": NbtList[Compound](),
            }),
            "Width": Short(w),
            "Height": Short(h),
            "Length": Short(d),
            "Offset": IntArray([-w//2, 0, -d//2]),
            "PaletteMax": Int(len(palette)),
            "Palette": Compound({
                pid: Int(i) for i, pid in enumerate(palette)
            }),
            "BlockData": ByteArray(block_data.ravel().tolist()),
        })

        with open(path, 'wb') as f:
            nbtlib.File(schematic, gzipped=True).save(f)

    def _write_raw_nbt(self, path, palette, block_data):
        """Write .schem using manual NBT encoding (fallback without nbtlib)."""
        w, h, d = block_data.shape

        # Build NBT binary
        data = bytearray()

        # NBT header
        data += b'\x0a'  # TAG_Compound
        data += b'\x00\x00'  # unnamed root

        # Schematic compound content
        content = bytearray()

        # Version (Int)
        content += self._nbt_int(b"Version", 2)
        # DataVersion (Int)
        content += self._nbt_int(b"DataVersion", 2865)

        # Width (Short)
        content += self._nbt_short(b"Width", w)
        # Height (Short)
        content += self._nbt_short(b"Height", h)
        # Length (Short)
        content += self._nbt_short(b"Length", d)

        # Offset (IntArray)
        offset_bytes = struct.pack('>iii', -w//2, 0, -d//2)
        content += self._nbt_int_array(b"Offset", offset_bytes)

        # PaletteMax (Int)
        content += self._nbt_int(b"PaletteMax", len(palette))

        # Palette (Compound of String→Int)
        pal_bytes = bytearray()
        for i, pid in enumerate(palette):
            pal_bytes += self._nbt_int(pid.encode(), i)
        content += self._nbt_compound(b"Palette", pal_bytes)

        # BlockData (ByteArray)
        flat_data = block_data.ravel().astype(np.uint8).tobytes()
        content += self._nbt_byte_array(b"BlockData", flat_data)

        # End tag
        content += b'\x00'

        data += struct.pack('>I', len(content))
        data += content

        with open(path, 'wb') as f:
            f.write(data)

        print(f"  ⚠ Written as raw NBT (install nbtlib for proper .schem)")

    # -- NBT encoding helpers --
    def _nbt_byte(self, name: bytes, value: int) -> bytes:
        return b'\x01' + struct.pack('>H', len(name)) + name + struct.pack('>b', value)

    def _nbt_short(self, name: bytes, value: int) -> bytes:
        return b'\x02' + struct.pack('>H', len(name)) + name + struct.pack('>h', value)

    def _nbt_int(self, name: bytes, value: int) -> bytes:
        return b'\x03' + struct.pack('>H', len(name)) + name + struct.pack('>i', value)

    def _nbt_long(self, name: bytes, value: int) -> bytes:
        return b'\x04' + struct.pack('>H', len(name)) + name + struct.pack('>q', value)

    def _nbt_string(self, name: bytes, value: bytes) -> bytes:
        return b'\x08' + struct.pack('>H', len(name)) + name + struct.pack('>H', len(value)) + value

    def _nbt_compound(self, name: bytes, content: bytes) -> bytes:
        return b'\x0a' + struct.pack('>H', len(name)) + name + content + b'\x00'

    def _nbt_byte_array(self, name: bytes, data: bytes) -> bytes:
        return b'\x07' + struct.pack('>H', len(name)) + name + struct.pack('>I', len(data)) + data

    def _nbt_int_array(self, name: bytes, data: bytes) -> bytes:
        # data should be packed as big-endian ints
        count = len(data) // 4
        return b'\x0b' + struct.pack('>H', len(name)) + name + struct.pack('>I', count) + data


# ============================================================
# Section 3: LEGO .ldr Exporter
# ============================================================

# LDraw color map: approximate RGB → LDraw color code
# LDraw color codes: https://www.ldraw.org/article/547.html
LDRAW_COLORS = {
    # (name, code, approximate RGB)
    "white":      (15, (255, 255, 255)),
    "light_gray": (7,  (170, 170, 170)),
    "dark_gray":  (8,  (100, 100, 100)),
    "black":      (0,  (33,  33,  33)),
    "red":        (4,  (196, 40,  28)),
    "blue":       (1,  (13,  105, 172)),
    "green":      (2,  (40,  127, 71)),
    "dark_green": (288,(24,  87,  48)),
    "yellow":     (14, (245, 205, 48)),
    "brown":      (6,  (100, 58,  42)),
    "tan":        (19, (226, 204, 158)),
    "orange":     (25, (254, 138, 24)),
    "dark_red":   (320,(114, 13,  19)),
    "dark_blue":  (272,(13,  42,  105)),
    "light_blue": (9,  (181, 210, 233)),
    "dark_tan":   (69, (144, 125, 85)),
    "lime":       (27, (187, 233, 11)),
    "pink":       (13, (252, 151, 172)),
}

# Common LEGO brick part files in LDraw library
# Format: (part_filename, studs_x, studs_z, plates_y, description)
LEGO_BRICKS = [
    # Plates (1/3 brick height = 8 LDU)
    ("3024.dat",  1, 1, 1, "Plate 1x1"),
    ("3023.dat",  1, 2, 1, "Plate 1x2"),
    ("3022.dat",  2, 2, 1, "Plate 2x2"),
    ("3020.dat",  2, 4, 1, "Plate 2x4"),
    ("3021.dat",  2, 3, 1, "Plate 2x3"),
    ("3666.dat",  1, 6, 1, "Plate 1x6"),
    ("3460.dat",  1, 8, 1, "Plate 1x8"),
    ("3034.dat",  2, 8, 1, "Plate 2x8"),
    ("3035.dat",  4, 8, 1, "Plate 4x8"),
    # Bricks (1 brick height = 24 LDU = 3 plates)
    ("3005.dat",  1, 1, 3, "Brick 1x1"),
    ("3004.dat",  1, 2, 3, "Brick 1x2"),
    ("3003.dat",  2, 2, 3, "Brick 2x2"),
    ("3001.dat",  2, 4, 3, "Brick 2x4"),
    ("3002.dat",  2, 3, 3, "Brick 2x3"),
    ("3009.dat",  1, 6, 3, "Brick 1x6"),
    ("3008.dat",  1, 8, 3, "Brick 1x8"),
    ("3007.dat",  2, 8, 3, "Brick 2x8"),
    # Tall bricks
    ("3010.dat",  1, 4, 3, "Brick 1x4"),
    ("2456.dat",  2, 6, 3, "Brick 2x6"),
    ("6112.dat",  1, 12, 3, "Brick 1x12"),
    # Specialty
    ("3040.dat",  1, 2, 1, "Slope 45 2x1"),  # slopes for roofs
    ("3665.dat",  1, 2, 3, "Slope 45 2x1 (tall)"),
]

# Sort bricks by volume (largest first) for greedy placement
LEGO_BRICKS.sort(key=lambda b: b[1] * b[2] * b[3], reverse=True)


def rgb_to_ldraw_color(r, g, b):
    """Find closest LDraw color code for an RGB value."""
    best_dist = float('inf')
    best_code = 15  # default white

    for name, (code, (lr, lg, lb)) in LDRAW_COLORS.items():
        dist = (r - lr)**2 + (g - lg)**2 + (b - lb)**2
        if dist < best_dist:
            best_dist = dist
            best_code = code

    return best_code


class LegoExporter:
    """Export VoxelGrid as LEGO .ldr file."""

    def __init__(self, voxel_grid: VoxelGrid, scale_denom: int = 150):
        """
        Args:
            voxel_grid: voxelized mesh
            scale_denom: scale denominator (e.g., 150 means 1:150 scale)
                         Real 1m → model 1 unit → LEGO at 1:150 = 6.67mm/m ≈ 0.83 stud/m
        """
        self.vox = voxel_grid
        self.scale = scale_denom

        # LEGO dimensions in LDU (1 LDU = 0.4mm)
        self.STUD_PITCH = 20   # LDU between studs (8mm)
        self.PLATE_HEIGHT = 8  # LDU (3.2mm)
        self.BRICK_HEIGHT = 24 # LDU (9.6mm)

    def export(self, path: Path):
        """Write .ldr file with greedy brick placement."""
        print(f"\n  === LEGO Export ===")
        print(f"  Scale: 1:{self.scale}")
        print(f"  Grid: {self.vox.grid.shape}")

        # Step 1: Resample voxel grid to LEGO-compatible resolution
        lego_grid, colors_3d = self._resample_to_lego()

        print(f"  LEGO grid: {lego_grid.shape} ({lego_grid.sum():,} voxels)")

        # Step 2: Greedy brick placement
        bricks = self._greedy_place(lego_grid, colors_3d)
        print(f"  Bricks placed: {len(bricks):,}")

        # Step 3: Write LDraw file
        self._write_ldr(path, bricks)
        size = path.stat().st_size
        print(f"  ✓ Written: {path} ({size/1024:.1f} KB)")

    def _resample_to_lego(self):
        """Resample voxel grid to LEGO stud pitch resolution."""
        # Real-world: 1 unit in our model ≈ 1 meter
        # At 1:150 scale: 1m → 6.67mm → 6.67/8 = 0.83 studs per model unit
        studs_per_unit = 1000.0 / (self.scale * 8.0)  # studs per model unit
        plate_per_unit = 1000.0 / (self.scale * 3.2)  # plates per model unit

        # We want each LEGO voxel = 1 stud × 1 stud × 1 plate
        # Target: resample model grid to LEGO pitch
        grid = self.vox.grid
        colors = self.vox.colors

        # Determine LEGO grid size
        # Floor: because partial voxels don't make good LEGO
        lego_x = max(1, int(grid.shape[0] * studs_per_unit))
        lego_y = max(1, int(grid.shape[2] * studs_per_unit))  # Z in model → Y in LDraw
        lego_z = max(1, int(grid.shape[1] * plate_per_unit))  # Y in model → Z (up) in LDraw

        # But limit to reasonable size
        max_dim = 200
        if max(lego_x, lego_y, lego_z) > max_dim:
            factor = max_dim / max(lego_x, lego_y, lego_z)
            lego_x = max(1, int(lego_x * factor))
            lego_y = max(1, int(lego_y * factor))
            lego_z = max(1, int(lego_z * factor))
            actual_studs = studs_per_unit * factor
        else:
            actual_studs = studs_per_unit

        # Resample
        lego_grid = np.zeros((lego_x, lego_z, lego_y), dtype=bool)  # X, Z(up), Y
        lego_colors = np.zeros((lego_x, lego_z, lego_y, 4), dtype=np.uint8)

        for x in range(lego_x):
            for y in range(lego_y):
                for z in range(lego_z):
                    # Map LEGO coords → model coords
                    mx = int(x / actual_studs)
                    my = int(z / plate_per_unit)  # model Y
                    mz = int(y / actual_studs)    # model Z

                    if (0 <= mx < grid.shape[0] and
                        0 <= my < grid.shape[1] and
                        0 <= mz < grid.shape[2] and
                        grid[mx, my, mz]):
                        lego_grid[x, z, y] = True
                        lego_colors[x, z, y] = colors[mx, my, mz]

        print(f"  Resampled: {lego_grid.shape} ({lego_grid.sum():,} occupied)")
        return lego_grid, lego_colors

    def _greedy_place(self, grid, colors):
        """Greedy largest-brick-first placement."""
        bricks = []
        remaining = grid.copy()
        X, Z, Y = remaining.shape  # X=studs wide, Z=plates tall, Y=studs deep

        # For each brick size (largest first), scan for placements
        for part_file, sx, sy, sz, desc in LEGO_BRICKS:
            if not remaining.any():
                break

            for x in range(X - sx + 1):
                for y in range(Y - sy + 1):
                    for z in range(Z - sz + 1):
                        # Check if this region is fully occupied
                        region = remaining[x:x+sx, z:z+sz, y:y+sy]
                        if region.all() and region.size > 0:
                            # Place brick here
                            # Get dominant color
                            region_colors = colors[x:x+sx, z:z+sz, y:y+sy][region]
                            if len(region_colors) > 0:
                                avg_r = region_colors[:, 0].mean()
                                avg_g = region_colors[:, 1].mean()
                                avg_b = region_colors[:, 2].mean()
                            else:
                                avg_r, avg_g, avg_b = 170, 170, 170

                            ldraw_color = rgb_to_ldraw_color(avg_r, avg_g, avg_b)

                            # LDraw position: center of brick in LDU
                            # X: studs → LDU (20 LDU per stud)
                            # Y: LDraw Z (up)
                            # Z: studs → LDU
                            lx = x * self.STUD_PITCH
                            ly = z * self.PLATE_HEIGHT  # Z in our grid = plates up
                            lz = y * self.STUD_PITCH

                            bricks.append({
                                'color': ldraw_color,
                                'x': lx, 'y': ly, 'z': lz,
                                'part': part_file,
                                'desc': desc,
                            })

                            # Mark as placed
                            remaining[x:x+sx, z:z+sz, y:y+sy] = False

        # Fill remaining with 1×1 plates
        if remaining.any():
            occ = np.argwhere(remaining)
            for idx in occ:
                x, z, y = idx
                c = colors[x, z, y]
                ldraw_color = rgb_to_ldraw_color(int(c[0]), int(c[1]), int(c[2]))

                lx = x * self.STUD_PITCH
                ly = z * self.PLATE_HEIGHT
                lz = y * self.STUD_PITCH

                bricks.append({
                    'color': ldraw_color,
                    'x': lx, 'y': ly, 'z': lz,
                    'part': '3024.dat',
                    'desc': 'Plate 1x1 (fill)',
                })

        return bricks

    def _write_ldr(self, path, bricks):
        """Write LDraw .ldr file."""
        lines = []
        # Header
        lines.append("0 HKUST Red Bird Square - LEGO Model")
        lines.append("0 Generated by HKUST_3D voxel export")
        lines.append(f"0 Scale: 1:{self.scale}")
        lines.append(f"0 Total bricks: {len(bricks)}")
        lines.append("0 Name: hkust_piazza.ldr")
        lines.append("0 Author: Claude Code")
        lines.append("")

        # Brick placements
        for b in bricks:
            # Type 1 line:
            # 1 <color> x y z a b c d e f g h i <part.dat>
            # Rotation matrix is identity (a=1,e=1,i=1) for axis-aligned bricks
            lines.append(
                f"1 {b['color']} "
                f"{b['x']} {b['y']} {b['z']} "
                f"1 0 0 0 1 0 0 0 1 "
                f"{b['part']}"
            )

        # Footer
        lines.append("0")

        with open(path, 'w') as f:
            f.write('\n'.join(lines))


# ============================================================
# Section 4: CLI & Main
# ============================================================

def cmd_crop(args):
    """Crop the model and export a GLB for visual inspection."""
    import trimesh

    mesh = trimesh.load(DEFAULT_INPUT, force='mesh')
    print(f"Loaded: {DEFAULT_INPUT}")
    print(f"  Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.faces):,}")

    bbox = parse_bbox(args.bbox)
    vox = Voxelizer(mesh, pitch=args.resolution)
    cropped = vox._crop_mesh(bbox)

    if cropped is None:
        print("Error: Empty crop. Try different bbox.")
        sys.exit(1)

    out_path = OUTPUT_DIR / "hkust_piazza_cropped.glb"
    cropped.export(str(out_path))
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n✓ Cropped model: {out_path} ({size_mb:.1f} MB)")
    print(f"  Open at: https://gltf-viewer.donmccurdy.com/")


def cmd_minecraft(args):
    """Export as Minecraft .schematic."""
    import trimesh

    mesh = trimesh.load(DEFAULT_INPUT, force='mesh')
    print(f"Loaded: {DEFAULT_INPUT}")

    bbox = parse_bbox(args.bbox) if args.bbox else DEFAULT_BBOX
    pitch = args.resolution  # meters per block

    vox = Voxelizer(mesh, pitch=pitch)
    grid = vox.voxelize(bbox=bbox)

    exporter = MinecraftExporter(grid, name="hkust_piazza")
    out_path = OUTPUT_DIR / "hkust_piazza.schem"
    exporter.export(out_path)

    # Print block counts
    grid_size = '×'.join(map(str, grid.grid.shape))
    real_size = f"{grid.grid.shape[0]*pitch:.0f}m × {grid.grid.shape[1]*pitch:.0f}m × {grid.grid.shape[2]*pitch:.0f}m"
    print(f"\n  Grid: {grid_size} ({real_size})")
    print(f"  Occupied: {grid.grid.sum():,} blocks")
    print(f"\n  To use in Minecraft:")
    print(f"    1. Install WorldEdit mod")
    print(f"    2. Place file in: .minecraft/config/worldedit/schematics/")
    print(f"    3. In game: //schem load hkust_piazza")
    print(f"    4. //paste")


def cmd_lego(args):
    """Export as LEGO .ldr file."""
    import trimesh

    mesh = trimesh.load(DEFAULT_INPUT, force='mesh')
    print(f"Loaded: {DEFAULT_INPUT}")

    bbox = parse_bbox(args.bbox) if args.bbox else DEFAULT_BBOX
    scale_denom = parse_scale(args.scale)

    print(f"\nStep 1: Voxelize mesh...")
    vox = Voxelizer(mesh, pitch=1.0)  # 1 unit ≈ 1m
    grid = vox.voxelize(bbox=bbox)

    print(f"\nStep 2: LEGO brick placement at 1:{scale_denom}...")
    exporter = LegoExporter(grid, scale_denom=scale_denom)
    out_path = OUTPUT_DIR / "hkust_piazza.ldr"
    exporter.export(out_path)

    print(f"\n  To open in BrickLink Studio:")
    print(f"    1. Download: https://www.bricklink.com/v3/studio/download.page")
    print(f"    2. File → Import → Import LDraw File")
    print(f"    3. Select: {out_path}")
    print(f"\n  Or use LDView: https://github.com/tcobbs/ldview")


def parse_bbox(bbox_str):
    """Parse bbox string 'x1,x2,y1,y2,z1,z2' or 'x1,x2,y1,y2'."""
    parts = [float(x) for x in bbox_str.split(',')]
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3], -100, 200
    elif len(parts) == 6:
        return tuple(parts)
    else:
        raise ValueError(f"Invalid bbox: {bbox_str}. Use x1,x2,y1,y2 or x1,x2,y1,y2,z1,z2")


def parse_scale(scale_str):
    """Parse scale string '1:150' or '150'."""
    if ':' in scale_str:
        return int(scale_str.split(':')[1])
    return int(scale_str)


def main():
    parser = argparse.ArgumentParser(
        description="HKUST 红鸟广场 → LEGO / Minecraft Voxel 导出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crop and visualize Red Bird Square area
  python scripts/09_voxel_export.py crop --bbox -20,40,-40,10

  # Export Minecraft schematic (1 block = 1 meter)
  python scripts/09_voxel_export.py minecraft --resolution 1.0

  # Export LEGO LDraw file (1:150 scale)
  python scripts/09_voxel_export.py lego --scale 1:150
        """
    )
    sub = parser.add_subparsers(dest='cmd')

    # Crop
    p_crop = sub.add_parser('crop', help='Crop model and export GLB')
    p_crop.add_argument('--bbox', default='-20,40,-40,10,-86,30',
                        help='Bounding box: x1,x2,y1,y2,z1,z2')
    p_crop.add_argument('--resolution', type=float, default=1.0,
                        help='Voxel resolution for info')

    # Minecraft
    p_mc = sub.add_parser('minecraft', help='Export Minecraft .schematic')
    p_mc.add_argument('--bbox', default=None,
                      help='Bounding box (default: Red Bird Square area)')
    p_mc.add_argument('--resolution', type=float, default=1.0,
                      help='Block size in meters (default: 1.0)')

    # LEGO
    p_lego = sub.add_parser('lego', help='Export LEGO .ldr')
    p_lego.add_argument('--bbox', default=None,
                        help='Bounding box (default: Red Bird Square area)')
    p_lego.add_argument('--scale', default='1:150',
                        help='Scale (default: 1:150)')

    args = parser.parse_args()

    if args.cmd == 'crop':
        cmd_crop(args)
    elif args.cmd == 'minecraft':
        cmd_minecraft(args)
    elif args.cmd == 'lego':
        cmd_lego(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

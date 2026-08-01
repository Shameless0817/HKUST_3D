#!/usr/bin/env python3
"""
HKUST Demo 增强 — 添加 SRTM 真实地形

将 Overture 建筑挤出模型放到 SRTM 地形上，
解决"平地一堆色块"的问题。
HKUST 校园高差达 60-100m，加入地形后立即可辨认。

使用方法:
  python 07_add_terrain.py                  # 下载SRTM + 生成带地形的GLB
  python 07_add_terrain.py --synthetic      # 使用合成地形(网络不可用时)
  python 07_add_terrain.py --help

输出:
  output/demo/hkust_with_terrain.glb        — 建筑+地形合并模型
  output/demo/hkust_elevation.npy           — 高程数据缓存
"""

import sys
import math
import json
import argparse
import struct
import io
import gzip
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HKUST 范围
HKUST = {
    "min_lat": 22.330,
    "max_lat": 22.342,
    "min_lon": 114.258,
    "max_lon": 114.272,
}

# 参考点 (经纬度)
REF_LON, REF_LAT = 114.265, 22.337

def latlon_to_meters(lon: float, lat: float, ref_lon: float, ref_lat: float):
    """经纬度 → 米 (相对参考点)"""
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))
    x = (lon - ref_lon) * m_per_deg_lon
    y = (lat - ref_lat) * m_per_deg_lat
    return x, y


# ============================================================
# Step 1: 获取高程数据
# ============================================================


def download_srtm_hgt() -> Optional[np.ndarray]:
    """
    从公共服务器下载 SRTM 高程数据

    尝试多个源，返回 HKUST 区域的高程数组 (米)
    """
    import urllib.request

    tile_lat = 22  # N22
    tile_lon = 114  # E114
    tile_name = f"N{tile_lat}E{tile_lon:03d}"

    # SRTM .hgt 文件的公共镜像
    urls = [
        # AWS Open Elevation tiles
        f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile_lat:02d}/{tile_name}.hgt.gz",
        # Alternate
        f"https://elevation-tiles-prod.s3.amazonaws.com/skadi/{tile_lat:02d}/{tile_name}.hgt.gz",
    ]

    raw_data = None
    for url in urls:
        try:
            print(f"  尝试下载: {url}")
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) HKUST-3D-Demo/1.0'
            })
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                if url.endswith('.gz'):
                    data = gzip.decompress(data)
                raw_data = data
                print(f"  ✓ 下载成功: {len(data)} bytes")
                break
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue

    if raw_data is None:
        return None

    # 解析 .hgt 文件
    # SRTM3 (3 arc-second) = 1201x1201, 每个样本 2 bytes (big-endian int16)
    # SRTM1 (1 arc-second) = 3601x3601
    size = len(raw_data)
    dim = int(math.sqrt(size / 2))
    if dim * dim * 2 != size:
        print(f"  ✗ 无法确定 HGT 尺寸: {size} bytes, dim={dim}")
        return None

    elev_full = np.frombuffer(raw_data, dtype='>i2').reshape(dim, dim)
    print(f"  高程网格: {dim}x{dim}")

    # HGT 文件行序: 第0行 = 最北 (tile_lat + 1°)
    # 我们需要从中提取 HKUST 子区域
    # Tile 覆盖: [tile_lat, tile_lat+1] × [tile_lon, tile_lon+1]
    pix_per_deg = dim - 1  # 每个像素 = 1/(dim-1) 度

    # HKUST 区域在 tile 内的像素范围
    # 行: 北→南，所以 lat_max 对应较小的行号
    row_north = int((tile_lat + 1 - HKUST["max_lat"]) * pix_per_deg)
    row_south = int((tile_lat + 1 - HKUST["min_lat"]) * pix_per_deg)
    col_west = int((HKUST["min_lon"] - tile_lon) * pix_per_deg)
    col_east = int((HKUST["max_lon"] - tile_lon) * pix_per_deg)

    # 确保在范围内
    row_north = max(0, row_north)
    row_south = min(dim, row_south)
    col_west = max(0, col_west)
    col_east = min(dim, col_east)

    sub = elev_full[row_north:row_south, col_west:col_east].astype(np.float32)
    # 无效值处理
    sub[sub <= -32768] = np.nan
    sub[sub > 9000] = np.nan

    # 用最近邻填充 NaN（海面等）
    if np.any(np.isnan(sub)):
        from scipy import ndimage
        # 简单方法: NaN → 0
        sub = np.nan_to_num(sub, nan=0.0)

    print(f"  HKUST 区域: {sub.shape} 网格")
    print(f"  高程范围: {sub.min():.1f}m - {sub.max():.1f}m")

    return sub


def generate_synthetic_terrain() -> np.ndarray:
    """
    生成合成地形 (网络不可用时的后备方案)

    基于 HKUST 已知的地形特征:
      - 东南→西北方向逐渐升高
      - 海岸线 (~0m) → 上校园入口 (~110m)
      - 陡坡，建筑阶梯状分布
    """
    print("  使用合成地形 (基于 HKUST 已知地形特征)")

    # 100x100 网格覆盖 HKUST 区域
    GRID = 100
    elev = np.zeros((GRID, GRID), dtype=np.float32)

    for i in range(GRID):
        for j in range(GRID):
            # lat: north→south (row), lon: west→east (col)
            lon = HKUST["min_lon"] + (HKUST["max_lon"] - HKUST["min_lon"]) * j / (GRID-1)
            lat = HKUST["max_lat"] - (HKUST["max_lat"] - HKUST["min_lat"]) * i / (GRID-1)

            # 基础高程: 从东南(海面, 0m)到西北(~110m)逐步升高
            # 归一化坐标
            nx = (lon - HKUST["min_lon"]) / (HKUST["max_lon"] - HKUST["min_lon"])
            ny = (lat - HKUST["min_lat"]) / (HKUST["max_lat"] - HKUST["min_lat"])

            # 主要坡度: 西北高、东南低
            base = (1 - nx) * 100 + ny * 20

            # 添加地形细节 (港科大的几个平台)
            # 海边平台 (~0-30m)
            if ny > 0.7 and nx < 0.3:
                base = max(base, 5 + 15 * (ny - 0.7) / 0.3 * (0.3 - nx) / 0.3)

            # 中层平台 (~40-70m) - 学术楼区域
            if 0.4 < ny < 0.7 and 0.3 < nx < 0.7:
                base = 45 + np.random.random() * 5

            # 上层平台 (~80-110m) - 宿舍区
            if ny < 0.5 and nx > 0.5:
                base = 80 + np.random.random() * 10

            # 微地形噪声
            base += np.random.random() * 5 - 2.5

            elev[i, j] = max(0, base)

    print(f"  合成网格: {GRID}x{GRID}")
    print(f"  高程范围: {elev.min():.1f}m - {elev.max():.1f}m")
    return elev


def get_elevation_data(use_synthetic: bool = False) -> np.ndarray:
    """
    获取 HKUST 区域高程数据

    Returns:
        2D numpy array, shape (H, W), 值 = 高程(米)
        行序: 北→南 (row 0 = max_lat)
        列序: 西→东 (col 0 = min_lon)
    """
    cache_path = OUTPUT_DIR / "hkust_elevation.npy"

    # 使用缓存
    if cache_path.exists():
        print(f"加载缓存高程数据: {cache_path}")
        elev = np.load(cache_path)
        print(f"  网格: {elev.shape}, 范围: {elev.min():.1f}-{elev.max():.1f}m")
        return elev

    elev = None

    if not use_synthetic:
        print("下载 SRTM 高程数据...")
        elev = download_srtm_hgt()

    if elev is None:
        print("SRTM 下载失败，生成合成地形...")
        elev = generate_synthetic_terrain()

    # 缓存
    np.save(cache_path, elev)
    print(f"✓ 高程数据已缓存: {cache_path}")
    return elev


# ============================================================
# Step 2: 生成地形 Mesh
# ============================================================


def build_terrain_mesh(elevation: np.ndarray,
                       bbox: dict = None,
                       ref_lon: float = REF_LON,
                       ref_lat: float = REF_LAT) -> "trimesh.Trimesh":
    """
    从高程数组生成地形三角网

    Args:
        elevation: 2D 高程数组 [rows(H→S), cols(W→E)]
        bbox: 经纬度范围
        ref_lon, ref_lat: 坐标参考点

    Returns:
        trimesh.Trimesh 地形 mesh
    """
    import trimesh

    if bbox is None:
        bbox = HKUST

    H, W = elevation.shape
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))

    # 每个像素的米尺寸
    pix_size_lon = (bbox["max_lon"] - bbox["min_lon"]) / (W - 1) * m_per_deg_lon
    pix_size_lat = (bbox["max_lat"] - bbox["min_lat"]) / (H - 1) * m_per_deg_lat

    # 生成顶点
    vertices = []
    for i in range(H):
        lat = bbox["max_lat"] - (bbox["max_lat"] - bbox["min_lat"]) * i / (H - 1)
        for j in range(W):
            lon = bbox["min_lon"] + (bbox["max_lon"] - bbox["min_lon"]) * j / (W - 1)
            x, y = latlon_to_meters(lon, lat, ref_lon, ref_lat)
            z = elevation[i, j]
            vertices.append([x, z, -y])  # 3D: X=东, Y=上, Z=-南

    vertices = np.array(vertices, dtype=np.float32)

    # 生成三角面 (每2x2像素 = 2个三角形)
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            a = i * W + j
            b = a + 1
            c = a + W
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])

    faces = np.array(faces, dtype=np.int32)

    # 按高程着色 (绿色→棕色→灰色)
    z_min, z_max = elevation.min(), elevation.max()
    colors = np.zeros((len(vertices), 4), dtype=np.float32)
    for vi, v in enumerate(vertices):
        z_norm = (v[1] - z_min) / (z_max - z_min + 0.01)
        z_norm = max(0, min(1, z_norm))

        if z_norm < 0.1:
            # 海滩/低地 → 沙色
            color = [0.85, 0.80, 0.65]
        elif z_norm < 0.3:
            # 低坡 → 绿色
            color = [0.30, 0.55, 0.25]
        elif z_norm < 0.6:
            # 中坡 → 深绿
            color = [0.25, 0.45, 0.20]
        elif z_norm < 0.85:
            # 高坡 → 灰绿
            color = [0.45, 0.45, 0.40]
        else:
            # 山顶 → 浅灰
            color = [0.60, 0.60, 0.58]

        colors[vi] = [color[0], color[1], color[2], 1.0]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    print(f"  地形: {len(vertices):,} 顶点, {len(faces):,} 三角面")
    print(f"  范围: X:[{vertices[:,0].min():.0f}, {vertices[:,0].max():.0f}] "
          f"Z:[{vertices[:,2].min():.0f}, {vertices[:,2].max():.0f}] "
          f"Y:[{vertices[:,1].min():.0f}, {vertices[:,1].max():.0f}]")

    return mesh


# ============================================================
# Step 3: 将建筑放到地形上
# ============================================================


def sample_elevation(lon: float, lat: float,
                     elevation: np.ndarray,
                     bbox: dict = None) -> float:
    """
    从高程网格中双线性采样某点的高程

    Args:
        lon, lat: 经纬度
        elevation: 高程数组 [H(北→南), W(西→东)]
    """
    if bbox is None:
        bbox = HKUST

    H, W = elevation.shape

    # 经纬度 → 网格坐标 (行=北→南, 列=西→东)
    row_f = (bbox["max_lat"] - lat) / (bbox["max_lat"] - bbox["min_lat"]) * (H - 1)
    col_f = (lon - bbox["min_lon"]) / (bbox["max_lon"] - bbox["min_lon"]) * (W - 1)

    row_f = max(0, min(H - 1.0001, row_f))
    col_f = max(0, min(W - 1.0001, col_f))

    row0, col0 = int(row_f), int(col_f)
    row1 = min(row0 + 1, H - 1)
    col1 = min(col0 + 1, W - 1)

    dr, dc = row_f - row0, col_f - col0

    return (
        elevation[row0, col0] * (1 - dr) * (1 - dc)
        + elevation[row0, col1] * (1 - dr) * dc
        + elevation[row1, col0] * dr * (1 - dc)
        + elevation[row1, col1] * dr * dc
    )


def build_buildings_on_terrain(geojson_path: Path,
                                elevation: np.ndarray,
                                ref_lon: float = REF_LON,
                                ref_lat: float = REF_LAT
                                ) -> "trimesh.Trimesh":
    """
    从 GeoJSON 建筑数据挤出 3D 模型并放在地形上
    (复用 06_overture_demo.py 的逻辑但加上地形高度)

    Returns:
        建筑 mesh (不含地形)
    """
    import trimesh

    with open(geojson_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    print(f"  建筑总数: {len(features)}")

    # 高度估算 (同 06_overture_demo.py)
    KNOWN_HEIGHTS = {
        "學術大樓": 35.0, "Academic Building": 35.0,
        "李兆基商學大樓": 25.0, "Lee Shau Kee Business Building": 25.0,
        "鄭裕彤樓": 20.0, "Cheng Yu Tung Building": 20.0,
        "羅桂祥樓": 15.0, "Lo Kwee-Seong Building": 15.0,
        "賽馬會集賢樓": 30.0, "The Jockey Club Global Graduate Tower": 30.0,
    }
    CLASS_DEFAULTS = {
        "university": 20, "dormitory": 24, "apartments": 25,
        "residential": 20, "school": 15, "house": 9,
        "roof": 3, "service": 6, "parking": 4, "warehouse": 8, "unknown": 10,
    }

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(ref_lat))

    all_verts = []
    all_faces = []
    all_colors = []
    stats = {"ok": 0, "no_height": 0, "skip": 0}

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        # 获取高度
        height = props.get("height")
        if not height:
            nf = props.get("num_floors")
            if nf:
                height = float(nf) * 3.5
            else:
                names = props.get("names", {})
                name = names.get("primary", "") if names else ""
                found = False
                for key, h in KNOWN_HEIGHTS.items():
                    if key in name:
                        height = h; found = True; break
                if not found:
                    cls = props.get("class") or "unknown"
                    height = CLASS_DEFAULTS.get(cls, 10.0)
                    stats["no_height"] += 1

        # 地块平均高程
        polygon_rings = []
        if geom["type"] == "Polygon":
            polygon_rings = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polygon_rings = geom["coordinates"]
        else:
            stats["skip"] += 1
            continue

        for rings in polygon_rings:
            coords = rings[0]
            if len(coords) < 4:
                continue
            if coords[0] == coords[-1]:
                coords = coords[:-1]

            # 计算地块平均高程作为建筑底部
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            base_z = float(sample_elevation(
                sum(lons)/len(lons), sum(lats)/len(lats), elevation
            ))

            # 转换为米坐标
            xy = []
            for lon, lat in coords:
                x = (lon - ref_lon) * m_per_deg_lon
                y = (lat - ref_lat) * m_per_deg_lat
                xy.append((x, y))

            n = len(xy)
            # 底部顶点 (z = terrain) + 顶部顶点 (z = terrain + height)
            verts = []
            for x, y in xy:
                verts.append([x, base_z, -y])
            for x, y in xy:
                verts.append([x, base_z + height, -y])

            # 顶底三角剖分 (扇形)
            faces = []
            for i in range(1, n - 1):
                faces.append([0, i, i + 1])
            top_off = n
            for i in range(1, n - 1):
                faces.append([top_off, top_off + i + 1, top_off + i])

            # 侧面
            for i in range(n):
                j = (i + 1) % n
                faces.append([i, j, j + n])
                faces.append([i, j + n, i + n])

            # 颜色 (按高度)
            if height <= 10:
                color = [0.85, 0.75, 0.60]
            elif height <= 20:
                color = [0.75, 0.68, 0.58]
            elif height <= 30:
                color = [0.60, 0.58, 0.55]
            else:
                color = [0.50, 0.50, 0.52]

            offset = len(all_verts)
            faces = [[f[0]+offset, f[1]+offset, f[2]+offset] for f in faces]
            all_verts.extend(verts)
            all_faces.extend(faces)
            all_colors.extend([color + [1.0]] * len(verts))
            stats["ok"] += 1

    print(f"  建筑: OK={stats['ok']}, no_height={stats['no_height']}, skip={stats['skip']}")
    print(f"  总顶点: {len(all_verts):,}, 总面: {len(all_faces):,}")

    verts = np.array(all_verts, dtype=np.float32)
    faces = np.array(all_faces, dtype=np.int32)
    colors = np.array(all_colors, dtype=np.float32)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    print(f"  清理后: {len(mesh.vertices):,} 顶点, {len(mesh.faces):,} 面")

    return mesh


# ============================================================
# Step 4: 合并 & 导出
# ============================================================


def merge_and_export(terrain_mesh: "trimesh.Trimesh",
                     building_mesh: "trimesh.Trimesh",
                     output_path: Path):
    """合并地形+建筑，导出 GLB"""
    import trimesh

    print(f"\n合并地形 + 建筑...")
    combined = trimesh.util.concatenate([terrain_mesh, building_mesh])
    print(f"  合并后: {len(combined.vertices):,} 顶点, {len(combined.faces):,} 面")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(output_path), file_type="glb")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ 已导出: {output_path} ({size_mb:.1f} MB)")

    bounds = combined.bounds
    extent = bounds[1] - bounds[0]
    print(f"  尺寸: X={extent[0]:.0f}m  Y(高程)={extent[1]:.0f}m  Z={extent[2]:.0f}m")
    print(f"  高程范围: {bounds[0,1]:.0f}m - {bounds[1,1]:.0f}m")


# ============================================================
# 主程序
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="HKUST Demo 增强 — 添加 SRTM 地形",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="使用合成地形 (SRTM下载失败时自动使用)")
    parser.add_argument("--geojson", default="output/demo/hkust_buildings.geojson",
                        help="建筑 GeoJSON 路径")
    parser.add_argument("--output", default="output/demo/hkust_with_terrain.glb",
                        help="输出 GLB 路径")
    parser.add_argument("--terrain-only", action="store_true",
                        help="仅生成地形，不含建筑")

    args = parser.parse_args()

    geojson_path = PROJECT_ROOT / args.geojson
    output_path = PROJECT_ROOT / args.output

    print("=" * 60)
    print("  HKUST 3D Demo — 添加 SRTM 地形")
    print("=" * 60)

    # Step 1: 获取高程
    print("\n>>> Step 1: 获取高程数据")
    elev = get_elevation_data(use_synthetic=args.synthetic)

    # Step 2: 生成地形
    print("\n>>> Step 2: 生成地形 Mesh")
    terrain = build_terrain_mesh(elev)

    # Step 3: 建筑放置
    if args.terrain_only:
        print("\n>>> (跳过建筑 — terrain-only 模式)")
        import trimesh
        print(f"\n>>> Step 4: 导出地形")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        terrain.export(str(output_path), file_type="glb")
        print(f"✓ 已导出: {output_path}")
    else:
        print("\n>>> Step 3: 建筑放置到地形上")
        if not geojson_path.exists():
            print(f"✗ GeoJSON 不存在: {geojson_path}")
            print("  请先运行: python scripts/06_overture_demo.py --download-only")
            sys.exit(1)
        buildings = build_buildings_on_terrain(geojson_path, elev)

        print("\n>>> Step 4: 合并 & 导出")
        merge_and_export(terrain, buildings, output_path)

    print(f"\n{'=' * 60}")
    print(f"  🎉 完成!")
    print(f"  查看: 打开 https://gltf-viewer.donmccurdy.com/")
    print(f"       拖入 {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

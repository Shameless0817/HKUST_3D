#!/usr/bin/env python3
"""
HKUST 即时 Demo — Overture Maps 建筑脚印 → 3D 挤出模型 → GLB

零等待、零 API Key、完全免费
数据源: Overture Maps Foundation (https://overturemaps.org)
输出: 3D 建筑模型 GLB 文件，可直接在浏览器中查看

使用方法:
  python 06_overture_demo.py

输出:
  output/demo/hkust_overture.glb    — 3D 建筑模型
  output/demo/hkust_buildings.geojson — 原始建筑数据

查看:
  打开 https://gltf-viewer.donmccurdy.com/ 拖入 GLB 文件
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HKUST 校园区域（比纯粹坐标范围稍大，包含周边地形参考）
HKUST_BBOX = {
    "min_lon": 114.258,
    "min_lat": 22.330,
    "max_lon": 114.272,
    "max_lat": 22.342,
}

# 核心校园区域（更紧密的范围，仅建筑）
HKUST_CORE = {
    "min_lon": 114.261,
    "min_lat": 22.334,
    "max_lon": 114.269,
    "max_lat": 22.340,
}

# 建筑颜色方案 — 按高度分层
# 知名 HKUST 建筑的高度估算（米）
# 数据来源：公开信息 + 层数×3.5m估算
HKUST_KNOWN_HEIGHTS = {
    "學術大樓": 35.0,           # Academic Building — 大型综合建筑，依山而建
    "Academic Building": 35.0,
    "李兆基商學大樓": 25.0,     # Lee Shau Kee Business Building
    "Lee Shau Kee Business Building": 25.0,
    "鄭裕彤樓": 20.0,           # Cheng Yu Tung Building
    "Cheng Yu Tung Building": 20.0,
    "羅桂祥樓": 15.0,           # Lo Kwee-Seong Building
    "Lo Kwee-Seong Building": 15.0,
    "新翼大樓": 15.0,           # Annex Building
    "Annex Building": 15.0,
    "賽馬會集賢樓": 30.0,       # Jockey Club Global Graduate Tower
    "The Jockey Club Global Graduate Tower": 30.0,
    "圖書館": 15.0,             # Library
    "Library": 15.0,
    "大疆創新樓": 28.0,         # DJI Hall (UG Hall XI)
    "DJI Hall": 28.0,
}

# 按类别默认高度
CLASS_DEFAULT_HEIGHTS = {
    "university": 20.0,
    "dormitory": 24.0,
    "apartments": 25.0,
    "residential": 20.0,
    "school": 15.0,
    "house": 9.0,
    "roof": 3.0,
    "service": 6.0,
    "parking": 4.0,
    "warehouse": 8.0,
    "unknown": 10.0,
}

BUILDING_COLORS = [
    (0.85, 0.75, 0.60),  # 低层 — 暖米色
    (0.75, 0.70, 0.60),  # 中层 — 砂岩色
    (0.65, 0.60, 0.55),  # 中高层 — 灰色调
    (0.55, 0.55, 0.55),  # 高层 — 浅灰
    (0.45, 0.45, 0.50),  # 超高层 — 深灰蓝
]

TERRAIN_COLOR = (0.35, 0.50, 0.30)  # 地形 — 绿色
WATER_COLOR = (0.25, 0.40, 0.60)    # 水面 — 蓝色

# ============================================================
# Step 1: 下载数据
# ============================================================


def download_overture_buildings(bbox: dict = None,
                                output_path: Path = None) -> Path:
    """
    使用 overturemaps-py 下载 HKUST 区域建筑数据

    数据列:
      - geometry: 建筑轮廓多边形
      - height: 建筑高度（米）
      - num_floors: 楼层数
      - class: 建筑类型
      - sources: 数据来源
    """
    if bbox is None:
        bbox = HKUST_BBOX
    if output_path is None:
        output_path = OUTPUT_DIR / "hkust_buildings.geojson"

    bbox_str = (f"{bbox['min_lon']},{bbox['min_lat']},"
                f"{bbox['max_lon']},{bbox['max_lat']}")

    print(f"下载 Overture Maps 建筑数据...")
    print(f"  区域: {bbox_str}")
    print(f"  类型: building")

    import subprocess

    try:
        result = subprocess.run(
            ["overturemaps", "download",
             "--bbox", bbox_str,
             "-f", "geojson",
             "-t", "building",
             "-o", str(output_path)],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  输出: {result.stdout}")
            print(f"  错误: {result.stderr}")
            if "STAC" in (result.stderr + result.stdout):
                # STAC 可能超时，重试
                print("  重试（跳过STAC直接读S3）...")
                result = subprocess.run(
                    ["overturemaps", "download",
                     "--bbox", bbox_str,
                     "-f", "geojson",
                     "-t", "building",
                     "--no-stac",
                     "-o", str(output_path)],
                    capture_output=True, text=True, timeout=300
                )
        print(result.stdout)
        if result.returncode != 0:
            print(f"  CLI 下载失败: {result.stderr}")
            raise RuntimeError(result.stderr)
    except FileNotFoundError:
        print("✗ overturemaps CLI 未找到: pip install overturemaps")
        return None
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("\n尝试备选方案: OpenStreetMap Overpass API...")
        return _download_from_osm_fallback(bbox, output_path)

    # 检查文件
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ 下载完成: {output_path} ({size_mb:.1f} MB)")
        return output_path
    else:
        print("✗ 下载完成但文件不存在")
        return None


def _download_from_s3_fallback(bbox: dict, output_path: Path) -> Optional[Path]:
    """
    备选方案：直接从 Overture S3 读取 GeoParquet 并转换为 GeoJSON
    需要: pip install pyarrow geoarrow-pyarrow
    """
    try:
        import pyarrow.parquet as pq
        import pyarrow.dataset as ds
    except ImportError:
        print("  需要 pyarrow: pip install pyarrow")
        return None

    # Overture 最新 release 的 S3 路径
    s3_path = ("s3://overturemaps-us-west-2/release/2025-03-19.0/"
               "theme=buildings/type=building/")

    print(f"  从 S3 读取: {s3_path}")
    print("  (这可能需要几分钟，取决于网络)")

    try:
        dataset = ds.dataset(s3_path, format="parquet")
        # 按 bbox 过滤
        import pyarrow.compute as pc
        table = dataset.to_table(
            filter=(
                (pc.field("bbox.minx") < bbox["max_lon"])
                & (pc.field("bbox.maxx") > bbox["min_lon"])
                & (pc.field("bbox.miny") < bbox["max_lat"])
                & (pc.field("bbox.maxy") > bbox["min_lat"])
            )
        )

        # 转 GeoJSON
        features = []
        for batch in table.to_batches(max_chunksize=1000):
            for row in batch.to_pydict().items():
                pass  # 需要根据实际列结构解析

        print(f"  获取到 {len(features)} 个建筑")
    except Exception as e:
        print(f"  S3 访问失败: {e}")
        print("  回退到 OpenStreetMap 数据...")
        return _download_from_osm_fallback(bbox, output_path)


def _download_from_osm_fallback(bbox: dict, output_path: Path) -> Optional[Path]:
    """
    第二备选方案：通过 Overpass API 从 OpenStreetMap 获取建筑数据
    免费、无需 API Key、但速度较慢，大区域可能超时
    """
    import requests
    import time

    overpass_url = "https://overpass-api.de/api/interpreter"

    query = f"""
    [out:json][timeout:60];
    (
      way["building"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
      relation["building"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
    );
    out body;
    >;
    out skel qt;
    """

    print(f"  从 Overpass API 获取 OSM 建筑数据...")
    resp = requests.post(overpass_url, data=query, timeout=90)
    if resp.status_code != 200:
        print(f"  Overpass 请求失败: HTTP {resp.status_code}")
        return None

    osm_data = resp.json()
    elements = osm_data.get("elements", [])
    print(f"  获取到 {len(elements)} 个元素")

    # 转换为 GeoJSON FeatureCollection
    nodes = {}
    ways = []
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el["type"] == "way":
            ways.append(el)

    features = []
    for way in ways:
        if "nodes" not in way:
            continue
        coords = []
        for node_id in way["nodes"]:
            if node_id in nodes:
                coords.append(list(nodes[node_id]))
        if len(coords) < 4:
            continue
        # 闭合多边形
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        tags = way.get("tags", {})
        height = _parse_height(tags)
        levels = int(tags.get("building:levels", 0))

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": {
                "id": str(way["id"]),
                "height": height,
                "num_floors": levels,
                "class": tags.get("building", "unknown"),
                "name": tags.get("name", ""),
                "source": "osm",
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"✓ OSM 数据已保存: {output_path} ({len(features)} 个建筑)")
    print(f"  (OSM 数据的高度信息可能不完整，使用了估算值)")
    return output_path


def _parse_height(tags: dict) -> float:
    """从 OSM tags 中解析建筑高度（米）"""
    # 直接的高度值
    if "height" in tags:
        try:
            h = float(tags["height"].replace("m", "").strip())
            return h
        except (ValueError, AttributeError):
            pass

    # 从层数估算
    if "building:levels" in tags:
        try:
            levels = int(tags["building:levels"])
            return levels * 3.5  # 假设每层 3.5m
        except (ValueError, AttributeError):
            pass

    # 建筑类型默认高度
    type_defaults = {
        "university": 15.0,
        "dormitory": 20.0,
        "apartments": 25.0,
        "academic": 15.0,
        "library": 12.0,
        "sports_centre": 10.0,
        "hall": 8.0,
        "house": 6.0,
        "yes": 10.0,
    }
    btype = tags.get("building", "yes")
    return type_defaults.get(btype, 10.0)


# ============================================================
# Step 2: 2D 多边形 → 3D 挤出模型
# ============================================================


def extrude_polygon(coords: List[Tuple[float, float]],
                    base_height: float,
                    building_height: float,
                    reference_point: Tuple[float, float]
                    ) -> Tuple[List, List, List]:
    """
    将 2D 多边形（经纬度）挤出为 3D 网格

    Args:
        coords: [(lon, lat), ...]
        base_height: 底部高度（地形，设为 0）
        building_height: 建筑高度（米）
        reference_point: 参考点（经纬度），用于坐标转换

    Returns:
        (vertices, faces, colors)
    """
    ref_lon, ref_lat = reference_point

    # 经纬度 → 米（近似，HKUST 尺度下足够精确）
    METERS_PER_DEG_LAT = 111320.0
    METERS_PER_DEG_LON = 111320.0 * math.cos(math.radians(ref_lat))

    # 转换为相对于参考点的米坐标
    xy_coords = []
    for lon, lat in coords:
        x = (lon - ref_lon) * METERS_PER_DEG_LON
        y = (lat - ref_lat) * METERS_PER_DEG_LAT
        xy_coords.append((x, y))

    if len(xy_coords) < 3:
        return [], [], []

    # 底部顶点 (z=0) + 顶部顶点 (z=building_height)
    n = len(xy_coords)
    vertices = []
    for x, y in xy_coords:
        vertices.append([x, y, 0.0])           # 底部
    for x, y in xy_coords:
        vertices.append([x, y, building_height])  # 顶部

    # 生成面
    faces = []

    # 顶面和底面（简单的扇形三角剖分）
    # 底面 (顶点 0 到 n-1)
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])

    # 顶面 (顶点 n 到 2n-1)，改逆时针
    top_offset = n
    for i in range(1, n - 1):
        faces.append([top_offset, top_offset + i + 1, top_offset + i])

    # 侧面
    for i in range(n):
        j = (i + 1) % n
        bottom_0 = i
        bottom_1 = j
        top_0 = i + n
        top_1 = j + n
        faces.append([bottom_0, bottom_1, top_1])
        faces.append([bottom_0, top_1, top_0])

    # 颜色
    colors = _get_building_colors(building_height, len(vertices))

    return vertices, faces, colors


def _get_building_colors(height: float, num_vertices: int) -> List:
    """根据建筑高度分配颜色"""
    if height <= 5:
        color = BUILDING_COLORS[0]
    elif height <= 15:
        color = BUILDING_COLORS[1]
    elif height <= 30:
        color = BUILDING_COLORS[2]
    elif height <= 60:
        color = BUILDING_COLORS[3]
    else:
        color = BUILDING_COLORS[4]

    return [list(color) + [1.0]] * num_vertices


# ============================================================
# Step 3: 导出 GLB
# ============================================================


def build_3d_model(geojson_path: Path, output_path: Path,
                   reference_point: Tuple[float, float] = None):
    """
    将 GeoJSON 建筑数据转为 3D GLB 模型

    Args:
        geojson_path: Overture/OSM 建筑 GeoJSON
        output_path: 输出 GLB 文件路径
        reference_point: 坐标原点 (lon, lat)
    """
    import trimesh
    import numpy as np

    print(f"\n构建 3D 模型...")

    # 加载建筑数据
    with open(geojson_path) as f:
        data = json.load(f)

    features = data.get("features", [])
    print(f"  建筑总数: {len(features)}")

    # 计算参考点（所有建筑的几何中心）
    if reference_point is None:
        all_lons, all_lats = [], []
        for feat in features:
            geom = feat.get("geometry", {})
            if geom["type"] == "Polygon":
                for lon, lat in geom["coordinates"][0]:
                    all_lons.append(lon)
                    all_lats.append(lat)
            elif geom["type"] == "MultiPolygon":
                for poly in geom["coordinates"]:
                    for lon, lat in poly[0]:
                        all_lons.append(lon)
                        all_lats.append(lat)
        if all_lons:
            reference_point = (
                sum(all_lons) / len(all_lons),
                sum(all_lats) / len(all_lats),
            )

    print(f"  参考点: ({reference_point[0]:.6f}, {reference_point[1]:.6f})")

    # 逐个建筑挤出
    all_vertices = []
    all_faces = []
    all_colors = []
    stats = {"processed": 0, "skipped": 0, "no_height": 0}

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        # 获取高度 — 多种数据源兼容
        height = props.get("height")
        if height is None or height <= 0:
            # 尝试从 num_floors 估算
            num_floors = props.get("num_floors")
            if num_floors and num_floors > 0:
                height = float(num_floors) * 3.5
            else:
                # 尝试从建筑名称匹配已知高度
                names = props.get("names", {})
                name = names.get("primary", "") if names else ""
                height = None
                if name:
                    for key, h in HKUST_KNOWN_HEIGHTS.items():
                        if key in name:
                            height = h
                            break
                if height is None:
                    # 按类别估算
                    bld_class = props.get("class", "unknown") or "unknown"
                    height = CLASS_DEFAULT_HEIGHTS.get(bld_class, 10.0)
                    stats["no_height"] += 1

        # 处理 Polygon 和 MultiPolygon
        # GeoJSON Polygon: coordinates = [outer_ring, hole1, ...]
        #                outer_ring = [[lon,lat], [lon,lat], ...]
        # GeoJSON MultiPolygon: coordinates = [[outer_ring, ...], [outer_ring, ...], ...]
        polygon_rings = []
        if geom["type"] == "Polygon":
            polygon_rings = [geom["coordinates"]]  # 统一包装
        elif geom["type"] == "MultiPolygon":
            polygon_rings = geom["coordinates"]
        else:
            stats["skipped"] += 1
            continue

        for rings in polygon_rings:
            coords = rings[0]  # 外环 = 第一个 ring
            if len(coords) < 4:
                continue

            # 去除重复的闭合点
            if coords[0] == coords[-1]:
                coords = coords[:-1]

            verts, faces, colors = extrude_polygon(
                coords, 0.0, height, reference_point
            )

            if not verts:
                continue

            # 偏移面索引
            offset = len(all_vertices)
            faces = [[f[0] + offset, f[1] + offset, f[2] + offset] for f in faces]

            all_vertices.extend(verts)
            all_faces.extend(faces)
            all_colors.extend(colors)
            stats["processed"] += 1

    print(f"  处理: {stats['processed']} 个多边形")
    print(f"  跳过: {stats['skipped']} 个")
    print(f"  估算高度: {stats['no_height']} 个")
    print(f"  总顶点: {len(all_vertices):,}")
    print(f"  总三角面: {len(all_faces):,}")

    if len(all_vertices) == 0:
        print("✗ 没有生成任何3D几何体")
        return False

    # 创建 Trimesh 对象
    vertices = np.array(all_vertices, dtype=np.float32)
    faces = np.array(all_faces, dtype=np.int32)
    colors = np.array(all_colors, dtype=np.float32)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors,
    )

    # 清理：合并重复顶点，移除退化面
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    print(f"  清理后: {len(mesh.vertices):,} 顶点, {len(mesh.faces):,} 面")

    # 导出 GLB
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n✓ 模型已导出: {output_path} ({size_mb:.1f} MB)")

    # 添加元信息
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    print(f"  尺寸: {extent[0]:.0f}m × {extent[1]:.0f}m × {extent[2]:.0f}m")
    print(f"  范围: X:[{bounds[0,0]:.0f}, {bounds[1,0]:.0f}] "
          f"Y:[{bounds[0,1]:.0f}, {bounds[1,1]:.0f}] "
          f"Z:[{bounds[0,2]:.0f}, {bounds[1,2]:.0f}]")

    return True


# ============================================================
# 附加: 生成地形和水面
# ============================================================


def add_terrain_plane(mesh: "trimesh.Trimesh", bbox: dict,
                      reference_point: Tuple[float, float]
                      ) -> "trimesh.Trimesh":
    """
    添加地形平面作为基底
    （简化版 — 实际HKUST地形高差很大，这里用平面近似）
    """
    import trimesh
    import numpy as np

    METERS_PER_DEG_LAT = 111320.0
    METERS_PER_DEG_LON = 111320.0 * math.cos(math.radians(reference_point[1]))

    min_x = (bbox["min_lon"] - reference_point[0]) * METERS_PER_DEG_LON
    max_x = (bbox["max_lon"] - reference_point[0]) * METERS_PER_DEG_LON
    min_y = (bbox["min_lat"] - reference_point[1]) * METERS_PER_DEG_LAT
    max_y = (bbox["max_lat"] - reference_point[1]) * METERS_PER_DEG_LAT

    # 扩大一点
    margin = 50
    min_x -= margin; max_x += margin
    min_y -= margin; max_y += margin

    terrain_verts = np.array([
        [min_x, min_y, -1.0],
        [max_x, min_y, -1.0],
        [max_x, max_y, -1.0],
        [min_x, max_y, -1.0],
    ])
    terrain_faces = np.array([[0, 1, 2], [0, 2, 3]])
    terrain_colors = np.array([list(TERRAIN_COLOR) + [1.0]] * 4)

    terrain = trimesh.Trimesh(
        vertices=terrain_verts,
        faces=terrain_faces,
        vertex_colors=terrain_colors,
    )

    return trimesh.util.concatenate([mesh, terrain])


# ============================================================
# 主程序
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="HKUST 即时 Demo — Overture Maps 建筑 → 3D GLB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 一键生成 Demo
  python 06_overture_demo.py

  # 分步执行
  python 06_overture_demo.py --download-only    # 仅下载数据
  python 06_overture_demo.py --build-only       # 仅构建模型（需已有GeoJSON）
  python 06_overture_demo.py --add-terrain      # 添加地形平面

  # 查看结果
  # 打开 https://gltf-viewer.donmccurdy.com/ 拖入 output/demo/hkust_overture.glb
        """,
    )

    parser.add_argument("--download-only", action="store_true",
                        help="仅下载建筑数据，不构建3D模型")
    parser.add_argument("--build-only", action="store_true",
                        help="仅构建3D模型（使用已有GeoJSON）")
    parser.add_argument("--add-terrain", action="store_true",
                        help="添加地形基底平面")
    parser.add_argument("--geojson", default=None,
                        help="GeoJSON 文件路径（默认: output/demo/hkust_buildings.geojson）")
    parser.add_argument("--output", default=None,
                        help="GLB 输出路径（默认: output/demo/hkust_overture.glb）")

    args = parser.parse_args()

    geojson_path = Path(args.geojson) if args.geojson else OUTPUT_DIR / "hkust_buildings.geojson"
    glb_path = Path(args.output) if args.output else OUTPUT_DIR / "hkust_overture.glb"

    print("=" * 60)
    print("  HKUST 即时 3D Demo — Overture Maps 建筑模型")
    print("  零 API Key · 零等待 · 完全免费")
    print("=" * 60)

    # Step 1: 下载数据
    if not args.build_only:
        print("\n>>> Step 1: 下载建筑数据")
        result = download_overture_buildings(output_path=geojson_path)
        if result is None:
            print("\n尝试从 OpenStreetMap 获取数据...")
            result = _download_from_osm_fallback(HKUST_BBOX, geojson_path)
        if result is None:
            print("\n✗ 无法获取建筑数据。请检查网络连接。")
            print("  也可以手动准备 GeoJSON 后用 --build-only 构建")
            sys.exit(1)

    if args.download_only:
        print(f"\n数据已保存: {geojson_path}")
        print("运行以下命令构建3D模型:")
        print(f"  python {__file__} --build-only")
        return

    # Step 2: 构建 3D 模型
    print("\n>>> Step 2: 构建 3D 挤出模型")
    if not geojson_path.exists():
        print(f"✗ GeoJSON 文件不存在: {geojson_path}")
        print("  请先运行: python 06_overture_demo.py --download-only")
        sys.exit(1)

    success = build_3d_model(geojson_path, glb_path)

    # Step 3: (可选) 添加地形
    if success and args.add_terrain:
        print("\n>>> Step 3: 添加地形平面")
        import trimesh
        mesh = trimesh.load(str(glb_path))
        # 用 HKUST 的近似中心点
        mesh_with_terrain = add_terrain_plane(
            mesh, HKUST_BBOX, (114.265, 22.337)
        )
        glb_terrain = glb_path.parent / "hkust_overture_with_terrain.glb"
        mesh_with_terrain.export(str(glb_terrain), file_type="glb")
        print(f"✓ 带地形版本: {glb_terrain}")

    if success:
        print(f"\n{'=' * 60}")
        print(f"  🎉 Demo 生成成功!")
        print(f"  模型文件: {glb_path}")
        print(f"\n  查看方式:")
        print(f"    1. 打开 https://gltf-viewer.donmccurdy.com/")
        print(f"    2. 拖入 {glb_path.name}")
        print(f"    3. 或使用 VS Code 的 glTF 预览插件")
        print(f"\n  模型说明:")
        print(f"    - 建筑颜色按高度分层（浅色=低层, 深色=高层）")
        print(f"    - 模型原点为 HKUST 校园几何中心")
        print(f"    - 坐标单位为米")
        print(f"\n  等 CSDI API Key 到后，可替换为照片级纹理模型")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

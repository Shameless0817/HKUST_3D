#!/usr/bin/env python3
"""
方案一（首选）：通过香港 CSDI API 下载 HKUST 区域 3D 模型

数据源: 香港地政总署 (LandsD) 三维可视化地图
覆盖: 香港全境，~122,000+栋建筑 + 基建
来源: 倾斜航空摄影 (oblique aerial imagery) → Mesh + 纹理
格式: OBJ, Cesium 3D Tiles, OSGB
费用: 完全免费

使用前：
  1. 发送邮件至 3dmap@landsd.gov.hk 申请 API Key
  2. 将 API Key 填入 config/api_keys.json 的 landsd_api_key 字段
  3. 运行: python 01_csdi_download.py

参考:
  - CSDI Portal: https://portal.csdi.gov.hk
  - Open3Dhk: https://3d.map.gov.hk
  - API文档: https://portal.csdi.gov.hk/csdi-webpage/apidoc/3d-visualisation-map-api
"""

import requests
import json
import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "api_keys.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "csdi"

# HKUST 坐标范围
HKUST_BBOX = {
    "min_lat": 22.330,
    "max_lat": 22.342,
    "min_lon": 114.258,
    "max_lon": 114.272,
}

# CSDI API 端点
API_BASE = "https://data.map.gov.hk/api/3d-data"

# 可用的 tileset 列表
TILESETS = {
    "f2": "3D Visualisation Map (textured, whole HK)",
    "t_bi_f1": "3D Visualisation Map - Buildings & Infrastructure (non-textured)",
    "t_t_f1": "3D Visualisation Map - Terrain (non-textured)",
    "building": "3D Spatial Data - Buildings",
    "infrastructure": "3D Spatial Data - Infrastructure",
}

# ============================================================
# 工具函数
# ============================================================


def load_api_key() -> str:
    """从配置文件加载 API Key"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        key = config.get("landsd_api_key", "")
        if key and key != "YOUR_API_KEY_HERE":
            return key

    print("=" * 60)
    print("⚠️  未找到 LandsD API Key")
    print("=" * 60)
    print()
    print("请先申请 API Key:")
    print("  1. 发送邮件至 3dmap@landsd.gov.hk")
    print("  2. 说明用途（如：学术研究、HKUST校园三维重建）")
    print("  3. 将收到的 API Key 填入:")
    print(f"     {CONFIG_FILE}")
    print()
    print("邮件模板请参考: email_template.md")
    print("=" * 60)

    key = input("\n或者现在直接输入 API Key（留空跳过）: ").strip()
    if key:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = json.load(f)
        config["landsd_api_key"] = key
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ API Key 已保存到 {CONFIG_FILE}")
    return key


def tile_in_bbox(tile_bounds: Dict, bbox: Dict) -> bool:
    """检查 3D tile 是否与目标包围盒相交"""
    if not tile_bounds:
        return True  # 无边界信息时默认下载
    # 简单的AABB相交检测
    return not (
        tile_bounds.get("max_lat", 90) < bbox["min_lat"]
        or tile_bounds.get("min_lat", -90) > bbox["max_lat"]
        or tile_bounds.get("max_lon", 180) < bbox["min_lon"]
        or tile_bounds.get("min_lon", -180) > bbox["max_lon"]
    )


def download_file(url: str, output_path: Path, params: Dict = None,
                  retries: int = 3) -> bool:
    """下载单个文件，含重试机制"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(resp.content)
                return True
            elif resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  速率限制，等待 {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}: {url}")
                return False
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  下载失败: {e}")
                return False
    return False


# ============================================================
# 核心功能
# ============================================================


def list_available_tilesets(api_key: str):
    """列出所有可用的 tileset"""
    print("\n" + "=" * 60)
    print("可用 Tilesets")
    print("=" * 60)
    for name, desc in TILESETS.items():
        url = f"{API_BASE}/3dtiles/{name}/tileset.json"
        resp = requests.get(url, params={"key": api_key})
        status = "✓ 可访问" if resp.status_code == 200 else "✗ 不可访问"
        print(f"  [{status}] {name}: {desc}")
    print()


def get_tileset_info(api_key: str, tileset_name: str) -> Optional[Dict]:
    """获取 tileset 元信息"""
    url = f"{API_BASE}/3dtiles/{tileset_name}/tileset.json"
    resp = requests.get(url, params={"key": api_key})
    if resp.status_code != 200:
        print(f"✗ 无法访问 tileset: {tileset_name}")
        return None
    return resp.json()


def count_tiles(node: Dict) -> int:
    """递归统计 tile 数量"""
    count = 1
    for child in node.get("children", []):
        count += count_tiles(child)
    return count


def collect_tiles(node: Dict, bbox: Dict, api_key: str,
                  tileset_name: str) -> List[Dict]:
    """递归收集与 bbox 相交的 tile 下载任务"""
    tiles = []

    # 检查当前 tile 的边界
    bvol = node.get("boundingVolume", {})
    region = bvol.get("region", [])  # Cesium region: [west, south, east, north, minH, maxH]
    bounds = None
    if len(region) >= 4:
        bounds = {
            "min_lon": region[0],  # west (radians)
            "min_lat": region[1],  # south (radians)
            "max_lon": region[2],  # east (radians)
            "max_lat": region[3],  # north (radians)
        }

    if "content" in node and "uri" in node["content"]:
        uri = node["content"]["uri"]
        if tile_in_bbox(bounds, bbox):
            tiles.append({
                "uri": uri,
                "bounds": bounds,
                "geometricError": node.get("geometricError", 0),
            })

    for child in node.get("children", []):
        tiles.extend(collect_tiles(child, bbox, api_key, tileset_name))

    return tiles


def download_tileset(api_key: str, tileset_name: str = "f2",
                     bbox: Dict = None, max_workers: int = 8,
                     dry_run: bool = False):
    """
    下载 3D Tiles 数据集

    Args:
        api_key: LandsD API Key
        tileset_name: tileset 名称 (默认 f2 = 纹理模型)
        bbox: 包围盒 (默认 HKUST)
        max_workers: 并行下载线程数
        dry_run: 仅统计不下载
    """
    if bbox is None:
        bbox = HKUST_BBOX

    print(f"\n{'=' * 60}")
    print(f"下载 Tileset: {tileset_name} ({TILESETS.get(tileset_name, 'Unknown')})")
    print(f"目标区域: lat [{bbox['min_lat']}, {bbox['max_lat']}], "
          f"lon [{bbox['min_lon']}, {bbox['max_lon']}]")
    print(f"{'=' * 60}\n")

    # 获取 tileset 结构
    tileset = get_tileset_info(api_key, tileset_name)
    if not tileset:
        return

    root = tileset.get("root", {})
    total_tiles = count_tiles(root)
    print(f"Tileset 共 {total_tiles} 个 tile")

    # 收集目标区域的 tile
    print("正在筛选目标区域内的 tile...")
    target_tiles = collect_tiles(root, bbox, api_key, tileset_name)
    print(f"目标区域内: {len(target_tiles)} 个 tile")

    if not target_tiles:
        print("⚠️  没有找到覆盖目标区域的 tile，请检查坐标范围")
        return

    if dry_run:
        print("\n[Dry Run] 将要下载的文件:")
        for t in target_tiles[:10]:
            print(f"  {t['uri']}")
        if len(target_tiles) > 10:
            print(f"  ... 以及 {len(target_tiles) - 10} 个文件")
        return

    # 并行下载
    output_dir = OUTPUT_DIR / tileset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 tileset.json
    tileset_path = output_dir / "tileset.json"
    with open(tileset_path, "w") as f:
        json.dump(tileset, f, indent=2)
    print(f"✓ 保存 tileset.json")

    print(f"\n开始下载 {len(target_tiles)} 个 tile (并行: {max_workers})...")
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for tile in target_tiles:
            uri = tile["uri"]
            output_path = output_dir / uri
            url = f"{API_BASE}/3dtiles/{tileset_name}/{uri}"
            futures[executor.submit(
                download_file, url, output_path, {"key": api_key}
            )] = uri

        for i, future in enumerate(as_completed(futures), 1):
            uri = futures[future]
            if future.result():
                success += 1
            else:
                failed += 1
            if i % 50 == 0 or i == len(target_tiles):
                print(f"  进度: {i}/{len(target_tiles)} "
                      f"(成功: {success}, 失败: {failed})")

    print(f"\n{'=' * 60}")
    print(f"下载完成! 成功: {success}, 失败: {failed}")
    print(f"输出目录: {output_dir}")
    print(f"{'=' * 60}")


def download_by_map_sheet(api_key: str, sheet_number: str):
    """
    按 1:1000 地图编号下载 OBJ 格式数据

    地图编号格式示例: 6-NW-12C (具体编号需在 LandsD 地图索引中查询)
    """
    # 此功能需要通过 LandsD Download API 实现
    # 具体端点和参数需联系 LandsD 获取
    print("按地图编号下载功能需要:")
    print("1. 查询 HKUST 区域对应的 1:1000 地图编号")
    print("2. 通过 Download API 下载 OBJ 文件")
    print("\n建议先在 Open3Dhk (https://3d.map.gov.hk) 中定位到 HKUST")
    print("查看页面中显示的 Map Sheet Number，然后使用此功能下载")


def convert_b3dm_to_glb(input_dir: Path = None):
    """
    将下载的 .b3dm 文件转换为 .glb 格式
    每个 b3dm = 28字节 header + GLB body
    """
    if input_dir is None:
        input_dir = OUTPUT_DIR / "f2"

    if not input_dir.exists():
        print(f"目录不存在: {input_dir}")
        return

    b3dm_files = list(input_dir.rglob("*.b3dm"))
    if not b3dm_files:
        print(f"目录中没有 .b3dm 文件: {input_dir}")
        return

    print(f"找到 {len(b3dm_files)} 个 .b3dm 文件")
    output_base = PROJECT_ROOT / "output" / "processed" / "glb"
    converted = 0

    for b3dm_path in b3dm_files:
        data = b3dm_path.read_bytes()
        # b3dm header: 28 bytes (magic, version, byteLength, featureTableJSONByteLength, featureTableBinaryByteLength, batchTableJSONByteLength, batchTableBinaryByteLength)
        if data[:4] != b"b3dm":
            print(f"  跳过非b3dm文件: {b3dm_path.name}")
            continue

        glb_data = data[28:]  # 跳过28字节header
        if glb_data[:4] != b"glTF":
            print(f"  警告: {b3dm_path.name} 的 payload 不是有效 GLB")
            continue

        rel_path = b3dm_path.relative_to(input_dir)
        glb_path = output_base / rel_path.with_suffix(".glb")
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(glb_data)
        converted += 1

    print(f"✓ 已转换 {converted} 个文件到 {output_base}")


# ============================================================
# 主程序
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="HK CSDI 3D 模型下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 列出所有可用 tileset
  python 01_csdi_download.py --list

  # 下载 HKUST 区域纹理3D模型 (推荐)
  python 01_csdi_download.py --tileset f2

  # 仅统计不下载
  python 01_csdi_download.py --tileset f2 --dry-run

  # 下载非纹理建筑物模型
  python 01_csdi_download.py --tileset t_bi_f1

  # 将下载的 B3DM 转换为 GLB
  python 01_csdi_download.py --convert

  # 自定义区域下载
  python 01_csdi_download.py --tileset f2 --bbox 22.33,22.34,114.26,114.27
        """,
    )

    parser.add_argument("--list", action="store_true", help="列出可用 tileset")
    parser.add_argument("--tileset", default=None, help="要下载的 tileset 名称 (默认: f2)")
    parser.add_argument("--bbox", default=None,
                        help="包围盒: min_lat,max_lat,min_lon,max_lon")
    parser.add_argument("--dry-run", action="store_true", help="仅统计不下载")
    parser.add_argument("--workers", type=int, default=8, help="并行下载线程数")
    parser.add_argument("--convert", action="store_true",
                        help="将已下载的 b3dm 转为 glb")
    parser.add_argument("--sheet", default=None, help="按 1:1000 地图编号下载")

    args = parser.parse_args()

    # 转换模式（不需要 API key）
    if args.convert:
        convert_b3dm_to_glb()
        return

    # 需要 API key 的操作
    api_key = load_api_key()
    if not api_key:
        print("❌ 需要 API Key 才能继续")
        sys.exit(1)

    if args.list:
        list_available_tilesets(api_key)
        return

    if args.sheet:
        download_by_map_sheet(api_key, args.sheet)
        return

    # 默认：下载
    tileset = args.tileset or "f2"
    bbox = None
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        bbox = {
            "min_lat": parts[0], "max_lat": parts[1],
            "min_lon": parts[2], "max_lon": parts[3],
        }

    download_tileset(
        api_key=api_key,
        tileset_name=tileset,
        bbox=bbox,
        max_workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

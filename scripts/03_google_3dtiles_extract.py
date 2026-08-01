#!/usr/bin/env python3
"""
方案三（补充）：从 Google Photorealistic 3D Tiles 提取 3D 模型

数据源: Google Map Tiles API / Cesium ion
格式: Cesium 3D Tiles (b3dm/glb)
端点: https://tile.googleapis.com/v1/3dtiles/root.json

注意:
  - Google 官方禁止缓存/存储/导出 tile 数据（仅允许实时流式渲染）
  - 本工具仅供学术研究参考，了解 3D Tiles 协议的结构
  - 需要一个 Google Cloud 项目并启用 Map Tiles API
  - 每个 root tileset 请求有效期约 3 小时

前置条件:
  1. 创建 Google Cloud 项目: https://console.cloud.google.com
  2. 启用 Map Tiles API
  3. 创建 API Key 并填入 config/api_keys.json

参考:
  - https://developers.google.com/maps/documentation/tile/3d-tiles
  - https://cesium.com/platform/cesium-ion/
  - https://cesium.com/learn/cesiumjs-learn/cesiumjs-photorealistic-3d-tiles/
"""

import requests
import json
import math
import struct
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "api_keys.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "google_3dtiles"

# Google Map Tiles API 端点
GOOGLE_3DTILES_ROOT = "https://tile.googleapis.com/v1/3dtiles/root.json"

# HKUST 坐标范围
HKUST_BBOX = {
    "min_lat": 22.330,
    "max_lat": 22.342,
    "min_lon": 114.258,
    "max_lon": 114.272,
}


# ============================================================
# 工具函数
# ============================================================


def load_google_api_key() -> Optional[str]:
    """从配置文件加载 Google API Key"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        key = config.get("google_maps_api_key", "")
        if key and key != "YOUR_GOOGLE_API_KEY_HERE":
            return key

    print("=" * 60)
    print("⚠️  未找到 Google Maps API Key")
    print("=" * 60)
    print()
    print("获取步骤:")
    print("  1. 访问 https://console.cloud.google.com")
    print("  2. 创建项目或选择现有项目")
    print("  3. 启用 Map Tiles API")
    print("  4. 创建 API Key → 限制到 Map Tiles API")
    print(f"  5. 填入: {CONFIG_FILE}")
    print()

    key = input("或者现在直接输入 API Key（留空跳过）: ").strip()
    if key:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                config = json.load(f)
        config["google_maps_api_key"] = key
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
    return key


# ============================================================
# 3D Tiles 协议解析
# ============================================================


def parse_b3dm_header(data: bytes) -> Dict:
    """
    解析 B3DM (Batched 3D Model) 文件头

    Header 结构 (28 bytes):
      magic (4 bytes): "b3dm"
      version (4 bytes): uint32, 目前为 1
      byteLength (4 bytes): uint32, 整个文件长度
      featureTableJSONByteLength (4 bytes): uint32
      featureTableBinaryByteLength (4 bytes): uint32
      batchTableJSONByteLength (4 bytes): uint32
      batchTableBinaryByteLength (4 bytes): uint32
    """
    if len(data) < 28 or data[:4] != b"b3dm":
        return {"error": "Not a valid B3DM file"}

    header = {
        "magic": data[0:4].decode("ascii"),
        "version": struct.unpack("<I", data[4:8])[0],
        "byteLength": struct.unpack("<I", data[8:12])[0],
        "featureTableJSONByteLength": struct.unpack("<I", data[12:16])[0],
        "featureTableBinaryByteLength": struct.unpack("<I", data[16:20])[0],
        "batchTableJSONByteLength": struct.unpack("<I", data[20:24])[0],
        "batchTableBinaryByteLength": struct.unpack("<I", data[24:28])[0],
    }

    # GLB payload 起始位置
    header["glb_offset"] = 28 + header["featureTableJSONByteLength"] \
        + header["featureTableBinaryByteLength"] \
        + header["batchTableJSONByteLength"] \
        + header["batchTableBinaryByteLength"]

    return header


def extract_glb_from_b3dm(b3dm_data: bytes) -> Optional[bytes]:
    """从 B3DM 中提取 GLB 数据"""
    header = parse_b3dm_header(b3dm_data)
    if "error" in header:
        return None
    glb_offset = header["glb_offset"]
    return b3dm_data[glb_offset:]


def extract_glb_from_tile(tile_data: bytes, tile_uri: str) -> Optional[bytes]:
    """根据 tile URI 类型提取 GLB 数据"""
    if tile_uri.endswith(".b3dm"):
        return extract_glb_from_b3dm(tile_data)
    elif tile_uri.endswith(".glb"):
        return tile_data  # 直接就是 GLB
    elif tile_uri.endswith(".json"):
        # 外部的 tileset，不是几何数据
        return None
    else:
        # 未知格式，尝试直接作为 GLB
        if tile_data[:4] == b"glTF":
            return tile_data
        return None


# ============================================================
# Tile 空间索引
# ============================================================


def latlon_to_radian(lat: float, lon: float) -> Tuple[float, float, float]:
    """经纬度 → 弧度 → ECEF方向（单位球面上的点）"""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    x = math.cos(lat_rad) * math.cos(lon_rad)
    y = math.cos(lat_rad) * math.sin(lon_rad)
    z = math.sin(lat_rad)
    return x, y, z


def bbox_in_region(bbox: Dict, region: List[float]) -> bool:
    """
    检查经纬度包围盒是否与 Cesium region 相交

    region: [west, south, east, north, minHeight, maxHeight] (弧度)
    """
    if len(region) < 4:
        return True  # 无有效边界时默认包含

    r_west = math.degrees(region[0])
    r_south = math.degrees(region[1])
    r_east = math.degrees(region[2])
    r_north = math.degrees(region[3])

    # 经度跨 180° 边界的情况
    if r_west > r_east:  # 跨越反子午线
        # 简化：只处理不跨越的情况
        r_west, r_east = r_east, r_west

    return not (
        bbox["min_lat"] > r_north
        or bbox["max_lat"] < r_south
        or bbox["min_lon"] > r_east
        or bbox["max_lon"] < r_west
    )


# ============================================================
# Tileset 遍历和下载
# ============================================================


class TilesetExplorer:
    """3D Tiles 数据集浏览器"""

    def __init__(self, api_key: str, bbox: Dict = None):
        self.api_key = api_key
        self.bbox = bbox or HKUST_BBOX
        self.session = requests.Session()
        self.stats = {"total_tiles": 0, "in_bbox": 0, "downloaded": 0, "skipped": 0,
                      "converted": 0}
        self.tile_list = []

    def get_tileset(self, url: str = None) -> Optional[Dict]:
        """获取 tileset JSON"""
        if url is None:
            url = f"{GOOGLE_3DTILES_ROOT}?key={self.api_key}"

        resp = self.session.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"✗ HTTP {resp.status_code}: {url}")
            print(f"  响应: {resp.text[:200]}")
            return None
        return resp.json()

    def explore(self, node: Dict, base_url: str, depth: int = 0,
                max_depth: int = 15) -> None:
        """递归遍历 tileset 树结构，收集与目标区域相交的 tile"""
        self.stats["total_tiles"] += 1

        if depth > max_depth:
            return

        # 检查包围体
        bvol = node.get("boundingVolume", {})
        region = bvol.get("region", [])

        if region and len(region) >= 4:
            if not bbox_in_region(self.bbox, region):
                self.stats["skipped"] += 1
                return  # 不在目标区域内，剪枝

        self.stats["in_bbox"] += 1

        # 记录 tile 内容
        if "content" in node and "uri" in node["content"]:
            uri = node["content"]["uri"]
            if not uri.startswith("http"):
                # 相对路径 → 绝对路径
                import urllib.parse
                uri = urllib.parse.urljoin(base_url, uri)

            # 对于 Google 3D Tiles，需要在 URI 上附加 API Key
            if "?" in uri:
                uri += f"&key={self.api_key}"
            else:
                uri += f"?key={self.api_key}"

            self.tile_list.append({
                "uri": uri,
                "depth": depth,
                "geometricError": node.get("geometricError", 0),
                "refine": node.get("refine", "ADD"),
            })

        # 递归子节点
        for child in node.get("children", []):
            self.explore(child, base_url, depth + 1, max_depth)

    def download_tiles(self, output_dir: Path, max_workers: int = 4,
                       max_tiles: int = 500, convert_to_glb: bool = True):
        """下载收集到的 tile"""
        if max_tiles and len(self.tile_list) > max_tiles:
            # 按 geometricError 排序，优先下载精细 tile
            self.tile_list.sort(key=lambda t: t["geometricError"])
            self.tile_list = self.tile_list[:max_tiles]
            print(f"限制下载前 {max_tiles} 个精细度最高的 tile")

        print(f"准备下载 {len(self.tile_list)} 个 tile...")
        output_dir.mkdir(parents=True, exist_ok=True)
        glb_dir = output_dir / "glb"
        glb_dir.mkdir(exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, tile in enumerate(self.tile_list):
                output_path = output_dir / f"tile_{i:06d}.b3dm"
                futures[executor.submit(
                    self._download_single, tile, output_path, convert_to_glb, glb_dir
                )] = (tile, output_path)

            for future in as_completed(futures):
                tile, output_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ✗ 下载失败: {tile['uri'][:80]} - {e}")

        print(f"\n下载统计:")
        print(f"  总 tile: {self.stats['total_tiles']}")
        print(f"  在区域内: {self.stats['in_bbox']}")
        print(f"  已下载: {self.stats['downloaded']}")
        print(f"  已跳过: {self.stats['skipped']}")
        print(f"  已转GLB: {self.stats['converted']}")

    def _download_single(self, tile: Dict, output_path: Path,
                         convert_to_glb: bool, glb_dir: Path) -> bool:
        """下载单个 tile"""
        uri = tile["uri"]

        try:
            resp = self.session.get(uri, timeout=30)
            if resp.status_code != 200:
                print(f"  ✗ HTTP {resp.status_code}: {uri[:80]}")
                return False

            data = resp.content
            output_path.write_bytes(data)
            self.stats["downloaded"] += 1

            if convert_to_glb:
                glb_data = extract_glb_from_tile(data, uri)
                if glb_data:
                    glb_path = glb_dir / f"{output_path.stem}.glb"
                    glb_path.write_bytes(glb_data)
                    self.stats["converted"] += 1

            return True
        except Exception as e:
            return False


# ============================================================
# Cesium ion 集成（可选）
# ============================================================


def cesium_ion_info():
    """Cesium ion 平台信息"""
    print("""
Cesium ion 是 Google Photorealistic 3D Tiles 的官方分发平台。

平台特性:
  - 提供 photorealistic 3D tiles 的流式访问
  - 免费套餐: 每月 5GB 流量额度
  - 支持 CesiumJS, Unreal, Unity, Omniverse, Godot 等

访问方式:
  1. 注册: https://ion.cesium.com/signup
  2. 获取 Access Token
  3. CesiumJS 中直接使用:
     const tileset = viewer.scene.primitives.add(
       await Cesium.createGooglePhotorealistic3DTileset()
     );
  4. 仅支持实时渲染，不支持下载/导出原始数据

局限性:
  - 无法导出 OBJ/FBX 等离线格式
  - Google Maps ToS 禁止缓存 tile 数据
  - 流量超过免费额度后需付费
""")


# ============================================================
# 主程序
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Google Photorealistic 3D Tiles 数据提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 探索 tileset 结构（不下载）
  python 03_google_3dtiles_extract.py explore --max-depth 8

  # 下载 HKUST 区域的 tile（限制 200 个）
  python 03_google_3dtiles_extract.py download --max-tiles 200

  # 查看 Cesium ion 使用说明
  python 03_google_3dtiles_extract.py cesium-info

  # 将已下载的 b3dm 批量转 glb
  python 03_google_3dtiles_extract.py convert --input output/google_3dtiles/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # explore
    explore_parser = subparsers.add_parser("explore", help="探索 tileset 结构")
    explore_parser.add_argument("--max-depth", type=int, default=10,
                                help="最大遍历深度 (默认: 10)")

    # download
    dl_parser = subparsers.add_parser("download", help="下载 tile")
    dl_parser.add_argument("--max-tiles", type=int, default=500,
                           help="最大下载数量 (默认: 500)")
    dl_parser.add_argument("--max-workers", type=int, default=4,
                           help="并行下载线程 (默认: 4)")
    dl_parser.add_argument("--no-convert", action="store_true",
                           help="不自动转换为 GLB")
    dl_parser.add_argument("--output", default=None, help="输出目录")

    # convert
    convert_parser = subparsers.add_parser("convert", help="b3dm 转 glb")
    convert_parser.add_argument("--input", required=True, help="输入目录")
    convert_parser.add_argument("--output", default=None, help="输出目录")

    # cesium-info
    subparsers.add_parser("cesium-info", help="Cesium ion 平台说明")

    args = parser.parse_args()

    if args.command == "cesium-info":
        cesium_ion_info()
        return

    if args.command == "explore":
        api_key = load_google_api_key()
        if not api_key:
            return

        print("正在获取 Google 3D Tiles 根节点...")
        explorer = TilesetExplorer(api_key)
        tileset = explorer.get_tileset()
        if not tileset:
            print("✗ 无法获取 tileset，请检查 API Key 是否启用了 Map Tiles API")
            return

        print(f"正在遍历 tileset (最大深度: {args.max_depth})...")
        root = tileset.get("root", {})
        explorer.explore(root, GOOGLE_3DTILES_ROOT, max_depth=args.max_depth)

        print(f"\n探索结果:")
        print(f"  总节点: {explorer.stats['total_tiles']}")
        print(f"  HKUST区域内: {explorer.stats['in_bbox']}")
        print(f"  可下载 tile: {len(explorer.tile_list)}")

    elif args.command == "download":
        api_key = load_google_api_key()
        if not api_key:
            return

        output = Path(args.output) if args.output else OUTPUT_DIR
        output.mkdir(parents=True, exist_ok=True)

        print("正在获取 tileset 并遍历...")
        explorer = TilesetExplorer(api_key)
        tileset = explorer.get_tileset()
        if not tileset:
            return

        root = tileset.get("root", {})
        explorer.explore(root, GOOGLE_3DTILES_ROOT, max_depth=10)

        explorer.download_tiles(
            output,
            max_workers=args.max_workers,
            max_tiles=args.max_tiles,
            convert_to_glb=not args.no_convert,
        )

    elif args.command == "convert":
        input_dir = Path(args.input)
        output_dir = Path(args.output) if args.output else input_dir / "glb"
        output_dir.mkdir(parents=True, exist_ok=True)

        b3dm_files = list(input_dir.glob("*.b3dm"))
        print(f"找到 {len(b3dm_files)} 个 .b3dm 文件")

        converted = 0
        for b3dm_path in b3dm_files:
            data = b3dm_path.read_bytes()
            glb = extract_glb_from_b3dm(data)
            if glb:
                glb_path = output_dir / f"{b3dm_path.stem}.glb"
                glb_path.write_bytes(glb)
                converted += 1

        print(f"✓ 已转换 {converted} 个文件到 {output_dir}")

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("⚠️  Google Maps ToS 提醒")
        print("=" * 60)
        print("Google 明确禁止下载、缓存或存储 3D Tiles 数据。")
        print("这些数据仅供实时流式渲染使用。")
        print("提取数据用于离线使用违反 Google 服务条款。")
        print()
        print("强烈建议优先使用方案一（HK CSDI 合法数据）: 01_csdi_download.py")
        print("=" * 60)


if __name__ == "__main__":
    main()

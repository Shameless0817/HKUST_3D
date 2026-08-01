#!/usr/bin/env python3
"""
方案二（备选）：从 Google Earth 提取 3D 模型

工具: retroplasma/earth-reverse-engineering
原理: 逆向 Google Earth 的八叉树(octree)协议，直接提取 3D Mesh + 纹理
输出: OBJ + MTL + BMP/JPG 纹理

注意：
  - 违反 Google ToS，仅建议用于学术研究
  - 不建议公开发布提取的数据
  - 该工具可能因 Google Earth 协议更新而失效

本脚本封装了 retroplasma/earth-reverse-engineering 的安装和使用流程

参考:
  - https://github.com/retroplasma/earth-reverse-engineering
  - https://deepwiki.com/retroplasma/earth-reverse-engineering
"""

import subprocess
import sys
import os
import json
import argparse
import shutil
from pathlib import Path

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ERE_DIR = PROJECT_ROOT / "tools" / "earth-reverse-engineering"
OUTPUT_DIR = PROJECT_ROOT / "output" / "google_earth"

# HKUST 坐标
HKUST_LAT = 22.3363
HKUST_LON = 114.2656

# ============================================================
# 安装
# ============================================================


def install_ere():
    """克隆并安装 earth-reverse-engineering"""
    if ERE_DIR.exists():
        print(f"✓ earth-reverse-engineering 已存在于 {ERE_DIR}")
        print("  如需重新安装，请先删除该目录")
        return True

    print("正在克隆 earth-reverse-engineering...")
    ERE_DIR.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone",
             "https://github.com/retroplasma/earth-reverse-engineering.git",
             str(ERE_DIR)],
            check=True, capture_output=True, text=True
        )
        print("✓ 克隆完成")
    except subprocess.CalledProcessError as e:
        print(f"✗ 克隆失败: {e.stderr}")
        print("\n可能原因:")
        print("  1. 网络问题（GitHub 被墙）")
        print("  2. 该仓库已被删除或设为私有")
        print(f"\n请手动下载并解压到: {ERE_DIR}")
        return False

    # 安装依赖
    try:
        print("正在安装 npm 依赖...")
        subprocess.run(
            ["npm", "install"],
            cwd=str(ERE_DIR),
            check=True, capture_output=True, text=True
        )
        print("✓ 依赖安装完成")
    except subprocess.CalledProcessError as e:
        print(f"✗ 依赖安装失败: {e.stderr}")
        return False

    return True


# ============================================================
# 提取
# ============================================================


def find_octant_path(lat: float, lon: float):
    """
    将经纬度转换为 Google Earth 的八叉树路径

    Google Earth 使用内部八叉树系统组织3D数据
    地球表面被映射到一个立方体上，然后递归细分
    """
    # 调用 retroplasma 的 lat_long_to_octant.js
    lat_long_script = ERE_DIR / "lat_long_to_octant.js"
    if not lat_long_script.exists():
        print(f"✗ 找不到脚本: {lat_long_script}")
        print("  请先运行 install 子命令安装工具")
        return None

    try:
        result = subprocess.run(
            ["node", str(lat_long_script), str(lat), str(lon)],
            cwd=str(ERE_DIR),
            capture_output=True, text=True,
            timeout=30
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"  Octant Path: {output}")
            return output
        else:
            print(f"✗ 转换失败: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("✗ 转换超时")
        return None
    except FileNotFoundError:
        print("✗ 需要安装 Node.js (https://nodejs.org)")
        return None


def extract_models(octant_path: str, output_dir: Path = None,
                   radius: float = 200.0, max_level: int = 20):
    """
    提取指定八叉树路径下的 3D 模型

    Args:
        octant_path: 八叉树路径
        output_dir: 输出目录
        radius: 提取半径（米）
        max_level: 最大八叉树深度（越大越精细，但数据量指数增长）
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "hkust_raw"

    dump_script = ERE_DIR / "dump_obj.js"
    if not dump_script.exists():
        print(f"✗ 找不到脚本: {dump_script}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "node", str(dump_script),
        "--octant-path", octant_path,
        "--output", str(output_dir),
        "--max-level", str(max_level),
    ]

    print(f"\n执行提取命令:")
    print(f"  {' '.join(cmd)}")
    print(f"\n提取参数:")
    print(f"  目标坐标: ({HKUST_LAT}, {HKUST_LON})")
    print(f"  八叉树路径: {octant_path}")
    print(f"  最大深度: {max_level}")
    print(f"  输出目录: {output_dir}")
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ERE_DIR),
            capture_output=True, text=True,
            timeout=600  # 10分钟超时
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"错误: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ 提取超时（超过10分钟），尝试降低 max_level")
        return False

    return True


def analyze_output(output_dir: Path = None):
    """分析提取结果"""
    if output_dir is None:
        output_dir = OUTPUT_DIR / "hkust_raw"

    if not output_dir.exists():
        print(f"输出目录不存在: {output_dir}")
        return

    obj_files = list(output_dir.rglob("*.obj"))
    jpg_files = list(output_dir.rglob("*.jpg"))
    bmp_files = list(output_dir.rglob("*.bmp"))
    mtl_files = list(output_dir.rglob("*.mtl"))

    print(f"\n提取结果统计:")
    print(f"  OBJ 网格文件: {len(obj_files)}")
    print(f"  MTL 材质文件: {len(mtl_files)}")
    print(f"  JPG 纹理文件: {len(jpg_files)}")
    print(f"  BMP 纹理文件: {len(bmp_files)}")

    total_obj_size = sum(f.stat().st_size for f in obj_files)
    total_tex_size = sum(f.stat().st_size for f in jpg_files + bmp_files)
    print(f"  网格总大小: {total_obj_size / 1024 / 1024:.1f} MB")
    print(f"  纹理总大小: {total_tex_size / 1024 / 1024:.1f} MB")

    return {
        "obj_count": len(obj_files),
        "texture_count": len(jpg_files) + len(bmp_files),
        "obj_size_mb": total_obj_size / 1024 / 1024,
        "texture_size_mb": total_tex_size / 1024 / 1024,
    }


# ============================================================
# OBJ 合并工具
# ============================================================


def merge_obj_files(input_dir: Path = None, output_file: Path = None):
    """
    将多个 OBJ 文件合并为一个（用于导入其他软件）

    注意：简单拼接方式，假设所有 OBJ 共享同一坐标系
    """
    if input_dir is None:
        input_dir = OUTPUT_DIR / "hkust_raw"
    if output_file is None:
        output_file = PROJECT_ROOT / "output" / "processed" / "hkust_merged.obj"

    obj_files = sorted(input_dir.rglob("*.obj"))
    if not obj_files:
        print(f"目录中没有 OBJ 文件: {input_dir}")
        return

    print(f"合并 {len(obj_files)} 个 OBJ 文件...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_vertices = 0
    total_faces = 0
    mtl_lib_added = False

    with open(output_file, "w") as out:
        out.write("# Merged OBJ from Google Earth extraction\n")
        out.write(f"# Source: {len(obj_files)} files\n")
        out.write(f"# HKUST area\n\n")

        for obj_path in obj_files:
            rel = obj_path.relative_to(input_dir)
            out.write(f"\n# --- {rel} ---\n")

            vertex_offset = total_vertices

            with open(obj_path, "r") as f:
                for line in f:
                    if line.startswith("mtllib "):
                        if not mtl_lib_added:
                            out.write(line)
                            mtl_lib_added = True
                    elif line.startswith("usemtl ") or line.startswith("o "):
                        out.write(line)
                    elif line.startswith("v "):
                        total_vertices += 1
                        out.write(line)
                    elif line.startswith("f "):
                        # 需要偏移面索引
                        parts = line.strip().split()
                        new_parts = ["f"]
                        for p in parts[1:]:
                            indices = p.split("/")
                            # 顶点索引偏移
                            if indices[0]:
                                indices[0] = str(int(indices[0]) + vertex_offset)
                            new_parts.append("/".join(indices))
                        total_faces += 1
                        out.write(" ".join(new_parts) + "\n")
                    elif line.startswith("vt ") or line.startswith("vn "):
                        out.write(line)
                    # 跳过注释和其他

    print(f"✓ 合并完成: {output_file}")
    print(f"  总顶点: {total_vertices}, 总面: {total_faces}")
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"  文件大小: {size_mb:.1f} MB")


# ============================================================
# 主程序
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Google Earth 3D 模型提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 安装工具
  python 02_google_earth_extract.py install

  # 2. 查找 HKUST 的八叉树路径
  python 02_google_earth_extract.py find-path

  # 3. 提取 3D 模型
  python 02_google_earth_extract.py extract --path <octant_path>

  # 4. 分析提取结果
  python 02_google_earth_extract.py analyze

  # 5. 合并所有 OBJ 为一个文件
  python 02_google_earth_extract.py merge
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # install
    install_parser = subparsers.add_parser("install", help="安装 earth-reverse-engineering")

    # find-path
    find_parser = subparsers.add_parser("find-path", help="查找目标坐标的八叉树路径")
    find_parser.add_argument("--lat", type=float, default=HKUST_LAT)
    find_parser.add_argument("--lon", type=float, default=HKUST_LON)

    # extract
    extract_parser = subparsers.add_parser("extract", help="提取 3D 模型")
    extract_parser.add_argument("--path", required=True, help="八叉树路径")
    extract_parser.add_argument("--output", default=None, help="输出目录")
    extract_parser.add_argument("--max-level", type=int, default=20,
                                help="最大八叉树深度 (默认: 20)")

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="分析提取结果")
    analyze_parser.add_argument("--input", default=None, help="输入目录")

    # merge
    merge_parser = subparsers.add_parser("merge", help="合并 OBJ 文件")
    merge_parser.add_argument("--input", default=None, help="输入目录")
    merge_parser.add_argument("--output", default=None, help="输出文件")

    args = parser.parse_args()

    if args.command == "install":
        install_ere()

    elif args.command == "find-path":
        print(f"查找坐标 ({args.lat}, {args.lon}) 的八叉树路径...")
        path = find_octant_path(args.lat, args.lon)
        if path:
            print(f"\n在 extract 命令中使用:")
            print(f"  python 02_google_earth_extract.py extract --path {path}")

    elif args.command == "extract":
        extract_models(args.path, args.output, max_level=args.max_level)
        analyze_output(args.output)

    elif args.command == "analyze":
        analyze_output(args.input)

    elif args.command == "merge":
        merge_obj_files(args.input, args.output)

    else:
        parser.print_help()
        print("\n⚠️  Google ToS 提醒:")
        print("  Google Earth 3D 数据受版权保护，提取行为违反 Google 服务条款。")
        print("  本工具仅供学术研究和学习目的，不建议公开发布提取的数据。")
        print("  强烈建议优先使用方案一（HK CSDI 合法数据）。")


if __name__ == "__main__":
    main()

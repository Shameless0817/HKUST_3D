#!/usr/bin/env python3
"""
后处理脚本：OBJ → GLB / b3dm → GLB / 格式转换 & 模型清理

支持的转换:
  - OBJ (+MTL+纹理) → GLB (glTF Binary)
  - FBX → GLB
  - B3DM → GLB
  - 合并多个模型文件
  - 减面 (simplification) 用于 Web 展示优化

依赖:
  pip install trimesh pygltflib numpy open3d

或使用 Blender 无头模式:
  blender --background --python 04_convert.py -- --input input.obj --output output.glb
"""

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "processed"


# ============================================================
# 方法1: Python 原生转换（trimesh + pygltflib）
# ============================================================


def convert_obj_to_glb_trimesh(obj_path: Path, glb_path: Path,
                                simplify_ratio: float = 0.0):
    """
    使用 trimesh 将 OBJ 转为 GLB

    Args:
        obj_path: OBJ 文件路径
        glb_path: 输出 GLB 路径
        simplify_ratio: 减面比例 (0=不减, 0.5=减50%)
    """
    try:
        import trimesh
    except ImportError:
        print("✗ 需要 trimesh: pip install trimesh")
        return False

    print(f"加载: {obj_path}")
    try:
        # 加载 OBJ (包含材质和纹理)
        scene = trimesh.load(str(obj_path), force="scene")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False

    if isinstance(scene, trimesh.Trimesh):
        # 单个 mesh
        mesh = scene
    elif isinstance(scene, trimesh.Scene):
        # 场景 → 合并所有 mesh
        meshes = []
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if not meshes:
            print("✗ 场景中无有效mesh")
            return False
        mesh = trimesh.util.concatenate(meshes)
    else:
        print(f"✗ 不支持的类型: {type(scene)}")
        return False

    # 减面
    if simplify_ratio > 0:
        from trimesh.simplify import simplify_quadratic_decimation
        original_faces = len(mesh.faces)
        target_faces = int(original_faces * (1 - simplify_ratio))
        mesh = mesh.simplify_quadratic_decimation(target_faces)
        print(f"  减面: {original_faces} → {len(mesh.faces)} faces")

    # 导出 GLB
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(glb_path), file_type="glb")
    size_kb = glb_path.stat().st_size / 1024
    print(f"✓ 已保存: {glb_path} ({size_kb:.0f} KB)")
    return True


# ============================================================
# 方法2: Blender 无头模式转换
# ============================================================


def convert_with_blender(input_path: Path, output_path: Path,
                          format: str = "GLB"):
    """
    使用 Blender 无头模式进行转换

    优点: 支持几乎所有3D格式，材质兼容性好
    需要: 安装 Blender (sudo apt install blender)
    """
    import subprocess

    blender_script = PROJECT_ROOT / "scripts" / "_blender_convert.py"
    # 写入临时脚本
    script_content = f'''
import bpy
import sys

# 清除默认场景
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 导入
ext = "{input_path.suffix.lower()}"
if ext == ".obj":
    bpy.ops.wm.obj_import(filepath="{input_path}")
elif ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath="{input_path}")
elif ext in [".glb", ".gltf"]:
    bpy.ops.import_scene.gltf(filepath="{input_path}")
elif ext == ".stl":
    bpy.ops.import_mesh.stl(filepath="{input_path}")
else:
    print(f"Unsupported format: {{ext}}")
    sys.exit(1)

print(f"Imported: {input_path}")

# 导出 GLB
bpy.ops.export_scene.gltf(
    filepath="{output_path}",
    export_format='GLB',
    export_apply=True,
    export_image_format='JPEG',
    export_jpeg_quality=90,
)
print(f"Exported: {output_path}")
'''
    blender_script.write_text(script_content)

    cmd = [
        "blender", "--background", "--factory-startup",
        "--python", str(blender_script),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Blender 错误: {result.stderr}")
            return False
        print(result.stdout)
        return True
    except FileNotFoundError:
        print("✗ Blender 未安装: sudo apt install blender")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Blender 转换超时")
        return False


# ============================================================
# B3DM → GLB
# ============================================================


def convert_b3dm_to_glb(b3dm_path: Path, output_path: Path) -> bool:
    """B3DM 格式 → GLB (纯二进制操作)"""
    data = b3dm_path.read_bytes()
    if len(data) < 28 or data[:4] != b"b3dm":
        print(f"✗ 无效的 B3DM 文件: {b3dm_path.name}")
        return False

    # 解析 header
    header = {
        "featureTableJSONByteLength": struct.unpack("<I", data[12:16])[0],
        "featureTableBinaryByteLength": struct.unpack("<I", data[16:20])[0],
        "batchTableJSONByteLength": struct.unpack("<I", data[20:24])[0],
        "batchTableBinaryByteLength": struct.unpack("<I", data[24:28])[0],
    }

    glb_offset = 28 + header["featureTableJSONByteLength"] \
        + header["featureTableBinaryByteLength"] \
        + header["batchTableJSONByteLength"] \
        + header["batchTableBinaryByteLength"]

    glb_data = data[glb_offset:]
    if glb_data[:4] != b"glTF":
        print(f"⚠️  {b3dm_path.name} 提取的数据不是标准 GLB")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_data)
    return True


def batch_convert_b3dm(input_dir: Path, output_dir: Path = None):
    """批量转换目录下所有 B3DM 文件"""
    if output_dir is None:
        output_dir = input_dir / "glb"

    b3dm_files = list(input_dir.rglob("*.b3dm"))
    if not b3dm_files:
        print(f"未找到 .b3dm 文件: {input_dir}")
        return

    print(f"找到 {len(b3dm_files)} 个 B3DM 文件")
    success = 0
    for f in b3dm_files:
        rel = f.relative_to(input_dir)
        out = output_dir / rel.with_suffix(".glb")
        if convert_b3dm_to_glb(f, out):
            success += 1

    print(f"✓ 转换完成: {success}/{len(b3dm_files)} → {output_dir}")


# ============================================================
# 模型分析
# ============================================================


def analyze_model(file_path: Path) -> Optional[dict]:
    """分析3D模型的统计信息"""
    try:
        import trimesh
    except ImportError:
        print("✗ 需要 trimesh: pip install trimesh")
        return None

    try:
        mesh = trimesh.load(str(file_path), force="mesh")
    except Exception as e:
        print(f"✗ 无法加载: {file_path} - {e}")
        return None

    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        if not meshes:
            return None
        mesh = trimesh.util.concatenate(meshes)

    info = {
        "file": str(file_path),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "size_mb": file_path.stat().st_size / 1024 / 1024,
        "bounds": {
            "min": mesh.vertices.min(axis=0).tolist(),
            "max": mesh.vertices.max(axis=0).tolist(),
        },
        "has_texture": hasattr(mesh.visual, "material") and mesh.visual.material is not None,
        "is_watertight": mesh.is_watertight if hasattr(mesh, "is_watertight") else "unknown",
    }

    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    info["extent"] = extent.tolist()

    return info


def print_analysis(info: dict):
    """美化打印模型分析结果"""
    print(f"\n文件: {info['file']}")
    print(f"  顶点数: {info['vertices']:,}")
    print(f"  面数:   {info['faces']:,}")
    print(f"  大小:   {info['size_mb']:.1f} MB")
    print(f"  范围:   X: {info['extent'][0]:.1f}m  "
          f"Y: {info['extent'][1]:.1f}m  Z: {info['extent'][2]:.1f}m")
    print(f"  纹理:   {'✓' if info['has_texture'] else '✗'}")
    if info["is_watertight"] != "unknown":
        print(f"  水密:   {'✓' if info['is_watertight'] else '✗'}")


# ============================================================
# 模型优化（Web 展示用）
# ============================================================


def optimize_for_web(input_path: Path, output_path: Path,
                     target_faces: int = 500000):
    """
    优化模型用于 Web 展示

    - 减面至目标面数
    - 纹理压缩 (JPEG)
    - 坐标系规整
    """
    try:
        import trimesh
        import numpy as np
    except ImportError:
        print("✗ 需要 trimesh numpy: pip install trimesh numpy")
        return False

    print(f"加载: {input_path}")
    mesh = trimesh.load(str(input_path), force="mesh")

    if isinstance(mesh, trimesh.Scene):
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)

    original_faces = len(mesh.faces)
    print(f"  原始面数: {original_faces:,}")

    # 减面
    if original_faces > target_faces:
        mesh = mesh.simplify_quadratic_decimation(target_faces)
        print(f"  减面后: {len(mesh.faces):,}")

    # 移动到原点
    center = mesh.vertices.mean(axis=0)
    mesh.vertices -= center
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()  # Z轴归零

    # 缩放到合理尺度 (最大维度 = 100)
    max_extent = (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max()
    if max_extent > 0:
        scale = 100.0 / max_extent
        mesh.vertices *= scale
    print(f"  缩放: {scale:.4f}x, 高度范围: "
          f"{mesh.vertices[:, 2].min():.1f} - {mesh.vertices[:, 2].max():.1f}")

    # 导出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path), file_type="glb")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ 已保存: {output_path} ({size_mb:.1f} MB)")
    return True


# ============================================================
# 主程序
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="3D 模型格式转换 & 优化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # OBJ → GLB
  python 04_convert.py convert --input model.obj --output model.glb

  # 批量 B3DM → GLB
  python 04_convert.py batch-b3dm --input output/csdi/f2/

  # 用 Blender 转换
  python 04_convert.py blender --input model.obj --output model.glb

  # 分析模型
  python 04_convert.py analyze --input model.obj

  # Web 优化
  python 04_convert.py optimize --input model.obj --output model_web.glb --target-faces 200000
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # convert
    cvt_parser = subparsers.add_parser("convert", help="OBJ → GLB (trimesh)")
    cvt_parser.add_argument("--input", required=True)
    cvt_parser.add_argument("--output", required=True)
    cvt_parser.add_argument("--simplify", type=float, default=0.0,
                            help="减面比例 (0-1)")

    # batch-b3dm
    batch_parser = subparsers.add_parser("batch-b3dm", help="批量 B3DM → GLB")
    batch_parser.add_argument("--input", required=True)
    batch_parser.add_argument("--output", default=None)

    # blender
    blender_parser = subparsers.add_parser("blender", help="Blender 无头模式转换")
    blender_parser.add_argument("--input", required=True)
    blender_parser.add_argument("--output", required=True)

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="分析模型")
    analyze_parser.add_argument("--input", required=True)

    # optimize
    opt_parser = subparsers.add_parser("optimize", help="Web优化")
    opt_parser.add_argument("--input", required=True)
    opt_parser.add_argument("--output", required=True)
    opt_parser.add_argument("--target-faces", type=int, default=500000)

    args = parser.parse_args()

    if args.command == "convert":
        convert_obj_to_glb_trimesh(
            Path(args.input), Path(args.output), args.simplify
        )

    elif args.command == "batch-b3dm":
        batch_convert_b3dm(
            Path(args.input),
            Path(args.output) if args.output else None,
        )

    elif args.command == "blender":
        convert_with_blender(Path(args.input), Path(args.output))

    elif args.command == "analyze":
        info = analyze_model(Path(args.input))
        if info:
            print_analysis(info)

    elif args.command == "optimize":
        optimize_for_web(
            Path(args.input), Path(args.output), args.target_faces
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

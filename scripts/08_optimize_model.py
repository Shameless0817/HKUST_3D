#!/usr/bin/env python3
"""
Google Earth 模型优化 — 减面 + 清理 + 纹理压缩

输入: output/google_earth/*/model.obj
输出:
  - output/demo/hkust_optimized.glb    (50K面, ~20MB)
  - output/demo/hkust_light.glb        (20K面, ~8MB)
"""

import sys, math, json, subprocess
from pathlib import Path
import numpy as np

PROJECT = Path("/home/zliki/HKUST_3D")
SRC_DIR = PROJECT / "output/google_earth/31416263636040-19-1012"
OUT_DIR = PROJECT / "output/demo"

# ============================================================
def load_and_merge(obj_path):
    """加载 OBJ 并合并为单个 mesh"""
    import trimesh
    print(f"加载: {obj_path}")
    scene = trimesh.load(str(obj_path), force="scene")

    meshes = []
    total_tex = 0
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            if len(geom.faces) > 0:
                meshes.append(geom)
                if hasattr(geom.visual, 'material') and geom.visual.material:
                    total_tex += 1

    print(f"  Sub-meshes: {len(meshes)} (有纹理: {total_tex})")
    if not meshes:
        return None

    mesh = trimesh.util.concatenate(meshes)
    print(f"  合并后: {len(mesh.vertices):,}v, {len(mesh.faces):,}f")
    return mesh


def clean_mesh(mesh):
    """清理：去重、删退化面、删碎片"""
    import trimesh
    orig_v, orig_f = len(mesh.vertices), len(mesh.faces)

    # 合并重复顶点
    mesh.merge_vertices()
    # 删退化面
    mesh.update_faces(mesh.nondegenerate_faces())
    # 删孤立顶点
    mesh.remove_unreferenced_vertices()

    # 删微小碎片（孤立的小连通分量）
    if len(mesh.faces) > 1000:
        try:
            components = trimesh.graph.connected_components(
                mesh.face_adjacency, min_len=50
            )
            if len(components) > 1:
                keep = np.zeros(len(mesh.faces), dtype=bool)
                for c in components:
                    keep[c] = True
                mesh.update_faces(keep)
                mesh.remove_unreferenced_vertices()
                print(f"  删除 {len(components)} 个小碎片后: "
                      f"{len(mesh.vertices):,}v, {len(mesh.faces):,}f")
        except Exception:
            pass  # face_adjacency 可能为 None

    v_removed = orig_v - len(mesh.vertices)
    f_removed = orig_f - len(mesh.faces)
    print(f"  清理: -{v_removed:,}v, -{f_removed:,}f")
    return mesh


def center_and_orient(mesh):
    """居中 + 旋转使北在上 (Y-up: Z=北, X=东)"""
    # 移到质心
    center = mesh.vertices.mean(axis=0)
    mesh.vertices -= center

    # 缩放到合理大小
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    scale = 200.0 / max(extent)
    mesh.vertices *= scale

    print(f"  居中+缩放: 尺寸={extent[0]*scale:.0f}x{extent[1]*scale:.0f}x{extent[2]*scale:.0f}")
    return mesh


def decimate_with_textures(mesh, target_faces):
    """
    减面到目标面数，同时保留纹理信息（烘焙为顶点颜色）

    策略：
    1. 从 UV 纹理采样 → 顶点颜色
    2. pyfqmr 减面（自动保留顶点颜色）
    3. 输出无纹理但带顶点颜色的 mesh（GLB 体积大幅缩小）
    """
    import trimesh
    from PIL import Image

    orig_f = len(mesh.faces)
    if orig_f <= target_faces:
        return mesh

    print(f"  烘焙纹理 → 顶点颜色...")

    # Step 1: 如果模型有纹理，烘焙为顶点颜色
    if hasattr(mesh.visual, 'material') or (
       hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None
    ):
        try:
            # 尝试用 trimesh 的 texture_to_vertex_color
            if hasattr(mesh.visual, 'to_color'):
                mesh.visual = mesh.visual.to_color()
                print(f"    ✓ 纹理→顶点颜色")
        except Exception as e:
            print(f"    纹理烘焙失败: {e}, 使用默认颜色")
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile([0.7, 0.65, 0.55, 1.0], (len(mesh.vertices), 1))
            )

    # Step 2: 如果顶点颜色不是 ColorVisuals，转换
    if not isinstance(mesh.visual, trimesh.visual.ColorVisuals):
        try:
            if hasattr(mesh.visual, 'vertex_colors'):
                colors = mesh.visual.vertex_colors
                if colors is not None:
                    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
        except Exception:
            pass

    # 确保有顶点颜色
    if not isinstance(mesh.visual, trimesh.visual.ColorVisuals):
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=np.tile([0.7, 0.65, 0.55, 1.0], (len(mesh.vertices), 1))
        )

    colors = mesh.visual.vertex_colors

    # Step 3: pyfqmr 减面
    reduction = 1.0 - (target_faces / orig_f)
    print(f"  减面: {orig_f:,} → ~{target_faces:,} faces (缩减 {reduction:.0%})...")

    import pyfqmr
    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(mesh.vertices, mesh.faces)
    # 使用最高 aggressiveness 以达到目标
    simplifier.simplify_mesh(
        target_count=target_faces,
        aggressiveness=10,
        preserve_border=False,
        verbose=False
    )
    new_verts, new_faces, _ = simplifier.getMesh()

    # Step 4: 顶点颜色映射
    # pyfqmr 返回的顶点包含原始顶点 + 新顶点，需重建颜色
    # 用最近邻方式从原始顶点映射颜色
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)
    _, indices = tree.query(new_verts, k=1)
    new_colors = colors[indices]

    new_mesh = trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        vertex_colors=new_colors
    )

    print(f"    ✓ {len(new_mesh.faces):,} faces, 顶点颜色已保留")
    return new_mesh


def decimate(mesh, target_faces):
    """减面入口：优先使用纹理烘焙+pyfqmr"""
    return decimate_with_textures(mesh, target_faces)


def export_glb(mesh, path, label=""):
    """导出 GLB"""
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="glb")
    mb = path.stat().st_size / 1024 / 1024
    print(f"  ✓ {label}: {path.name} ({mb:.1f} MB)")
    return mb


# ============================================================
def main():
    print("=" * 60)
    print("  Google Earth 模型优化")
    print("=" * 60)

    obj_path = SRC_DIR / "model.obj"
    if not obj_path.exists():
        print(f"✗ 模型文件不存在: {obj_path}")
        sys.exit(1)

    # Step 1: 加载+合并
    print("\n>>> Step 1: 加载模型")
    mesh = load_and_merge(obj_path)
    if mesh is None:
        sys.exit(1)

    # Step 2: 清理
    print("\n>>> Step 2: 清理碎片 & 去重")
    mesh = clean_mesh(mesh)

    # Step 3: 居中 & 缩放
    print("\n>>> Step 3: 居中 & 缩放")
    mesh = center_and_orient(mesh)

    # ---- 导出版本 ----

    # 版本A: 全精度 (不减面，仅清理)
    print("\n>>> 导出: 全精度版")
    full_path = OUT_DIR / "hkust_full.glb"
    export_glb(mesh, full_path, "全精度")

    # 版本B: 中等优化 (50K 面)
    print("\n>>> 导出: 优化版 (50K faces)")
    mesh_opt = decimate(mesh.copy(), 50000)
    opt_path = OUT_DIR / "hkust_optimized.glb"
    export_glb(mesh_opt, opt_path, "优化版")

    # 版本C: 轻量版 (20K 面)
    print("\n>>> 导出: 轻量版 (20K faces)")
    mesh_light = decimate(mesh.copy(), 20000)
    light_path = OUT_DIR / "hkust_light.glb"
    export_glb(mesh_light, light_path, "轻量版")

    print(f"\n{'=' * 60}")
    print(f"  完成！")
    print(f"")
    print(f"  全精度:  {full_path}")
    print(f"  优化版:  {opt_path}  ← 推荐日常使用")
    print(f"  轻量版:  {light_path}  ← Web/移动端")
    print(f"")
    print(f"  查看: https://gltf-viewer.donmccurdy.com/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

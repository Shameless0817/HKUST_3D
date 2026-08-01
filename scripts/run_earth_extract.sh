#!/bin/bash
# ============================================================
# 方案二：Google Earth 3D 模型提取
# 一键运行脚本
# ============================================================

set -e

NODE_BIN="/tmp/node-v20.18.0-linux-x64/bin"
export PATH="$NODE_BIN:$PATH"

PROJECT_DIR="/home/zliki/HKUST_3D"
ERE_DIR="$PROJECT_DIR/tools/earth-reverse-engineering/exporter"
OUTPUT_DIR="$PROJECT_DIR/output/google_earth"

# HKUST octant paths (从 lat_long_to_octant.js 获得)
# Level 14: 覆盖 ~1.2km 范围
OCTANT_L14="31416263636040"
# Level 16: 覆盖 ~600m 范围 (核心校园)
OCTANT_L16="3141626363604061"

echo "============================================================"
echo "  HKUST Google Earth 3D 模型提取"
echo "============================================================"
echo ""

# Step 1: 确认 Node.js
if ! command -v node &> /dev/null; then
    echo "安装 Node.js..."
    curl -fsSL https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz -o /tmp/node.tar.xz
    tar -xf /tmp/node.tar.xz -C /tmp/
    export PATH="/tmp/node-v20.18.0-linux-x64/bin:$PATH"
fi
echo "Node.js: $(node --version)"
echo ""

# Step 2: 进入 exporter 目录
cd "$ERE_DIR"

# Step 3: 确认依赖已安装
if [ ! -d "node_modules" ]; then
    echo "安装 npm 依赖..."
    npm install
fi

# Step 4: 提取 3D 模型
echo "============================================================"
echo "提取 HKUST 3D 模型..."
echo "  Octant (Level 14): $OCTANT_L14"
echo "  Max Level: 19"
echo "  预计时间: 2-5 分钟"
echo "============================================================"
echo ""

node dump_obj.js "$OCTANT_L14" 19 --parallel-search

# Step 5: 分析结果
echo ""
echo "============================================================"
echo "提取完成！分析结果..."
echo "============================================================"

DL_DIR="$ERE_DIR/downloaded_files"

if [ -d "$DL_DIR/obj" ]; then
    OBJ_COUNT=$(find "$DL_DIR/obj" -name "*.obj" | wc -l)
    JPG_COUNT=$(find "$DL_DIR/obj" -name "*.jpg" -o -name "*.bmp" | wc -l)
    TOTAL_SIZE=$(du -sh "$DL_DIR/obj" 2>/dev/null | cut -f1)

    echo "  OBJ 文件: $OBJ_COUNT"
    echo "  纹理文件: $JPG_COUNT"
    echo "  总大小:   $TOTAL_SIZE"
    echo ""
    echo "输出目录: $DL_DIR/obj"
fi

# Step 6: 复制到项目 output 目录
mkdir -p "$OUTPUT_DIR"
if [ -d "$DL_DIR/obj" ]; then
    LATEST=$(ls -dt "$DL_DIR/obj"/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo ""
        echo "复制到项目输出目录..."
        cp -r "$LATEST"/* "$OUTPUT_DIR/" 2>/dev/null || true
        echo "✓ 已复制到: $OUTPUT_DIR"
    fi
fi

echo ""
echo "============================================================"
echo "✓ 完成！"
echo ""
echo "查看模型:"
echo "  1. 安装 Blender: sudo apt install blender"
echo "  2. 打开 Blender → File → Import → Wavefront (.obj)"
echo "  3. 选择 $DL_DIR/obj 中的 .obj 文件"
echo "  4. 或转换为 GLB: python scripts/04_convert.py"
echo "============================================================"

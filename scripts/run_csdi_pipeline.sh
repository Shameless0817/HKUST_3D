#!/bin/bash
# ============================================================
# CSDI 完整流水线 — 一键下载 HKUST 照片级 3D 模型
# API Key: 56c56a5bed7f400ebc55db7b2d8d839d
# ============================================================
set -e
cd /home/zliki/HKUST_3D

echo "============================================================"
echo "  HKUST CSDI 3D 模型获取流水线 v3"
echo "  API Key: 56c56a5b..."
echo "  $(date)"
echo "============================================================"
echo ""

# Run the Python pipeline
python3 scripts/10_csdi_pipeline.py

echo ""
echo "============================================================"
echo "  流水线完成！"
echo ""
echo "  输出文件:"
echo "    output/csdi/f2/hkust_csdi_merged.glb   (合并模型)"
echo "    output/demo/hkust_csdi_merged.glb       (副本)"
echo "    output/csdi/f2/glb/                     (各 tile 的 GLB)"
echo "    output/csdi/f2/b3dm/                    (原始 B3DM)"
echo ""
echo "  在线预览: https://gltf-viewer.donmccurdy.com/"
echo "============================================================"

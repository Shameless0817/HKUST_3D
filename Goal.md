# HKUST 三维重建 — Google Earth & 在线平台导出方案

## Context
用户是计算机视觉专家，希望获取香港科技大学（HKUST）校园的3D模型，
采用纯在线数据导出方式（不涉及无人机、不亲自拍摄）。
目标是从 Google Earth 等在线平台提取/下载现有3D数据用于研究和可视化。

**关键发现（2025年）**：香港地政总署（LandsD）已于2025年3月完成**全港三维数码地图**，
覆盖~122,000+栋建筑，提供纹理3D模型，**完全免费下载**，格式包括 OBJ、Cesium 3D Tiles、OSGB。
这是获取 HKUST 3D 模型的最佳途径。

---

## 🏆 方案排序（按推荐度）

| 优先级 | 方案 | 质量 | 难度 | 合法性 |
|--------|------|------|------|--------|
| **🥇 首选** | 香港 CSDI 三维可视化地图 OBJ 下载 | ⭐⭐⭐⭐⭐ | 低 | ✅ 完全合法 |
| 🥈 备选 | Google Earth 逆向工程导出 OBJ | ⭐⭐⭐⭐ | 中 | ⚠️ ToS灰色地带 |
| 🥉 补充 | Google 3D Tiles 流式提取 | ⭐⭐⭐⭐ | 高 | ⚠️ ToS灰色地带 |
| 4 | Mapscaping 建筑轮廓+高度 | ⭐⭐ | 极低 | ✅ 合法 |
| 5 | VoxCity Python 程序化城市模型 | ⭐⭐ | 低 | ✅ 合法 |

---

## 🥇 方案一：香港 CSDI 三维可视化地图（强烈推荐）

### 数据源信息

- **提供方**：香港地政总署（LandsD）Survey and Mapping Office
- **完成时间**：2025年3月（全港覆盖）
- **数据内容**：~122,000+栋建筑 + ~3,300+基建（天桥、隧道、行人桥等）
- **数据来源**：倾斜航空摄影（oblique aerial imagery）→ Mesh模型 + 纹理贴图
- **覆盖范围**：香港全境，包括清水湾/HKUST区域
- **费用**：完全免费（商用和非商用均可，需注明来源）

### 可用数据产品

| 产品名称 | 格式 | 纹理 | API |
|----------|------|------|-----|
| **3D Visualisation Map (Tile-based models)** | **OBJ**, Cesium 3D Tiles, OSGB | ✅ 有纹理 | 有 |
| 3D Visualisation Map (Non-textured models) | FBX, Cesium 3D Tiles | ❌ 无纹理 | 有 |
| 3D Spatial Data (Buildings) | Cesium 3D Tiles | - | 有 |
| 3D Spatial Data (Infrastructure) | Cesium 3D Tiles | - | 有 |

### 操作步骤

#### Step 1：获取 API Key

```
发送邮件至：3dmap@landsd.gov.hk
（GIS Projects Section, Survey and Mapping Office, LandsD）
说明用途（如学术研究、三维重建实验），请求发放 API Key
```

#### Step 2：通过 API 下载 OBJ 格式模型

**Tile-based 3D Visualisation Map API：**
```
端点：https://data.map.gov.hk/api/3d-data/3dtiles/f2/tileset.json?key=YOUR_KEY
```

**用 Python 批量下载特定区域（HKUST）：**

```python
"""
通过 HK CSDI API 下载 HKUST 区域的 3D 模型
HKUST 坐标范围：lat 22.330-22.340, lon 114.260-114.270
"""

import requests
import json
import os
from pathlib import Path

API_KEY = "YOUR_API_KEY_FROM_LANDSD"  # 从 LandsD 获取
BASE_URL = "https://data.map.gov.hk/api/3d-data"
OUTPUT_DIR = Path("./hkust_3d_models")
OUTPUT_DIR.mkdir(exist_ok=True)

# === 方法1：通过 3D Tiles API 下载 tileset ===
def download_3dtiles_tileset(tileset_name, output_dir):
    """下载 Cesium 3D Tiles 格式数据"""
    url = f"{BASE_URL}/3dtiles/{tileset_name}/tileset.json"
    resp = requests.get(url, params={"key": API_KEY})
    resp.raise_for_status()
    tileset = resp.json()

    # 保存 tileset.json
    with open(output_dir / "tileset.json", "w") as f:
        json.dump(tileset, f, indent=2)

    # 递归下载所有子tile
    def download_tiles(node):
        if "content" in node:
            tile_url = node["content"]["uri"]
            tile_path = output_dir / tile_url
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(f"{BASE_URL}/3dtiles/{tileset_name}/{tile_url}",
                           params={"key": API_KEY})
            if r.status_code == 200:
                tile_path.write_bytes(r.content)
                print(f"  ✓ {tile_url}")
        for child in node.get("children", []):
            download_tiles(child)

    download_tiles(tileset["root"])
    print(f"Tileset 下载完成: {output_dir}")

# === 方法2：通过 Download API 按1:1000地图编号下载 OBJ ===
def download_by_map_sheet(sheet_number, output_dir):
    """
    按1:1000地形图编号下载 OBJ 文件
    HKUST 区域涉及的地图编号需在 LandsD 地图索引中查询
    """
    # 联系 LandsD 获取具体的地图编号索引
    # 或通过 Open3Dhk 网页版 (https://3d.map.gov.hk)
    # 定位到 HKUST 后查看数据可用的地图编号
    pass

download_3dtiles_tileset("f2", OUTPUT_DIR / "3dtiles")
```

#### Step 3：通过 Open3Dhk 网页平台交互式浏览

1. 打开 **[https://3d.map.gov.hk](https://3d.map.gov.hk)**（Open3Dhk 平台）
2. 搜索 "Hong Kong University of Science and Technology" 或 "香港科技大学"
3. 在网页端交互式查看 HKUST 的3D模型（确认数据质量）
4. 查看是否有导出功能（部分区域可能支持直接下载）

#### Step 4：通过 DATA.GOV.HK 下载完整数据集

1. 访问 [DATA.GOV.HK - 3D Visualisation Map (Tile-based models)](https://data.gov.hk/en-data/dataset/hk-landsd-openmap-3d-visualisation-map-tile-based-models)
2. 查看 "Resources" 部分 → 找到 OBJ 格式的数据下载链接
3. 按 1:1000 地图编号（Map Sheet Number）分批下载

#### Step 5：OBJ 转其他格式 & 导入 Blender/UE/Web

```bash
# OBJ → GLTF (用于Web展示)
pip install obj2gltf
obj2gltf -i hkust_building.obj -o hkust_building.glb

# 或用 Blender 批量转换
blender --background --python convert_obj_to_fbx.py

# Cesium 3D Tiles → GLB 提取
pip install 3d-tiles-tools
npx 3d-tiles-tools b3dmToGlb -i tile.b3dm -o tile.glb
```

---

## 🥈 方案二：Google Earth 逆向工程提取

### 工具：retroplasma/earth-reverse-engineering

- **GitHub**: https://github.com/retroplasma/earth-reverse-engineering
- **原理**：逆向 Google Earth 的内部八叉树（octree）协议，直接提取 3D Mesh + 纹理
- **输出格式**：OBJ + MTL + BMP/JPG纹理
- **法律状态**：违反 Google ToS（仅建议用于学术研究，不建议公开发布数据）

### 操作步骤

```bash
# 1. 克隆仓库
git clone https://github.com/retroplasma/earth-reverse-engineering.git
cd earth-reverse-engineering

# 2. 安装依赖（Node.js）
npm install

# 3. 找到 HKUST 的 Octant Path
# HKUST 坐标: 22.3363°N, 114.2656°E
node lat_long_to_octant.js 22.3363 114.2656

# 4. 提取 3D 模型
# 使用返回的 octant path 参数
node dump_obj.js --octant-path="<从步骤3获取的路径>" --output="./hkust_earth"

# 输出目录结构：
# hkust_earth/
#   ├── *.obj       (3D网格)
#   ├── *.mtl       (材质定义)
#   └── *.bmp/*.jpg (纹理贴图)
```

### 注意事项
- 该工具可能因 Google Earth 协议更新而失效（需确认2025年是否仍可用）
- 提取的模型质量：几何精度~1-2m，纹理分辨率为Google Earth渲染纹理
- 建筑物屋顶质量好，但建筑立面可能有拉伸/变形
- 强烈建议仅用于学术研究，不公开发布提取的数据

---

## 🥉 方案三：Google Photorealistic 3D Tiles 提取

### 背景
Google 通过 **Map Tiles API** 提供 Photorealistic 3D Tiles，与 Cesium ion 合作分发。
Google Map Tiles API 的 3D Tiles 根端点：

```
https://tile.googleapis.com/v1/3dtiles/root.json?key=YOUR_GOOGLE_API_KEY
```

### 限制
- Google 官方只允许**实时流式渲染**，**禁止缓存/存储/导出tile数据**
- 需要一个 Google Cloud 项目并启用 Map Tiles API（需绑定信用卡，但有免费额度）
- 每个 root tileset 请求有效 ~3小时

### 技术路线（仅供研究参考）

```python
"""
研究性代码 — 遍历 Google 3D Tiles 并提取 GLB
注意：违反 Google Maps ToS，仅用于学术理解
"""
import requests
import json

GOOGLE_API_KEY = "YOUR_KEY"
ROOT_URL = f"https://tile.googleapis.com/v1/3dtiles/root.json?key={GOOGLE_API_KEY}"

def traverse_tileset(url):
    resp = requests.get(url)
    tileset = resp.json()

    for child in tileset.get("root", {}).get("children", []):
        content_url = child.get("content", {}).get("uri", "")
        if content_url:
            # 每个 tile 是 .b3dm (Batched 3D Model) 或 .glb
            tile_data = requests.get(content_url)
            # b3dm = 28字节header + GLB body
            if content_url.endswith(".b3dm"):
                glb_data = tile_data.content[28:]  # 提取 GLB
                # 保存为 .glb 文件
```

---

## 4️⃣ 方案四：Mapscaping 3D Building Extractor（快速预览）

- **网址**: https://mapscaping.com/3d-building-extractor-for-google-earth
- **数据来源**: Overture Maps / OpenStreetMap 建筑脚印
- **输出格式**: 3D KML（带高度属性的建筑轮廓块）
- **局限性**: 仅有简单的立方体块模型（建筑脚印+估算高度），**无纹理**

适合快速获得粗略的体量模型，不适合精细重建。

---

## 5️⃣ 方案五：VoxCity Python 包

```bash
pip install voxcity

# 程序化生成城市3D模型
python -m voxcity download \
    --bbox "114.26,22.33,114.27,22.34" \
    --source overture \
    --format obj \
    --output hkust_voxcity
```

同样只有建筑轮廓+高度的简单块模型。

---

## 📋 推荐执行路线

```
【Step 1 立即执行】联系 LandsD 获取 API Key
    ↓
【Step 2 同步进行】在 Open3Dhk 网页版确认 HKUST 数据覆盖和质量
    ↓
【Step 3 主路线】通过 CSDI API 或 DATA.GOV.HK 下载 OBJ 格式纹理模型
    ↓
【Step 4 备选】如 CSDI 数据不满足需求，用 earth-reverse-engineering 提取 Google Earth 数据做补充
    ↓
【Step 5 后处理】导入 Blender → 清理/修整 → 导出 GLB 用于 Web/UE/研究
```

### 预期数据质量

HK CSDI 三维地图的数据来自**倾斜航空摄影测量**（oblique aerial photogrammetry），
质量预期如下：

| 指标 | 预期 |
|------|------|
| 几何精度 | ~0.3–1m（专业航空摄影） |
| 纹理质量 | 照片级真实纹理，分辨率取决于飞行高度和照片重叠度 |
| 覆盖完整度 | HKUST 所有建筑+基建均覆盖 |
| 模型类型 | Mesh（三角网格）+ UV纹理 |
| 文件格式 | OBJ/MTL + JPG纹理 / Cesium 3D Tiles |

---

## Verification

验证步骤：
1. ✅ 成功从 LandsD 获取 API Key
2. ✅ 在 Open3Dhk (3d.map.gov.hk) 中可浏览 HKUST 区域的3D模型
3. ✅ 通过 API/下载链接成功获取 OBJ 格式的 HKUST 建筑模型
4. ✅ 在 Blender 中正确导入并显示带有纹理的模型
5. ✅ （可选）转换为 GLB 格式并在浏览器中使用 Three.js/Cesium 渲染

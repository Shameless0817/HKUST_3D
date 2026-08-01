# LandD API Key 申请邮件模板

以下为中英文模板，发送至 **3dmap@landsd.gov.hk**

---

## English Template

```
Subject: Request for API Key — 3D Visualisation Map

Dear GIS Projects Section,

I am a researcher at [Your Institution/University - e.g., HKUST / independent researcher].

I am writing to request an API Key for accessing the 3D Visualisation Map
datasets available through the CSDI Portal (https://portal.csdi.gov.hk).

Research Purpose:
- 3D reconstruction and visualization research
- Target area: Hong Kong University of Science and Technology (HKUST) campus
- Intended use: Academic research on 3D urban modeling and computer graphics

I would like to access the following datasets:
1. 3D Visualisation Map (Tile-based models) — OBJ / Cesium 3D Tiles format
2. 3D Visualisation Map (Non-textured models)
3. 3D Spatial Data (Buildings & Infrastructure)

I understand and agree to comply with the data usage terms, including proper
attribution to the Government of the Hong Kong SAR and the CSDI Portal.

Thank you for your assistance.

Best regards,
[Your Name]
[Your Affiliation]
[Contact Email]
```

---

## 中文模板

```
主题：申请 3D Visualisation Map API Key

地政总署测绘处同仁：

本人是[香港科技大学/独立研究者]的研究人员，现希望申请 CSDI 平台
三维可视化地图（3D Visualisation Map）的 API Key。

研究用途：
- 三维重建与可视化研究
- 目标区域：香港科技大学校园
- 用途：城市三维建模与计算机图形学学术研究

希望获取以下数据集：
1. 三维可视化地图（有纹理模型）— OBJ / Cesium 3D Tiles 格式
2. 三维可视化地图（无纹理模型）
3. 三维空间数据（建筑物与基建）

本人已知悉并同意遵守数据使用条款，包括注明数据来源于
香港特别行政区政府及 CSDI 平台。

感谢您的协助。

此致
[姓名]
[机构]
[联系电邮]
```

---

## 补充说明

1. **API Key 发放时间**：通常 1-3 个工作日
2. **需要提供的信息**：
   - 姓名和机构信息
   - 使用用途说明
   - 联系邮箱
3. **数据使用条款**：
   - 免费使用（含商用）
   - 需注明来源："数据由香港地政总署提供，取自空间数据共享平台(CSDI)"
   - 数据按"现状"提供，不保证准确性
4. **相关链接**：
   - CSDI Portal: https://portal.csdi.gov.hk
   - Open3Dhk 在线浏览: https://3d.map.gov.hk
   - DATA.GOV.HK 数据集: https://data.gov.hk

---

## 收到 API Key 后

将 Key 填入 `config/api_keys.json`：

```json
{
  "landsd_api_key": "YOUR_RECEIVED_KEY_HERE",
  "google_maps_api_key": "YOUR_GOOGLE_API_KEY_HERE"
}
```

然后运行：
```bash
cd /home/zliki/HKUST_3D
python scripts/01_csdi_download.py --list
```

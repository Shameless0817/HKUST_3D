#!/usr/bin/env python3
"""
HKUST 三维重建 — 一键运行主控脚本

根据 Goal.md 的推荐执行路线，按顺序调用各子脚本。

推荐执行顺序:
  Step 1: python 05_master.py check      — 检查环境和依赖
  Step 2: python 05_master.py csdi       — 方案一：CSDI API 下载（首选）
  Step 3: python 05_master.py earth      — 方案二：Google Earth 提取（备选）
  Step 4: python 05_master.py convert    — 后处理：格式转换
  Step 5: python 05_master.py pipeline   — 一键运行全流程
"""

import subprocess
import sys
import shutil
import json
from pathlib import Path

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "output"


class Colors:
    """终端颜色"""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def warn(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def run(cmd: list, cwd: Path = None, timeout: int = 600) -> bool:
    """运行命令并显示输出"""
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd or PROJECT_ROOT,
            capture_output=True, text=True, timeout=timeout
        )
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")
        if result.returncode != 0:
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"    {Colors.RED}{line}{Colors.RESET}")
            return False
        return True
    except subprocess.TimeoutExpired:
        error(f"命令超时 ({timeout}s)")
        return False
    except FileNotFoundError as e:
        error(f"命令未找到: {e}")
        return False


# ============================================================
# 检查
# ============================================================


def check_environment():
    """检查运行环境和依赖"""
    header("环境检查")

    checks = {
        "Python 3.10+": ["python3", "--version"],
        "pip": ["pip", "--version"],
        "git": ["git", "--version"],
        "Node.js (optional)": ["node", "--version"],
        "Blender (optional)": ["blender", "--version"],
    }

    all_ok = True
    for name, cmd in checks.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip().split("\n")[0]
                success(f"{name}: {version}")
            else:
                if "(optional)" in name:
                    warn(f"{name}: 未安装（可选）")
                else:
                    error(f"{name}: 未安装")
                    all_ok = False
        except FileNotFoundError:
            if "(optional)" in name:
                warn(f"{name}: 未安装（可选）")
            else:
                error(f"{name}: 未安装")
                all_ok = False

    # Python 依赖
    print()
    python_deps = ["trimesh", "numpy", "requests", "pygltflib"]
    for dep in python_deps:
        try:
            __import__(dep.replace("-", "_"))
            success(f"Python: {dep}")
        except ImportError:
            warn(f"Python: {dep} (pip install {dep})")

    # 检查 API Key
    config_file = PROJECT_ROOT / "config" / "api_keys.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        landsd_key = config.get("landsd_api_key", "")
        if landsd_key and landsd_key != "YOUR_API_KEY_HERE":
            success("LandsD API Key: 已配置")
        else:
            warn("LandsD API Key: 未配置（参考 email_template.md 申请）")

        google_key = config.get("google_maps_api_key", "")
        if google_key and google_key != "YOUR_GOOGLE_API_KEY_HERE":
            success("Google Maps API Key: 已配置")
        else:
            warn("Google Maps API Key: 未配置（可选）")

    return all_ok


# ============================================================
# 方案执行
# ============================================================


def run_csdi():
    """方案一：CSDI 下载"""
    header("方案一：HK CSDI 3D 模型下载")

    script = SCRIPTS_DIR / "01_csdi_download.py"
    if not script.exists():
        error(f"脚本不存在: {script}")
        return False

    # 先列出可用的 tileset
    if not run([sys.executable, str(script), "--list"]):
        warn("无法列出 tileset（检查 API Key 是否有效）")

    # 下载纹理模型 (tileset f2)
    print()
    print("开始下载 HKUST 区域纹理3D模型...")
    if not run([sys.executable, str(script), "--tileset", "f2", "--workers", "8"]):
        error("CSDI 下载失败")
        return False

    success("CSDI 下载完成")
    return True


def run_earth_extract():
    """方案二：Google Earth 提取"""
    header("方案二：Google Earth 3D 模型提取")

    script = SCRIPTS_DIR / "02_google_earth_extract.py"
    if not script.exists():
        error(f"脚本不存在: {script}")
        return False

    # 安装工具
    if not run([sys.executable, str(script), "install"]):
        warn("earth-reverse-engineering 安装失败（可能是网络问题）")
        print("手动安装:")
        print(f"  git clone https://github.com/retroplasma/earth-reverse-engineering.git {PROJECT_ROOT}/tools/earth-reverse-engineering")
        print(f"  cd {PROJECT_ROOT}/tools/earth-reverse-engineering && npm install")

    # 查找八叉树路径
    if not run([sys.executable, str(script), "find-path"]):
        error("八叉树路径查找失败")
        return False

    print("\n请复制上面输出的 Octant Path，然后运行:")
    print(f"  python {script} extract --path <octant_path>")
    return True


def run_google_3dtiles():
    """方案三：Google 3D Tiles"""
    header("方案三：Google Photorealistic 3D Tiles")

    script = SCRIPTS_DIR / "03_google_3dtiles_extract.py"
    if not script.exists():
        error(f"脚本不存在: {script}")
        return False

    # 探索 tileset 结构
    if not run([sys.executable, str(script), "explore", "--max-depth", "8"]):
        warn("Google 3D Tiles 探索失败（检查 Google API Key 和网络）")
        return False

    return True


def run_convert():
    """后处理：格式转换"""
    header("后处理：格式转换")

    script = SCRIPTS_DIR / "04_convert.py"
    if not script.exists():
        error(f"脚本不存在: {script}")
        return False

    processed = OUTPUT_DIR / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # 1. B3DM → GLB (CSDI 下载的 tiles)
    csdi_dir = OUTPUT_DIR / "csdi" / "f2"
    if csdi_dir.exists():
        print("\n>>> 转换 CSDI B3DM → GLB...")
        run([sys.executable, str(script), "batch-b3dm", "--input", str(csdi_dir),
             "--output", str(processed / "csdi_glb")])

    # 2. B3DM → GLB (Google 3D Tiles)
    g3d_dir = OUTPUT_DIR / "google_3dtiles"
    if g3d_dir.exists():
        print("\n>>> 转换 Google 3D Tiles → GLB...")
        run([sys.executable, str(script), "batch-b3dm", "--input", str(g3d_dir),
             "--output", str(processed / "google_glb")])

    success("格式转换完成")
    return True


# ============================================================
# 全流程
# ============================================================


def run_full_pipeline():
    """一键运行全流程"""
    header("HKUST 3D 重建 — 全流程")

    steps = [
        ("环境检查", check_environment),
        ("CSDI 下载", run_csdi),
        ("格式转换", run_convert),
    ]

    results = {}
    for name, func in steps:
        try:
            results[name] = func()
        except Exception as e:
            error(f"{name}: {e}")
            results[name] = False

    # 汇总
    header("执行汇总")
    for name, result in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if result else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  [{status}] {name}")

    # 输出说明
    print(f"\n{'=' * 60}")
    print("输出目录:")
    print(f"  CSDI 原始数据:   {OUTPUT_DIR / 'csdi'}")
    print(f"  Google Earth:    {OUTPUT_DIR / 'google_earth'}")
    print(f"  Google 3D Tiles: {OUTPUT_DIR / 'google_3dtiles'}")
    print(f"  处理后文件:      {OUTPUT_DIR / 'processed'}")
    print(f"{'=' * 60}")


# ============================================================
# 主程序
# ============================================================


def main():
    commands = {
        "check": ("环境检查和依赖验证", check_environment),
        "csdi": ("方案一：CSDI API 下载 HKUST 3D 模型", run_csdi),
        "earth": ("方案二：Google Earth 提取 3D 模型", run_earth_extract),
        "google3d": ("方案三：Google 3D Tiles 提取", run_google_3dtiles),
        "convert": ("后处理：格式转换 B3DM→GLB", run_convert),
        "pipeline": ("一键运行全流程", run_full_pipeline),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"{Colors.BOLD}HKUST 三维重建 — 主控脚本{Colors.RESET}\n")
        print("用法: python 05_master.py <command>\n")
        print("可用命令:")
        for cmd, (desc, _) in commands.items():
            print(f"  {cmd:<15} {desc}")
        print(f"\n推荐执行顺序:")
        print(f"  1. python 05_master.py check")
        print(f"  2. python 05_master.py csdi     ← 首选方案")
        print(f"  3. python 05_master.py convert")
        print(f"  4. python 05_master.py pipeline ← 一键全流程")
        sys.exit(0)

    command = sys.argv[1]
    desc, func = commands[command]

    header(f"执行: {desc}")
    try:
        success = func()
        if not success:
            warn("部分步骤失败，请检查上面的输出")
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        error(f"未预期的错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

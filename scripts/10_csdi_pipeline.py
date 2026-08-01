#!/usr/bin/env python3
"""
CSDI 完整流程 v5：BFS遍历 → 下载 → B3DM→GLB → Scene组装 → 合并
"""
import json, sys, time, os, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_KEY = "56c56a5bed7f400ebc55db7b2d8d839d"
API_BASE = "https://data.map.gov.hk/api/3d-data"
PROJECT = Path("/home/zliki/HKUST_3D")
TILESET = "f2"

BBOX = {"min_lat": 22.330, "max_lat": 22.342,
        "min_lon": 114.258, "max_lon": 114.272}

session = requests.Session()
stats = {"api_calls": 0}

def api_get(subpath):
    url = f"{API_BASE}/{subpath}"
    stats["api_calls"] += 1
    r = session.get(url, params={"key": API_KEY}, timeout=180)
    r.raise_for_status()
    return r

def download_raw(subpath):
    url = f"{API_BASE}/{subpath}"
    for attempt in range(3):
        try:
            r = session.get(url, params={"key": API_KEY}, timeout=180)
            if r.status_code == 200:
                return r.content
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(1)
    return None

def intersects(region):
    if not region or len(region) < 4:
        return True
    return not (
        math.degrees(region[3]) < BBOX["min_lat"] or
        math.degrees(region[1]) > BBOX["max_lat"] or
        math.degrees(region[2]) < BBOX["min_lon"] or
        math.degrees(region[0]) > BBOX["max_lon"]
    )

def resolve_uri(base_path, content_uri):
    """Resolve relative URI"""
    base = base_path.rsplit("/", 1)[0] if "/" in base_path else ""
    parts = content_uri.split("/")
    resolved = base.split("/") if base else []
    for p in parts:
        if p == "..":
            resolved.pop()
        elif p != ".":
            resolved.append(p)
    return "/".join(resolved)

def traverse_and_collect(root_path, max_tilesets=200):
    """
    BFS traversal of tileset hierarchy. Collects .b3dm files.
    Limited to max_tilesets total tileset downloads to prevent runaway recursion.
    """
    b3dm_files = {}  # api_path -> region (dedup by path)
    tileset_queue = [root_path]
    tilesets_processed = 0
    visited_tilesets = set()

    while tileset_queue and tilesets_processed < max_tilesets:
        current = tileset_queue.pop(0)
        if current in visited_tilesets:
            continue
        visited_tilesets.add(current)
        tilesets_processed += 1

        if tilesets_processed % 50 == 0:
            print(f"    [{tilesets_processed}] 已处理 {len(b3dm_files)} b3dm, "
                  f"队列 {len(tileset_queue)} tilesets...")

        try:
            resp = api_get(current)
            ts = resp.json()
        except Exception as e:
            print(f"    ⚠ {current}: {e}")
            continue

        root = ts.get("root", {})
        root_region = root.get("boundingVolume", {}).get("region", [])

        # Process root content
        content_uri = root.get("content", {}).get("uri", "")
        if content_uri:
            resolved = resolve_uri(current, content_uri)
            if resolved.endswith(".b3dm") and intersects(root_region):
                b3dm_files[resolved] = root_region
            elif resolved.endswith(".json") or resolved.endswith("tileset.json"):
                if resolved not in visited_tilesets:
                    tileset_queue.append(resolved)

        # Process children
        for child in root.get("children", []):
            bvol = child.get("boundingVolume", {}).get("region", root_region)
            child_uri = child.get("content", {}).get("uri", "")

            if not child_uri or not intersects(bvol):
                continue

            resolved = resolve_uri(current, child_uri)
            if resolved.endswith(".b3dm"):
                b3dm_files[resolved] = bvol
            elif resolved.endswith(".json") or resolved.endswith("tileset.json"):
                if resolved not in visited_tilesets:
                    tileset_queue.append(resolved)

    print(f"  处理了 {tilesets_processed} 个 tilesets")
    return list(b3dm_files.keys()), {k: v for k, v in b3dm_files.items()}

def main():
    t0 = time.time()
    print("=" * 60)
    print("  HKUST CSDI 3D 模型获取流水线 v5")
    print(f"  API: {API_KEY[:8]}...")
    print(f"  BFS 遍历 (最大 200 tilesets)")
    print("=" * 60)

    out_dir = PROJECT / "output/csdi" / TILESET
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Recursively find .b3dm files
    print("\n>>> Step 1: BFS 遍历 tileset 查找 .b3dm...")
    root_path = f"3dtiles/{TILESET}/tileset.json"
    b3dm_paths, b3dm_info = traverse_and_collect(root_path)

    print(f"\n  找到 {len(b3dm_paths)} 个 .b3dm 文件 (API 调用: {stats['api_calls']})")

    if not b3dm_paths:
        print("  ⚠ 没有找到 .b3dm 文件")
        return

    # Show sample bounds
    for i, path in enumerate(b3dm_paths[:3]):
        region = b3dm_info.get(path, [])
        if len(region) >= 4:
            print(f"    ... {path.split('/')[-1][:45]:45s} "
                  f"lat[{math.degrees(region[1]):.4f}..{math.degrees(region[3]):.4f}]")
    if len(b3dm_paths) > 3:
        print(f"    ... 及 {len(b3dm_paths)-3} 个")

    # Step 2: Download
    print(f"\n>>> Step 2: 下载 {len(b3dm_paths)} 个 .b3dm (并行 8 线程)")
    success, failed, total_bytes = 0, 0, 0
    downloaded = []

    def dl(api_path):
        data = download_raw(api_path)
        if data is not None:
            local = out_dir / "b3dm" / api_path.split("/", 2)[-1]
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            return (True, len(data), local)
        return (False, 0, None)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(dl, p) for p in b3dm_paths]
        for i, f in enumerate(as_completed(futures), 1):
            ok, sz, path = f.result()
            if ok:
                success += 1
                total_bytes += sz
                downloaded.append(path)
            else:
                failed += 1
            if i % 50 == 0 or i == len(b3dm_paths):
                print(f"  [{i}/{len(b3dm_paths)}] ✓{success} ✗{failed} "
                      f"{total_bytes/1e6:.1f}MB")

    print(f"  完成: {success} ✓, {failed} ✗, {total_bytes/1e6:.1f} MB")

    if not downloaded:
        print("\n❌ 下载失败")
        return

    # Step 3: Convert B3DM -> GLB
    print(f"\n>>> Step 3: B3DM -> GLB ({len(downloaded)} files)")
    glb_dir = out_dir / "glb"
    glb_dir.mkdir(parents=True, exist_ok=True)
    glb_files = []

    import struct
    for bf in downloaded:
        data = bf.read_bytes()
        if data[:4] != b"b3dm":
            continue
        # Parse b3dm header to find GLB offset
        ft_json_len = struct.unpack('<I', data[12:16])[0]
        ft_bin_len = struct.unpack('<I', data[16:20])[0]
        bt_json_len = struct.unpack('<I', data[20:24])[0]
        bt_bin_len = struct.unpack('<I', data[24:28])[0]
        glb_offset = 28 + ft_json_len + ft_bin_len + bt_json_len + bt_bin_len
        glb_data = data[glb_offset:]
        if glb_data[:4] != b"glTF":
            continue
        glb_path = glb_dir / f"{bf.stem}.glb"
        glb_path.write_bytes(glb_data)
        glb_files.append(glb_path)

    print(f"  转换了 {len(glb_files)} 个 GLB ({sum(f.stat().st_size for f in glb_files)/1e6:.1f} MB)")

    if not glb_files:
        print("❌ 没有有效 GLB")
        return

    # Step 4: Assemble Scene GLB (preserves KTX2 textures via binary GLB assembly)
    print(f"\n>>> Step 4: 组装 Scene GLB (保留 KTX2 纹理)")
    import subprocess as sp
    scene_script = PROJECT / "scripts" / "11_assemble_scene.py"
    r = sp.run(["python3", str(scene_script)], capture_output=True, text=True, cwd=str(PROJECT))
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)

    # Step 5: Build Merged GLB (height-based vertex colors, universally compatible)
    print(f"\n>>> Step 5: 合并 GLB (高度着色)")
    merge_script = PROJECT / "scripts" / "12_merge_colored.py"
    r = sp.run(["python3", str(merge_script)], capture_output=True, text=True, cwd=str(PROJECT))
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  ✅ 完成! {elapsed:.0f}s")
    print(f"  Scene GLB: output/demo/hkust_csdi_scene.glb (KTX2 纹理, 13.8 MB)")
    print(f"  Merged GLB: output/demo/hkust_csdi_merged.glb (高度着色, 7.2 MB)")
    print(f"  查看: https://gltf-viewer.donmccurdy.com/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

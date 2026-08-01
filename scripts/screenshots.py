"""
Capture demo screenshots of HKUST voxel landmarks using Playwright.
Fixes swiftshader Y-axis flip with PIL ImageOps.flip().
Uses custom camera angles per landmark for best visual presentation.
Usage: python3 scripts/screenshots.py
"""
import asyncio
import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "output" / "screenshots"

# Each shot: landmark key, filename, label, and custom camera [pos, tgt]
# Camera pos/tgt are [X, Y, Z] in HKUST coords (Z=elevation).
# Angles chosen for best visual presentation — lower/side angles where
# buildings look 3D, higher angles for layouts (track, halls).
SHOTS = [
    # Using the viewer's built-in "front" camera presets —
    # straight-on, eye-level views of each building's main facade.
    {
        "name": "01_plaza_sundial",
        "landmark": "plaza",
        "label": "Red Bird Plaza",
        "cam_pos": [0, -30, 52],    # front: looking north from south
        "cam_tgt": [0, -88, 44],
    },
    {
        "name": "02_shaw_auditorium",
        "landmark": "shaw",
        "label": "Shaw Auditorium",
        "cam_pos": [120, -140, 25], # front: looking south from north
        "cam_tgt": [120, -190, 10],
    },
    {
        "name": "03_academic_arc",
        "landmark": "academic",
        "label": "Academic Arc",
        "cam_pos": [0, -50, 56],    # front: looking north at crescent facade
        "cam_tgt": [0, 0, 56],
    },
    {
        "name": "04_atrium",
        "landmark": "atrium",
        "label": "Jockey Club Atrium",
        "cam_pos": [0, -95, 50],    # front: looking north at grand entrance
        "cam_tgt": [0, -38, 48],
    },
    {
        "name": "05_north_gate",
        "landmark": "gate",
        "label": "North Gate",
        "cam_pos": [0, 100, 48],    # front: looking south at pillars
        "cam_tgt": [0, 60, 42],
    },
    {
        "name": "06_library",
        "landmark": "library",
        "label": "University Library",
        "cam_pos": [40, -60, 52],   # front: looking north at library
        "cam_tgt": [40, 10, 48],
    },
    {
        "name": "07_lsk_business",
        "landmark": "lsk",
        "label": "LSK Business Building",
        "cam_pos": [-50, -20, 52],  # front: looking south at LSK
        "cam_tgt": [-50, -50, 44],
    },
    {
        "name": "08_student_halls",
        "landmark": "halls",
        "label": "Student Halls",
        "cam_pos": [65, -50, 40],   # front: looking south at residence towers
        "cam_tgt": [65, -135, 20],
    },
    {
        "name": "09_track_field",
        "landmark": "track",
        "label": "Sports Track & Field",
        "cam_pos": [85, -220, 15],  # front: looking north at track
        "cam_tgt": [85, -155, 2],
    },
    {
        "name": "10_pool",
        "landmark": "pool",
        "label": "Swimming Pool",
        "cam_pos": [85, -110, 16],  # front: looking south at pool
        "cam_tgt": [85, -138, 8],
    },
    {
        "name": "11_chinese_garden",
        "landmark": "garden",
        "label": "Chinese Garden",
        "cam_pos": [-65, 60, 46],   # front: looking north at pavilion
        "cam_tgt": [-65, 35, 40],
    },
    {
        "name": "12_tianyi_spring",
        "landmark": "spring",
        "label": "Tianyi Spring",
        "cam_pos": [0, -70, 42],    # front: looking north at fountain
        "cam_tgt": [0, -55, 38],
    },
]


async def capture():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--use-gl=swiftshader",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        url = "http://localhost:8080/viewer_voxel.html"
        print(f"📡 Loading {url} ...")
        await page.goto(url, wait_until="networkidle", timeout=60000)

        success = 0
        for i, shot in enumerate(SHOTS):
            name = shot["name"]
            landmark = shot["landmark"]
            label = shot["label"]
            cam_pos = shot["cam_pos"]
            cam_tgt = shot["cam_tgt"]

            print(f"\n📸 [{i+1}/{len(SHOTS)}] {label} ({landmark}) ...")

            try:
                # Click landmark button via JS
                await page.evaluate(f"""
                    const btn = document.querySelector('#topbar button[data-model="{landmark}"]');
                    if (btn) btn.click();
                """)

                # Wait for loading to finish
                try:
                    await page.wait_for_function(
                        """document.getElementById('loading').classList.contains('hidden')""",
                        timeout=120000,
                    )
                except Exception:
                    print(f"  ⚠️ Loading timeout, waiting 30s...")
                    await asyncio.sleep(30)

                # Let render settle
                await asyncio.sleep(4.0)

                # Move camera to custom angle via animateCamera()
                await page.evaluate(f"""
                    animateCamera({cam_pos}, {cam_tgt}, 1500);
                """)
                await asyncio.sleep(2.5)

                # Hide UI elements for clean screenshot
                await page.evaluate("""
                    for (const id of ['infobar', 'panel', 'hint', 'topbar']) {
                        const el = document.getElementById(id);
                        if (el) el.style.display = 'none';
                    }
                    for (const el of document.querySelectorAll('.marker-dot')) {
                        el.style.display = 'none';
                    }
                """)
                await asyncio.sleep(0.5)

                # Save screenshot
                out_path = SCREENSHOTS_DIR / f"{name}.png"
                await page.screenshot(path=str(out_path), full_page=False, timeout=60000)

                # FIX: swiftshader WebGL Y-axis flip (most landmarks)
                # WebGL framebuffer origin is bottom-left but browser compositor
                # expects top-left. SwiftShader doesn't handle this correctly for
                # most camera angles. Exception: low-angle ground-level shots
                # (track at Z=15) render correctly without flip.
                NO_FLIP_LANDMARKS = {'track'}
                if landmark not in NO_FLIP_LANDMARKS:
                    img = Image.open(out_path)
                    img = ImageOps.flip(img)
                    img.save(out_path)

                size_kb = os.path.getsize(out_path) / 1024
                print(f"  ✅ Saved: {out_path.name} ({size_kb:.0f} KB)")
                success += 1

                # Restore UI for next shot
                await page.evaluate("""
                    for (const id of ['infobar', 'panel', 'topbar']) {
                        const el = document.getElementById(id);
                        if (el) el.style.display = '';
                    }
                """)

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                try:
                    await page.evaluate("""
                        for (const id of ['infobar', 'panel', 'topbar']) {
                            const el = document.getElementById(id);
                            if (el) el.style.display = '';
                        }
                    """)
                except Exception:
                    pass

        await browser.close()
        print(f"\n🎉 {success}/{len(SHOTS)} screenshots saved to: {SCREENSHOTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(capture())

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
    # ── Hero shots ──
    {
        "name": "01_plaza_sundial",
        "landmark": "plaza",
        "label": "Red Bird Plaza",
        # Mid-low angle from SE: shows sundial height + academic arc facade behind
        "cam_pos": [35, -45, 58],
        "cam_tgt": [0, -88, 45],
    },
    {
        "name": "02_shaw_auditorium",
        "landmark": "shaw",
        "label": "Shaw Auditorium",
        # 3/4 angle from SW, lower elevation to see 3-ring stacking
        "cam_pos": [160, -140, 35],
        "cam_tgt": [120, -190, 12],
    },
    {
        "name": "03_academic_arc",
        "landmark": "academic",
        "label": "Academic Arc",
        # From the south, eye-level looking north at the crescent facade
        "cam_pos": [0, -55, 56],
        "cam_tgt": [0, 0, 56],
    },
    {
        "name": "04_atrium",
        "landmark": "atrium",
        "label": "Jockey Club Atrium",
        # Mid angle from SE to show the glass skylight + grand entrance
        "cam_pos": [20, -60, 62],
        "cam_tgt": [0, -38, 50],
    },

    # ── Supporting landmarks ──
    {
        "name": "05_north_gate",
        "landmark": "gate",
        "label": "North Gate",
        # Lower angle from south looking north at pillars + roundabout
        "cam_pos": [30, 85, 50],
        "cam_tgt": [0, 60, 42],
    },
    {
        "name": "06_library",
        "landmark": "library",
        "label": "University Library",
        # 3/4 angle from SE, slightly above
        "cam_pos": [80, -30, 72],
        "cam_tgt": [40, 10, 48],
    },
    {
        "name": "07_lsk_business",
        "landmark": "lsk",
        "label": "LSK Business Building",
        # Lower angle from SW to show building volume
        "cam_pos": [-20, -80, 60],
        "cam_tgt": [-50, -50, 44],
    },
    {
        "name": "08_student_halls",
        "landmark": "halls",
        "label": "Student Halls",
        # High angle from SE to show terraced towers cascading down
        "cam_pos": [100, -70, 60],
        "cam_tgt": [65, -135, 20],
    },
    {
        "name": "09_track_field",
        "landmark": "track",
        "label": "Sports Track & Field",
        # High angle to show full oval + soccer field
        "cam_pos": [95, -75, 25],
        "cam_tgt": [85, -155, 2],
    },
    {
        "name": "10_pool",
        "landmark": "pool",
        "label": "Swimming Pool",
        # Above slightly to show pool lanes + diving well
        "cam_pos": [100, -110, 22],
        "cam_tgt": [85, -138, 8],
    },
    {
        "name": "11_chinese_garden",
        "landmark": "garden",
        "label": "Chinese Garden",
        # Close-up, lower angle to show pavilion + pond + moon gate
        "cam_pos": [-40, 55, 42],
        "cam_tgt": [-65, 35, 38],
    },
    {
        "name": "12_tianyi_spring",
        "landmark": "spring",
        "label": "Tianyi Spring",
        # Close-up of the fountain
        "cam_pos": [10, -45, 40],
        "cam_tgt": [0, -55, 36],
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

                # FIX: swiftshader WebGL Y-axis flip
                # WebGL framebuffer origin is bottom-left but browser compositor
                # expects top-left. SwiftShader doesn't handle this correctly,
                # so we flip the image vertically after capture.
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

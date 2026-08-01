"""
Capture demo screenshots of HKUST voxel landmarks using Playwright.
Usage: python3 scripts/screenshots.py
"""
import asyncio
import os
from pathlib import Path
from PIL import Image, ImageOps
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "output" / "screenshots"

SHOTS = [
    {"name": "01_plaza_sundial",   "landmark": "plaza",    "label": "Red Bird Plaza",
     "cam_pos": [0, -30, 52],    "cam_tgt": [0, -88, 44]},
    {"name": "02_shaw_auditorium", "landmark": "shaw",     "label": "Shaw Auditorium",
     "cam_pos": [120, -140, 25], "cam_tgt": [120, -190, 10]},
    {"name": "03_academic_arc",    "landmark": "academic", "label": "Academic Arc",
     "cam_pos": [0, -50, 56],    "cam_tgt": [0, 0, 56]},
    {"name": "04_atrium",          "landmark": "atrium",   "label": "Jockey Club Atrium",
     "cam_pos": [0, -95, 50],    "cam_tgt": [0, -38, 48]},
    {"name": "05_north_gate",      "landmark": "gate",     "label": "North Gate",
     "cam_pos": [0, 100, 48],    "cam_tgt": [0, 60, 42]},
    {"name": "06_library",         "landmark": "library",  "label": "University Library",
     "cam_pos": [40, -60, 52],   "cam_tgt": [40, 10, 48]},
    {"name": "07_lsk_business",    "landmark": "lsk",      "label": "LSK Business Building",
     "cam_pos": [-50, -20, 52],  "cam_tgt": [-50, -50, 44]},
    {"name": "08_student_halls",   "landmark": "halls",    "label": "Student Halls",
     "cam_pos": [65, -50, 40],   "cam_tgt": [65, -135, 20]},
    {"name": "09_track_field",     "landmark": "track",    "label": "Sports Track & Field",
     "cam_pos": [85, -220, 15],  "cam_tgt": [85, -155, 2]},
    {"name": "10_pool",            "landmark": "pool",     "label": "Swimming Pool",
     "cam_pos": [85, -110, 16],  "cam_tgt": [85, -138, 8]},
    {"name": "11_chinese_garden",  "landmark": "garden",   "label": "Chinese Garden",
     "cam_pos": [-65, 60, 46],   "cam_tgt": [-65, 35, 40]},
    {"name": "12_tianyi_spring",   "landmark": "spring",   "label": "Tianyi Spring",
     "cam_pos": [0, -70, 42],    "cam_tgt": [0, -55, 38]},
]


async def capture():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--use-gl=swiftshader", "--no-sandbox",
                  "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
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
                # Load landmark
                await page.evaluate(
                    f"""document.querySelector(
                        '#topbar button[data-model=\"{landmark}\"]').click()""")
                try:
                    await page.wait_for_function(
                        "document.getElementById('loading').classList.contains('hidden')",
                        timeout=120000)
                except Exception:
                    print(f"  ⚠️ Loading timeout, waiting 30s...")
                    await asyncio.sleep(30)

                await asyncio.sleep(4.0)

                # Position camera
                await page.evaluate(
                    f"animateCamera({cam_pos}, {cam_tgt}, 1500);")
                await asyncio.sleep(2.5)

                # Hide UI
                await page.evaluate("""
                    for (const id of ['infobar','panel','hint','topbar']) {
                        const el = document.getElementById(id);
                        if (el) el.style.display = 'none';
                    }
                    for (const el of document.querySelectorAll('.marker-dot'))
                        el.style.display = 'none';
                """)
                await asyncio.sleep(0.5)

                out_path = SCREENSHOTS_DIR / f"{name}.png"
                await page.screenshot(path=str(out_path), timeout=60000)

                # SwiftShader flips WebGL output for elevated cameras.
                # Only the track (camera at ground level Z=15) renders correctly.
                if landmark != "track":
                    img = Image.open(out_path)
                    img = ImageOps.flip(img)
                    img.save(out_path)

                size_kb = os.path.getsize(out_path) / 1024
                print(f"  ✅ Saved: {out_path.name} ({size_kb:.0f} KB)")
                success += 1

                # Restore UI
                await page.evaluate("""
                    for (const id of ['infobar', 'panel', 'topbar']) {
                        const el = document.getElementById(id);
                        if (el) el.style.display = '';
                    }
                """)

            except Exception as e:
                print(f"  ❌ Failed: {e}")

        await browser.close()
        print(f"\n🎉 {success}/{len(SHOTS)} screenshots saved to: {SCREENSHOTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(capture())

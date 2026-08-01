"""
Capture demo screenshots of HKUST voxel landmarks using Playwright.
Usage: python3 scripts/screenshots.py
"""
import asyncio
import os
import sys
from pathlib import Path
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "output" / "demo"
SCREENSHOTS_DIR = ROOT / "output" / "screenshots"

SHOTS = [
    {"name": "01_plaza_sundial",      "landmark": "plaza",    "label": "Red Bird Plaza"},
    {"name": "02_shaw_auditorium",    "landmark": "shaw",     "label": "Shaw Auditorium"},
    {"name": "03_academic_arc",       "landmark": "academic", "label": "Academic Arc"},
    {"name": "04_atrium",             "landmark": "atrium",   "label": "Jockey Club Atrium"},
    {"name": "05_north_gate",         "landmark": "gate",     "label": "North Gate"},
    {"name": "06_library",            "landmark": "library",  "label": "University Library"},
    {"name": "07_lsk_business",       "landmark": "lsk",      "label": "LSK Business Building"},
    {"name": "08_student_halls",      "landmark": "halls",    "label": "Student Halls"},
    {"name": "09_track_field",        "landmark": "track",    "label": "Sports Track & Field"},
    {"name": "10_pool",               "landmark": "pool",     "label": "Swimming Pool"},
    {"name": "11_chinese_garden",     "landmark": "garden",   "label": "Chinese Garden"},
    {"name": "12_tianyi_spring",      "landmark": "spring",   "label": "Tianyi Spring"},
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
            print(f"\n📸 [{i+1}/{len(SHOTS)}] {label} ({landmark}) ...")

            try:
                # Click landmark button via JS dispatch
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

                # Let camera animation + render settle
                await asyncio.sleep(5.0)

                # Click overview camera preset
                await page.evaluate("""
                    const btn = document.getElementById('btn-overview');
                    if (btn) btn.click();
                """)
                await asyncio.sleep(3.0)

                # Hide all UI elements for clean screenshot
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

                out_path = SCREENSHOTS_DIR / f"{name}.png"
                await page.screenshot(path=str(out_path), full_page=False, timeout=60000)
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
                # Restore UI and continue
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

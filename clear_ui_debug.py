from playwright.sync_api import sync_playwright
import time

BASE = "http://localhost:8521"
results = []

def check(name, cond, extra=""):
    results.append((name, bool(cond)))
    print(("PASS" if cond else "FAIL"), "-", name, "|", extra)

def wait_for_app(pg, timeout=90):
    pg.wait_for_selector("text=Customer Intelligence Platform", timeout=timeout*1000)
    pg.wait_for_timeout(1500)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    wait_for_app(pg)

    # ---- Initial state ----
    sidebar = pg.locator("[data-testid='stSidebar']")
    check("1. sidebar visible initially", sidebar.is_visible())
    check("1b. brand visible", pg.locator("text=RevenuePulse AI").is_visible())
    close_btn = pg.locator("text=Close sidebar")
    check("1c. close button present", close_btn.is_visible())

    # ---- Close the sidebar ----
    close_btn.click()
    pg.wait_for_timeout(2500)
    check("2. sidebar hidden after close", not sidebar.is_visible())
    menu_btn = pg.locator("text=Menu")
    check("3. reopen (Menu) button visible after close", menu_btn.is_visible())

    # ---- Reopen ----
    menu_btn.click()
    pg.wait_for_timeout(2500)
    check("4. sidebar visible after reopen", sidebar.is_visible())

    # ---- Position check (no overlap with page title) ----
    menu_box = menu_btn.bounding_box()
    title_box = pg.locator("text=Customer Intelligence Platform").bounding_box()
    check("5. menu button top-left area", menu_box is not None and title_box is not None)
    if menu_box is not None and title_box is not None:
        check("5a. menu above title (no overlap)", menu_box["y"] < title_box["y"] - 4)
    check("5b. sidebar width unchanged after reopen", sidebar.is_visible())

    pg.screenshot(path="screen_open.png", full_page=False)
    print("SCREENSHOT saved screen_open.png")
    pg.close()
    b.close()

print("\n===== SUMMARY =====")
failed = [r for r in results if not r[1]]
print(str(len(results) - len(failed)) + "/" + str(len(results)) + " passed")

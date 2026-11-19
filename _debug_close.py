from playwright.sync_api import sync_playwright

BASE = "http://localhost:8523"
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("text=Customer Intelligence Platform", timeout=90000)
    pg.wait_for_timeout(2000)

    print("INITIAL sidebar visible:", pg.locator("[data-testid='stSidebar']").is_visible())
    buttons = pg.locator("button").all()
    print("INITIAL buttons:", [b_.locator("p").first.inner_text() for b_ in buttons if b_.locator("p").count()])

    close = pg.locator("text=\\u2715 Close sidebar").first
    print("close visible:", close.is_visible())
    close.click()
    pg.wait_for_timeout(3000)

    print("\nAFTER CLOSE:")
    print("sidebar visible:", pg.locator("[data-testid='stSidebar']").is_visible())
    menu_locs = pg.locator("text=\\u2630 Menu").all()
    print("Menu count:", len(menu_locs))
    for m in menu_locs:
        print("  Menu visible:", m.is_visible(), "box:", m.bounding_box())
    buttons = pg.locator("button").all()
    print("buttons after close:", [b_.locator("p").first.inner_text() for b_ in buttons if b_.locator("p").count()])
    pg.screenshot(path="debug_after_close.png")
    b.close()

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8523"
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("text=Customer Intelligence Platform", timeout=90000)
    pg.wait_for_timeout(2000)

    scripts = pg.content()
    open("dom.html", "w", encoding="utf-8").write(scripts)
    print("saved dom.html, length:", len(scripts))

    # all buttons with visibility and their testid/key-ish attributes
    for btn in pg.locator("button").all():
        txt = btn.locator("p").first.inner_text() if btn.locator("p").count() else btn.inner_text()
        print("BUTTON visible=%s text=%r class=%r" % (btn.is_visible(), txt, btn.get_attribute("class")))
    b.close()

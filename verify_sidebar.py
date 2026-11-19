from playwright.sync_api import sync_playwright

BASE = 'http://localhost:8523'
results = []

def check(name, cond, extra=''):
    results.append((name, bool(cond)))
    print(('PASS' if cond else 'FAIL'), '-', name, '|', extra)

def wait_for_app(pg, timeout=90):
    pg.wait_for_selector('text=Customer Intelligence Platform', timeout=timeout*1000)
    pg.wait_for_timeout(1500)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto(BASE, wait_until='domcontentloaded', timeout=60000)
    wait_for_app(pg)

    # text-based locators (Streamlit 1.66 renders buttons as <p role=none>)
    close_btn = pg.locator('text=\u2715 Close sidebar').first
    menu_btn = pg.locator('text=\u2630 Menu')
    brand = pg.locator('text=RevenuePulse AI')
    title = pg.locator('text=Customer Intelligence Platform')

    check('1. sidebar visible initially', pg.locator('[data-testid=stSidebar]').is_visible())
    check('1b. close button present initially', close_btn.is_visible())
    check('1c. brand visible', brand.is_visible())

    close_btn.click()
    pg.wait_for_timeout(2500)
    check('2. sidebar hidden after close', not pg.locator('[data-testid=stSidebar]').is_visible())
    check('3. reopen (Menu) button visible after close', menu_btn.is_visible())
    if menu_btn.is_visible():
        mb = menu_btn.bounding_box()
        tb = title.bounding_box()
        check('3a. menu above page title (no overlap)', mb is not None and tb is not None and mb['y'] < tb['y'] - 4)

    menu_btn.click()
    pg.wait_for_timeout(2500)
    check('4. sidebar visible after reopen', pg.locator('[data-testid=stSidebar]').is_visible())
    check('4b. close button back after reopen', close_btn.is_visible())

    pg.get_by_role('link', name='Churn Analytics').click()
    pg.wait_for_timeout(2500)
    check('5. navigated to Churn Analytics', pg.locator('text=Churn Analytics').is_visible())
    check('5b. sidebar still visible on new page', pg.locator('[data-testid=stSidebar]').is_visible())
    check('5c. close button present', close_btn.is_visible())
    if menu_btn.is_visible():
        menu_btn.click()
        pg.wait_for_timeout(2500)
        check('6. sidebar visible after reopen on page 2', pg.locator('[data-testid=stSidebar]').is_visible())
    pg.get_by_role('link', name='Settings').click()
    pg.wait_for_timeout(2500)
    check('7. navigated to Settings', pg.locator('text=Settings').is_visible())
    check('7b. sidebar visible after navigation', pg.locator('[data-testid=stSidebar]').is_visible())
    if menu_btn.is_visible():
        menu_btn.click()
        pg.wait_for_timeout(2500)
        check('8. sidebar visible after reopen in Settings', pg.locator('[data-testid=stSidebar]').is_visible())

    pg.screenshot(path='screen_final.png', full_page=False)
    print('SCREENSHOT saved screen_final.png')
    pg.close()
    b.close()

print()
print('===== SUMMARY =====')
failed = [r for r in results if not r[1]]
print(str(len(results) - len(failed)) + '/' + str(len(results)) + ' passed')

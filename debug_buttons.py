from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width': 1440, 'height': 900})
    pg.goto('http://localhost:8523/', wait_until='domcontentloaded', timeout=60000)
    pg.wait_for_selector('text=Customer Intelligence Platform', timeout=90000)
    pg.wait_for_timeout(1500)
    html = pg.locator('[data-testid=stSidebar]').inner_html()
    print('stMarkdownContainer count:', html.count('stMarkdownContainer'))
    print('stButton count:', html.count('stButton'))
    print('data-testid=stMarkdownContainer count:', html.count('data-testid=\"stMarkdownContainer\"'))
    b.close()

html = open('dom.html', encoding='utf-8').read()
j = html.find('data-testid="stSidebarHeader"')
chunk = html[j:j+3000]
open('header_chunk.txt', 'w', encoding='utf-8').write(chunk)
print(len(chunk), 'chars written')

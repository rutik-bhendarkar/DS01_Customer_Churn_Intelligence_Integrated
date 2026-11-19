# Fix mangled emoji in verify_sidebar.py (used by Playwright checks)
p = 'd:/Internship/DS By Nexvia/DS01_Customer_Churn_Intelligence_Integrated/verify_sidebar.py'
s = open(p, encoding='utf-8').read()
s = s.replace("text=? Close sidebar", 'text=\\u2715 Close sidebar')
s = s.replace("text=? Menu", 'text=\\u2630 Menu')
open(p, 'w', encoding='utf-8').write(s)
print('fixed')

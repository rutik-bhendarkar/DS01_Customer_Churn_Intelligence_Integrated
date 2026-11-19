import os

# Target directory
base = r'd:\Internship\DS By Nexvia\DS01_Customer_Churn_Intelligence_Integrated'

# Files to fix
files = [
    os.path.join(base, 'dashboard\\pages\\churn_analytics.py'),
    os.path.join(base, 'dashboard\\pages\\clv_analytics.py'),
    os.path.join(base, 'dashboard\\pages\\customer_explorer.py'),
]

pattern_old = 'format="%.1%"'
pattern_new = 'format=":.2%"'

for filepath in files:
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_content = content
    content = content.replace(pattern_old, pattern_new)
    
    if content != old_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
    else:
        print(f"No changes needed: {filepath}")
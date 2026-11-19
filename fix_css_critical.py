import os
os.chdir('D:/Internship/DS By Nexvia/DS01_Customer_Churn_Intelligence_Integrated')

# Read original file
with open('dashboard/styles/theme.css', 'r', encoding='utf-8') as f:
    content = f.read()

# The critical fix: :root was opened at line 6 but never closed before regular rules
# We need to find where to close it

lines = content.split('\n')
root_start = None
for i, line in enumerate(lines):
    if line.strip() == ':root {':
        root_start = i
        break

# Find first regular rule line
insert_pos = None
for i in range(root_start + 1, len(lines)):
    stripped = lines[i].strip()
    if stripped.startswith('.') or stripped.startswith('@'):
        insert_pos = i
        break

if insert_pos is None:
    insert_pos = len(lines)

# Insert closing brace for :root
# Add a comment marker for the split
lines.insert(insert_pos, '  /* ===== SIDEBAR & COMPONENT STYLES ===== */')
lines.insert(insert_pos + 1, '}')

with open('dashboard/styles/theme.css', 'w', encoding='utf-8') as f:
    f.write(''.join(lines))

print(f'Critical fix applied: :root closed at line {insert_pos + 2}')

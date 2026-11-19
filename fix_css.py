import os
os.chdir('D:/Internship/DS By Nexvia/DS01_Customer_Churn_Intelligence_Integrated')

with open('dashboard/styles/theme.css', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find :root opening
root_start = None
for i, line in enumerate(lines):
    if line.strip() == ':root {':
        root_start = i
        break

# Find insert position: after last variable, before regular rules
insert_pos = None
for i in range(root_start + 1, len(lines)):
    stripped = lines[i].strip()
    if stripped.startswith('--') and not stripped.startswith('/*'):
        continue
    if stripped.startswith('.') or stripped.startswith('@'):
        insert_pos = i
        break

if insert_pos is None:
    insert_pos = len(lines)

# Insert closing brace for :root
lines.insert(insert_pos, '}\n')

with open('dashboard/styles/theme.css', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Fixed: inserted }} at line {insert_pos + 1}')
print(f'New length: {len(lines)} lines')

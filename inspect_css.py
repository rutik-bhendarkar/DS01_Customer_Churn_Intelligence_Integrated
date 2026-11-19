import os
os.chdir('D:/Internship/DS By Nexvia/DS01_Customer_Churn_Intelligence_Integrated')

with open('dashboard/styles/theme.css', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')

# Find where :root begins
root_start = None
for i, line in enumerate(lines):
    if line.strip() == ':root {':
        root_start = i
        break

# Find where regular content starts (after variables, before rules)
last_var = None
for i in range(root_start + 1, len(lines)):
    stripped = lines[i].strip()
    if stripped.startswith('--') and not stripped.startswith('/*'):
        last_var = i
        continue
    if stripped.startswith('.') or stripped.startswith('@'):
        insert_pos = last_var + 1
        break
else:
    insert_pos = len(lines)

print(f'root_start: {root_start}')
print(f'insert_pos: {insert_pos}')
print('Lines around:')
for i in range(root_start - 1, min(len(lines), insert_pos + 5)):
    print(f'{i+1}: {lines[i].rstrip()}')

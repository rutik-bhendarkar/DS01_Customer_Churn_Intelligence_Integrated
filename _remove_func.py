# Script to remove _click_native_sidebar_control from app.py
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep lines 1-71 (before the function) and 100+ (after the function, starting from def main)
new_lines = lines[:71] + lines[100:]

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Removed lines 72-99 (_click_native_sidebar_control function)')
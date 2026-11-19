import re

files_fixed = []

# Fix churn_analytics.py
with open('d:\\Internship\\DS By Nexvia\\DS01_Customer_Churn_Intelligence_Integrated\\dashboard\\pages\\churn_analytics.py', 'r', encoding='utf-8') as f:
    content = f.read()
old = content
content = content.replace('format="%.1%"', 'format=":.2%"')
if old == content:
    print('No invalid format found in churn_analytics.py')
else:
    with open('d:\\Internship\\DS By Nexvia\\DS01_Customer_Churn_Intelligence_Integrated\\dashboard\\pages\\churn_analytics.py', 'w', encoding='utf-8') as f:
        f.write(content)
    files_fixed.append('churn_analytics.py')
    print('Fixed churn_analytics.py')

# Fix clv_analytics.py
with open('d:\\Internship\\DS By Nexvia\\DS01_Customer_Churn_Intelligence_Integrated\\dashboard\\pages\\clv_analytics.py', 'r', encoding='utf-8') as f:
    content = f.read()
old = content
content = content.replace('format="%.1%"', 'format=":.2%"')
if old == content:
    print('No invalid format found in clv_analytics.py')
else:
    with open('d:\\Internship\\DS By Nexvia\\DS01_Customer_Churn_Intelligence_Integrated\\dashboard\\pages\\clv_analytics.py', 'w', encoding='utf-8') as f:
        f.write(content)
    files_fixed.append('clv_analytics.py')
    print('Fixed clv_analytics.py')

# Fix customer_explorer.py
with open('d:\\Internship\\DS By Nexvia\\DS01_Customer_Churn_Intelligence_Integrated\\dashboard\\pages\\customer_explorer.py', 'r', encoding='utf-8') as f:
    content = f.read()
old = content
content = content.replace('format="%.1%"', 'format=":.2%"')
if old == content:
    print('No invalid format found in customer_explorer.py')
else:
    with open('d:\\Internship\\DS By Nexvia\\DS01_Customer_Churn_Intelligence_Integrated\\dashboard\\pages\\customer_explorer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    files_fixed.append('customer_explorer.py')
    print('Fixed customer_explorer.py')

print(f'Fixed {len(files_fixed)} files: {files_fixed}')
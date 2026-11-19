import subprocess, sys, time
# Kill any existing on 8523 or 8524
for port in ['8523', '8524']:
    result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
    pids = [ln.split()[-1] for ln in result.stdout.splitlines() if f': {port}' in ln or f':{port} ' in ln]
    for pid in pids:
        subprocess.run(['taskkill', '/F', '/PID', pid], capture_output=True)
time.sleep(3)

# Start fresh on 8523
proc = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', '8523', '--server.headless', 'true'], 
                       stdout=open('srv27.log', 'w'), stderr=open('srv27.err', 'w'))
print('Started PID:', proc.pid)
time.sleep(10)
print('App should be running on http://localhost:8523')
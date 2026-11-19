"""Run data preparation, churn scoring, CLV estimation, and prepare dashboard inputs."""
import subprocess,sys
steps=[[sys.executable,"-m","src.run_part1"],[sys.executable,"-m","src.train_churn_model"],[sys.executable,"-m","src.predict_churn"],[sys.executable,"-m","src.clv_model"]]
for cmd in steps:
    print("\n>>> "," ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)
print("\nPipeline complete. Launch dashboard with: streamlit run app.py")

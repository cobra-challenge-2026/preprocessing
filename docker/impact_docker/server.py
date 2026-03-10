from fastapi import FastAPI
import subprocess

app = FastAPI()

@app.post("/run-registration")
def run_registration(fixed_path: str, moving_path: str, output_dir: str):

    cmd = [
        "/lib/elastix-install/bin/elastix",
        "-f", fixed_path,
        "-m", moving_path,
        "-p", "/code/configs/impact_params.txt",
        "-out", output_dir,
        "-threads", "12"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }
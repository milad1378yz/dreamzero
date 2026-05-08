import os
import sys
import time
import glob
import signal
import socket
import subprocess
from pathlib import Path

PORT = 5000
GPUS = "0,1"  # change to "0,1,2,3" if you want more
CKPT = Path("./checkpoints/DreamZero-DROID").resolve()

def run(cmd, **kw):
    print(" ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)

def wait_port(host="127.0.0.1", port=5000, timeout=900, process=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"Server exited before opening port {port}.")
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(3)
    raise RuntimeError("Server did not open port in time.")

def latest_mp4():
    files = glob.glob(str(CKPT.parent / "**/*.mp4"), recursive=True)
    return max(files, key=os.path.getmtime) if files else None

if not CKPT.exists():
    run([
        "hf", "download",
        "GEAR-Dreams/DreamZero-DROID",
        "--repo-type", "model",
        "--local-dir", str(CKPT),
    ])

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = GPUS
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

server_cmd = [
    sys.executable, "-m", "torch.distributed.run",
    "--standalone",
    f"--nproc_per_node={len(GPUS.split(','))}",
    "socket_test_optimized_AR.py",
    "--port", str(PORT),
    "--enable-dit-cache",
    "--attention-backend", "FA2",
    "--quantization", "bitsandbytes-int8",
    "--max-chunk-size", "1",
    "--model-path", str(CKPT),
]

server = subprocess.Popen(server_cmd, env=env)

try:
    wait_port(port=PORT, process=server)

    run([
        sys.executable,
        "test_client_AR.py",
        "--port", str(PORT),
        "--num-chunks", "4",
        "--prompt", "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    ])

    video = latest_mp4()
    print("\nGenerated video:", video)

    # In a Jupyter notebook this displays the video inline.
    try:
        from IPython.display import Video, display
        display(Video(video, embed=True))
    except Exception:
        pass

finally:
    server.send_signal(signal.SIGINT)
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        server.kill()

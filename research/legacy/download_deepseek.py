"""Download DeepSeek-R1-Distill-Qwen-7B to local Models directory.

Uses huggingface_hub snapshot_download for resumable transfer.
"""
import os
from huggingface_hub import snapshot_download

TARGET_DIR = "/home/kec23008/docker-sys/Models/DeepSeek-R1-Distill-Qwen-7B"
REPO_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

os.makedirs(TARGET_DIR, exist_ok=True)

print(f"Downloading {REPO_ID} -> {TARGET_DIR}")

path = snapshot_download(
    repo_id=REPO_ID,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=4,
    # Skip large optimizer files if any
    ignore_patterns=["*.gguf", "*.onnx"],
)

print(f"Download complete: {path}")

# Quick sanity check
files = os.listdir(TARGET_DIR)
print(f"\nFiles in {TARGET_DIR}:")
for f in sorted(files):
    p = os.path.join(TARGET_DIR, f)
    size = os.path.getsize(p) / 1e9 if os.path.isfile(p) else 0
    print(f"  {f} ({size:.2f} GB)" if size > 0.01 else f"  {f}")

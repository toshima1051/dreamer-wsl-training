import os
import sys

# Force CUDA backend before importing JAX/dreamerv3
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")

# Ensure cloned repo is on sys.path so configs.yaml resolves
REPO_DIR = "/home/ruku/source/dreamer/external/dreamerv3"
if REPO_DIR not in sys.path:
	sys.path.insert(0, REPO_DIR)

from dreamerv3 import main as dv3_main  # noqa: E402

if __name__ == "__main__":
	dv3_main.main()



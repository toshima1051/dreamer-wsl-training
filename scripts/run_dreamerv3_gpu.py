import os
import sys


def main() -> None:
	# Force CUDA before importing JAX / DreamerV3
	os.environ.setdefault("JAX_PLATFORMS", "cuda")
	os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
	# Hide ROCm devices to avoid accidental ROCm backend probing
	os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

	# Allow running this script from repo root or elsewhere
	repo_dir = "/home/ruku/source/dreamer/external/dreamerv3"
	if repo_dir not in sys.path:
		sys.path.insert(0, repo_dir)

	from dreamerv3 import main as dv3_main  # noqa: E402
	dv3_main.main()


if __name__ == "__main__":
	main()



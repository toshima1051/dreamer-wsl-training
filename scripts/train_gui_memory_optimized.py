#!/usr/bin/env python3
"""
GUI描画でメモリ節約設定で学習を実行するスクリプト
"""
import os
import sys


def main():
    # ROCmを無効化してCUDAを強制
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_PLATFORM_NAME"] = "cuda"
    os.environ["MUJOCO_GL"] = "glfw"
    
    # Allow running this script from repo root or elsewhere
    repo_dir = "/home/ruku/source/dreamer/external/dreamerv3"
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    
    # コマンドライン引数を設定（シェルスクリプトから渡された引数を使用）
    sys.argv = ["dreamerv3/main.py"] + sys.argv[1:]
    
    from dreamerv3 import main as dv3_main  # noqa: E402
    dv3_main.main()


if __name__ == "__main__":
    main()


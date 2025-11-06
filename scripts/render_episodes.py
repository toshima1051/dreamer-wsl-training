#!/usr/bin/env python3
"""
学習前後の動画を生成・比較するスクリプト

使い方:
  python scripts/render_episodes.py <logdir> [--checkpoint <ckpt_path>] [--num_episodes 5]
"""
import os
import sys
import argparse
from pathlib import Path

# CUDA固定（ROCm回避）
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")  # オフスクリーン描画

# DreamerV3 repoをパスに追加
repo_dir = "/home/ruku/source/dreamer/external/dreamerv3"
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from dreamerv3 import main as dv3_main  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Render episodes from DreamerV3 checkpoint")
    parser.add_argument("logdir", type=str, help="Log directory containing checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path (optional)")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to render")
    parser.add_argument("--task", type=str, default=None, help="Task name (auto-detect from logdir if not specified)")
    
    args = parser.parse_args()
    
    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Error: Log directory {logdir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # コマンドライン引数を構築
    cmd_args = [
        "--logdir", str(logdir),
        "--run.eval_every", "0",  # 評価は手動実行
        "--run.train_ratio", "0",  # 学習はしない
        "--jax.platform", "gpu",
    ]
    
    if args.checkpoint:
        cmd_args.extend(["--checkpoint", args.checkpoint])
    
    if args.task:
        cmd_args.extend(["--task", args.task])
    
    # DreamerV3のevalスクリプトを実行（動画生成）
    # 注意: DreamerV3のeval機能が動画保存に対応している場合
    print(f"Rendering {args.num_episodes} episodes from {logdir}")
    print("Note: Check Scope viewer at http://localhost:8000 for episode videos")
    print("      or look in logdir for video files if DreamerV3 saves them directly.")
    
    # 実際の動画生成はDreamerV3のeval機能に依存
    # ここではScopeビューアでの確認を推奨
    sys.argv = ["dreamerv3/main.py"] + cmd_args
    dv3_main.main()


if __name__ == "__main__":
    main()


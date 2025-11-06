#!/usr/bin/env python3
"""
学習前後のチェックポイントから動画を生成して比較するスクリプト

使い方:
  python scripts/generate_comparison_videos.py <logdir> [--early_ckpt <path>] [--late_ckpt <path>] [--num_episodes 5]
"""
import os
import sys
import argparse
from pathlib import Path

# CUDA固定（ROCm回避）
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_PLATFORM_NAME", "cuda")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")  # オフスクリーン描画（動画生成用）

# DreamerV3 repoをパスに追加
repo_dir = "/home/ruku/source/dreamer/external/dreamerv3"
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from dreamerv3 import main as dv3_main  # noqa: E402


def find_checkpoints(logdir):
    """ログディレクトリからチェックポイントを検索"""
    logdir_path = Path(logdir)
    ckpt_dir = logdir_path / "ckpt"
    
    if not ckpt_dir.exists():
        return [], []
    
    checkpoints = sorted(ckpt_dir.glob("*"), key=lambda x: x.stat().st_mtime)
    checkpoint_paths = [str(ckpt) for ckpt in checkpoints if ckpt.is_dir()]
    
    return checkpoint_paths


def generate_videos_from_checkpoint(logdir, checkpoint_path, output_dir, num_episodes=5, task=None):
    """チェックポイントから動画を生成"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating videos from checkpoint: {checkpoint_path}")
    print(f"  Output directory: {output_path}")
    print(f"  Number of episodes: {num_episodes}")
    
    # eval_onlyスクリプトで動画を生成
    cmd_args = [
        "--logdir", str(output_path),
        "--script", "eval_only",
        "--from_checkpoint", checkpoint_path,
        "--run.eval_every", "0",
        "--run.train_ratio", "0",
        "--run.length", str(num_episodes * 100),  # 十分な長さを確保
        "--jax.platform", "gpu",
    ]
    
    if task:
        cmd_args.extend(["--task", task])
    
    sys.argv = ["dreamerv3/main.py"] + cmd_args
    
    try:
        dv3_main.main()
        print(f"✅ Videos generated in {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error generating videos: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate comparison videos from checkpoints")
    parser.add_argument("logdir", type=str, help="Log directory containing checkpoints")
    parser.add_argument("--early_ckpt", type=str, default=None, help="Early checkpoint path (learning start)")
    parser.add_argument("--late_ckpt", type=str, default=None, help="Late checkpoint path (learning end)")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to render")
    parser.add_argument("--task", type=str, default=None, help="Task name")
    
    args = parser.parse_args()
    
    logdir = Path(args.logdir)
    if not logdir.exists():
        print(f"Error: Log directory {logdir} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # チェックポイントを検索
    checkpoints = find_checkpoints(logdir)
    
    if not checkpoints:
        print(f"Error: No checkpoints found in {logdir}/ckpt", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for i, ckpt in enumerate(checkpoints):
        print(f"  {i}: {Path(ckpt).name}")
    
    # 早期・後期チェックポイントを決定
    early_ckpt = args.early_ckpt or checkpoints[0]
    late_ckpt = args.late_ckpt or checkpoints[-1]
    
    print(f"\nUsing checkpoints:")
    print(f"  Early (before): {Path(early_ckpt).name}")
    print(f"  Late (after):   {Path(late_ckpt).name}")
    
    # 出力ディレクトリ
    output_base = logdir / "comparison_videos"
    early_output = output_base / "before"
    late_output = output_base / "after"
    
    # 動画を生成
    print("\n" + "="*60)
    print("Generating BEFORE videos (early checkpoint)...")
    print("="*60)
    success1 = generate_videos_from_checkpoint(
        logdir, early_ckpt, early_output, args.num_episodes, args.task
    )
    
    print("\n" + "="*60)
    print("Generating AFTER videos (late checkpoint)...")
    print("="*60)
    success2 = generate_videos_from_checkpoint(
        logdir, late_ckpt, late_output, args.num_episodes, args.task
    )
    
    if success1 and success2:
        print("\n" + "="*60)
        print("✅ Comparison videos generated successfully!")
        print("="*60)
        print(f"\nVideos saved in:")
        print(f"  Before: {early_output}")
        print(f"  After:  {late_output}")
        print(f"\nView in Scope viewer:")
        print(f"  bash /home/ruku/source/dreamer/scripts/start_scope_viewer.sh {logdir.parent} 8000")
    else:
        print("\n❌ Some videos failed to generate", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


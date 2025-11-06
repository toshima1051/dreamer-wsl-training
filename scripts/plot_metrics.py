#!/usr/bin/env python3
"""
学習メトリクスを可視化するスクリプト

使い方:
  python scripts/plot_metrics.py <logdir> [--output plot.png]
"""
import json
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # ヘッドレス環境用
import matplotlib.pyplot as plt


def load_metrics(logdir):
    """metrics.jsonlからメトリクスを読み込む"""
    metrics_file = Path(logdir) / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found", file=sys.stderr)
        return []
    
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def plot_learning_curve(metrics, output_path):
    """学習曲線をプロット"""
    if not metrics:
        print("No metrics found", file=sys.stderr)
        return
    
    # ステップ数とスコアを抽出
    steps = [m.get("step", 0) for m in metrics]
    scores = [m.get("episode/score", 0) for m in metrics]
    image_loss = [m.get("train/loss/image", 0) for m in metrics]
    policy_loss = [m.get("train/loss/policy", 0) for m in metrics]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # スコア
    axes[0].plot(steps, scores, label="Episode Score", alpha=0.7)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Learning Progress: Episode Score")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 損失
    axes[1].plot(steps, image_loss, label="Image Loss", alpha=0.7)
    axes[1].plot(steps, policy_loss, label="Policy Loss", alpha=0.7)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Losses")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot learning metrics")
    parser.add_argument("logdir", type=str, help="Log directory")
    parser.add_argument("--output", type=str, default="learning_curve.png", help="Output image path")
    
    args = parser.parse_args()
    
    metrics = load_metrics(args.logdir)
    if metrics:
        plot_learning_curve(metrics, args.output)
    else:
        print(f"No metrics found in {args.logdir}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()


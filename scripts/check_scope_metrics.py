#!/usr/bin/env python3
"""
Scopeビューアでメトリクスが表示されるか確認するスクリプト
"""
import json
import sys
from pathlib import Path


def check_logdir(logdir_path):
    """ログディレクトリの構造とメトリクスファイルを確認"""
    logdir = Path(logdir_path)
    
    if not logdir.exists():
        print(f"Error: {logdir} does not exist", file=sys.stderr)
        return False
    
    print(f"Checking logdir: {logdir}")
    print(f"  Exists: {logdir.exists()}")
    
    # メトリクスファイルを確認
    metrics_file = logdir / "metrics.jsonl"
    scores_file = logdir / "scores.jsonl"
    
    print(f"\nMetrics files:")
    print(f"  metrics.jsonl: {metrics_file.exists()}")
    print(f"  scores.jsonl: {scores_file.exists()}")
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            lines = [line for line in f if line.strip()]
            print(f"  metrics.jsonl lines: {len(lines)}")
            if lines:
                first = json.loads(lines[0])
                print(f"  First entry keys: {list(first.keys())[:10]}")
    
    # チェックポイントディレクトリ
    ckpt_dir = logdir / "ckpt"
    print(f"\nCheckpoints:")
    print(f"  ckpt/ exists: {ckpt_dir.exists()}")
    if ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("*"))
        print(f"  Checkpoint count: {len(ckpts)}")
    
    # Scopeが期待する構造か確認
    print(f"\nScope compatibility:")
    print(f"  Has metrics.jsonl: {metrics_file.exists()}")
    print(f"  Has scores.jsonl: {scores_file.exists()}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_scope_metrics.py <logdir>", file=sys.stderr)
        sys.exit(1)
    
    logdir = sys.argv[1]
    check_logdir(logdir)


if __name__ == "__main__":
    main()


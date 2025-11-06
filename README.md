# DreamerV3 Training Setup for WSL

DreamerV3の学習環境をWSL上でセットアップし、GUI描画で学習を実行するためのプロジェクトです。

## 概要

このリポジトリは、DreamerV3をWSL環境で実行するための設定とスクリプトを含んでいます。特に、X11転送を使用したGUI描画での学習をサポートしています。

## 主な機能

- WSL環境でのDreamerV3学習セットアップ
- GUI描画での学習実行（X11転送対応）
- メモリ最適化された学習スクリプト
- チェックポイントからの学習再開機能
- ROCmエラー回避のための設定

## セットアップ

詳細なセットアップ手順は以下のドキュメントを参照してください：

- [SETUP_DREAMERV3_WSL.md](SETUP_DREAMERV3_WSL.md) - 基本的なセットアップ手順
- [X11_SETUP.md](X11_SETUP.md) - X11転送の設定方法
- [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) - 完全なガイド

## 使用方法

### 新しい学習を開始

```bash
bash scripts/train_gui_memory_optimized.sh
```

### 既存のログから再開

```bash
bash scripts/train_gui_memory_optimized.sh \
  "$HOME/logdir/dreamer" \
  "dmc_cartpole_swingup" \
  "$HOME/logdir/dreamer/YYYYMMDD-HHMMSS"
```

## スクリプト

- `scripts/train_gui_memory_optimized.sh` - GUI描画でメモリ最適化された学習
- `scripts/train_with_gui.sh` - GUI描画での学習（基本版）
- `scripts/train_memory_efficient.sh` - オフスクリーン描画での学習
- `scripts/render_episodes.py` - エピソードのレンダリング
- `scripts/plot_metrics.py` - メトリクスのプロット

## 要件

- WSL2
- Python 3.11
- CUDA対応GPU（オプション）
- Xサーバー（VcXsrv等、GUI描画を使用する場合）

## ライセンス

このプロジェクトは、DreamerV3の学習環境セットアップ用のスクリプトとドキュメントを含んでいます。
DreamerV3自体のライセンスについては、`external/dreamerv3`を参照してください。


# Sample Training Logs

このディレクトリには、学習のサンプルログが含まれています。

## ファイル説明

- `metrics.jsonl` - 学習メトリクス（損失、FPSなど）の時系列データ
- `scores.jsonl` - エピソードスコアと長さのデータ
- `config.yaml` - 学習に使用された設定ファイル
- `videos/epstats-policy_image/` - エピソードごとのポリシー画像動画（学習過程の可視化）

## ログの可視化

これらのログファイルは、`scripts/plot_metrics.py`を使用して可視化できます：

```bash
python scripts/plot_metrics.py logs/sample/metrics.jsonl
```

## 動画について

`videos/epstats-policy_image/`ディレクトリには、学習中のエピソードを可視化した動画が含まれています。各動画は、エージェントが環境と相互作用している様子を示しています。

## 完全なログ

完全なログ（チェックポイント、リプレイバッファなど）は、`~/logdir/dreamer/`に保存されていますが、サイズが大きいためGitHubには含まれていません。


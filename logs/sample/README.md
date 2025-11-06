# Sample Training Logs

このディレクトリには、学習のサンプルログが含まれています。

## ファイル説明

- `metrics.jsonl` - 学習メトリクス（損失、FPSなど）の時系列データ
- `scores.jsonl` - エピソードスコアと長さのデータ
- `config.yaml` - 学習に使用された設定ファイル
- `videos/epstats-policy_image/` - エピソードごとのポリシー画像動画（学習過程の可視化、初期段階）
- `videos/epstats-policy_image_latest/` - 最新の学習結果の動画（Agent Step 47,552まで、スコア176.33まで向上）

## ログの可視化

これらのログファイルは、`scripts/plot_metrics.py`を使用して可視化できます：

```bash
python scripts/plot_metrics.py logs/sample/metrics.jsonl
```

## 動画について

### epstats-policy_image/
初期段階の学習動画（Agent Step 10,672〜49,232）が含まれています。学習の初期段階でのエージェントの動作を確認できます。

### epstats-policy_image_latest/
最新の学習結果の動画（Agent Step 10,672〜49,448）が含まれています。学習が進んだ後のエージェントの動作を確認できます。

**学習の進捗**:
- エピソードスコア: 15.03 → 101.79 → 158.76 → 176.33（向上）
- 画像損失: 223.57 → 4.93 → 3.46（改善）
- リプレイバッファ: 27.16 → 33.04（安定）

各動画は、エージェントが環境と相互作用している様子を示しています。学習が進むにつれて、エージェントの動作が改善していく様子が確認できます。

## 完全なログ

完全なログ（チェックポイント、リプレイバッファなど）は、`~/logdir/dreamer/`に保存されていますが、サイズが大きいためGitHubには含まれていません。


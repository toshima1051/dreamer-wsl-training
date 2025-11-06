# WSLでX11転送を使って動画を表示する完全ガイド

## 前提条件
- Windows側でXサーバー（VcXsrv等）を起動
- WSLでX11転送が有効になっている

## セットアップ手順

### 1. Windows側でXサーバーをインストール・起動

#### VcXsrvのインストール（推奨）
1. https://sourceforge.net/projects/vcxsrv/ からダウンロード
2. インストール後、XLaunchを起動
3. 設定:
   - Display number: 0
   - Client startup: Start no client
   - Extra settings: ✅ Disable access control（重要！）
4. 起動

#### または、WSLgを使用（Windows 11の場合）
- WSLgが有効なら、自動的にX11転送が機能します

### 2. WSL側でDISPLAY環境変数を設定

```bash
# ~/.bashrc または ~/.zshrc に追加
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
# または
export DISPLAY=:0
```

### 3. GUI描画で学習を実行

```bash
bash /home/ruku/source/dreamer/scripts/train_with_gui.sh
```

## 注意事項
- Xサーバーが起動していないと、GUI描画は失敗します
- オフスクリーン描画（osmesa/egl）の方が高速ですが、動画は真っ白になります
- GUI描画は遅いですが、動画が正しく表示されます


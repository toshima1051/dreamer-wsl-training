# GitHubリポジトリへのアップロード手順

## 1. GitHubでリポジトリを作成

1. GitHubにログインして、https://github.com/new にアクセス
2. リポジトリ名を入力（例: `dreamer-wsl-training`）
3. 説明を追加（オプション）
4. **Public** または **Private** を選択
5. **README、.gitignore、ライセンスは追加しない**（既にローカルにあるため）
6. 「Create repository」をクリック

## 2. リモートリポジトリを追加してプッシュ

GitHubでリポジトリを作成したら、以下のコマンドを実行してください：

```bash
cd /home/ruku/source/dreamer

# リモートリポジトリを追加（YOUR_USERNAMEとREPO_NAMEを置き換え）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# またはSSHを使用する場合
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# メインブランチをプッシュ
git push -u origin main
```

## 3. 認証

GitHubへのプッシュ時に認証が求められる場合：

- **HTTPSの場合**: Personal Access Tokenが必要です
  - Settings → Developer settings → Personal access tokens → Tokens (classic)
  - `repo`スコープでトークンを作成
  - プッシュ時にユーザー名とトークンを入力

- **SSHの場合**: SSH鍵を設定する必要があります
  - `ssh-keygen`で鍵を生成
  - GitHubのSettings → SSH and GPG keysに公開鍵を追加

## 4. 確認

プッシュが成功したら、GitHubのリポジトリページでファイルが表示されているか確認してください。

## 今後の更新

変更をコミットしてプッシュする場合：

```bash
git add .
git commit -m "Update: 変更内容の説明"
git push
```


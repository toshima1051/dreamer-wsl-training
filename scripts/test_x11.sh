#!/usr/bin/env bash
# X11転送が動作しているか確認するスクリプト

echo "Checking X11 forwarding..."

# DISPLAY環境変数を確認
if [ -z "${DISPLAY:-}" ]; then
    echo "⚠ DISPLAY not set, trying to detect..."
    export DISPLAY=$(cat /etc/resolv.conf 2>/dev/null | grep nameserver | awk '{print $2}'):0.0 || export DISPLAY=:0
fi

echo "DISPLAY: ${DISPLAY}"

# xeyesやxclockなどの簡単なXアプリでテスト
if command -v xeyes &> /dev/null; then
    echo "Testing with xeyes..."
    timeout 2 xeyes 2>&1 && echo "✅ X11 forwarding works!" || echo "❌ X11 forwarding failed"
elif command -v xclock &> /dev/null; then
    echo "Testing with xclock..."
    timeout 2 xclock 2>&1 && echo "✅ X11 forwarding works!" || echo "❌ X11 forwarding failed"
else
    echo "⚠ xeyes/xclock not found. Installing x11-apps..."
    sudo apt-get update -y && sudo apt-get install -y x11-apps
    echo "Please run this script again after installation."
fi

echo ""
echo "If X11 forwarding works, you can use GUI rendering with:"
echo "  bash /home/ruku/source/dreamer/scripts/train_with_gui.sh"


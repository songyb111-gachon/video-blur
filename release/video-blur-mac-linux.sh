#!/bin/sh
# video-blur one-file starter for macOS and Linux.  Run it in a terminal:  sh video-blur-mac-linux.sh
# It installs Python if it is missing, downloads the app into ~/.local/share/video-blur, lets the app's launch.py
# install the Python packages and ffmpeg, and opens the app in the browser.  Run it again any time to start the app.
# To remove everything, delete the folder ~/.local/share/video-blur.
VER="v1.1.0"
APP="${XDG_DATA_HOME:-$HOME/.local/share}/video-blur"
URL="https://raw.githubusercontent.com/songyb111-gachon/video-blur/$VER/launch.py"
mkdir -p "$APP" || exit 1
findpy() {
  PY=""
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then PY="$c"; return; fi
  done
}
findpy
if [ -z "$PY" ]; then
  echo "Python 3.9 이상을 찾지 못해 설치합니다."
  if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then brew install python
  elif command -v apt-get >/dev/null 2>&1; then sudo apt-get install -y python3 python3-venv python3-pip
  elif command -v dnf >/dev/null 2>&1; then sudo dnf install -y python3
  elif command -v pacman >/dev/null 2>&1; then sudo pacman -S --noconfirm python
  fi
  findpy
fi
if [ -z "$PY" ]; then echo "Python을 자동으로 설치하지 못했습니다. https://www.python.org/downloads/ 에서 설치한 뒤 다시 실행해 주세요."; exit 1; fi
if [ ! -f "$APP/launch.py" ] || [ ! -f "$APP/installed-$VER.txt" ]; then
  echo "앱을 내려받습니다."
  "$PY" -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$URL" "$APP/launch.py" || { echo "앱을 내려받지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요."; exit 1; }
fi
exec "$PY" "$APP/launch.py" --install "$VER" "$@"

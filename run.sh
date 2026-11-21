#!/bin/sh
# Starts the app. Installs Python first if it is missing; launch.py then takes care of the packages and ffmpeg.
cd "$(dirname "$0")" || exit 1
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
exec "$PY" launch.py "$@"

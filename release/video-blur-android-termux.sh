#!/data/data/com.termux/files/usr/bin/sh
# video-blur starter for Android phones. It runs inside the Termux app:
#   curl -L -o vb.sh https://github.com/songyb111-gachon/video-blur/releases/latest/download/video-blur-android-termux.sh && sh vb.sh
# It installs Python if needed, downloads the app into ~/video-blur, lets the app's launch.py install the rest with Termux's
# package manager, and opens the app in the phone's browser. Run "sh vb.sh" again any time to start the app.
# The "all people" mode is not available on a phone. To remove everything: rm -rf ~/video-blur ~/.cache/vblur
VER="v1.1.0"
[ -n "$VB_VER" ] && VER="$VB_VER"      # VB_VER=main tries the newest code instead of the released version
APP="$HOME/video-blur"
URL="https://raw.githubusercontent.com/songyb111-gachon/video-blur/$VER/launch.py"
command -v pkg >/dev/null 2>&1 || { echo "Termux 앱 안에서 실행해 주세요."; exit 1; }
if ! command -v python >/dev/null 2>&1; then
  echo "Python을 설치합니다."
  pkg update -y
  pkg install -y python || { echo "Python을 설치하지 못했습니다."; exit 1; }
fi
mkdir -p "$APP" || exit 1
if [ ! -f "$APP/launch.py" ] || [ ! -f "$APP/installed-$VER.txt" ] || [ "$VER" = main ]; then
  echo "앱을 내려받습니다."
  python -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$URL" "$APP/launch.py.new" && mv "$APP/launch.py.new" "$APP/launch.py" || { [ -f "$APP/launch.py" ] && [ "$VER" = main ] || { echo "앱을 내려받지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요."; exit 1; }; }
fi
exec python "$APP/launch.py" --install "$VER" "$@"

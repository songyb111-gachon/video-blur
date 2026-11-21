#!/usr/bin/env python3
"""launch.py - gets everything ready, then starts app.py.

Run it with any Python 3.9+ (run.bat and run.sh do that). It uses only the standard library and, whenever something is missing:
  - creates a private Python environment in .venv and installs the packages there
  - installs ffmpeg (Windows: winget, or a download into tools/ if winget is not there; macOS: Homebrew; Linux: the system package manager)
The model files are fetched later by the app itself, the first time a mode needs them."""
import os, sys, glob, shutil, zipfile, subprocess, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); WIN = os.name == 'nt'; MAC = sys.platform == 'darwin'
VENV = os.path.join(HERE, '.venv'); VPY = os.path.join(VENV, 'Scripts', 'python.exe') if WIN else os.path.join(VENV, 'bin', 'python')
TOOLS = os.path.join(HERE, 'tools', 'ffmpeg', 'bin')
FFMPEG_WIN_ZIP = 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip'   # one of the two Windows builds ffmpeg.org links to

def say(s): print(s, flush=True)
def die(s): say('\n' + s); sys.exit(1)
def run(cmd): return subprocess.run(cmd).returncode == 0
def ok(code): return subprocess.run([VPY, '-c', code], capture_output=True).returncode == 0

def venv():
    if os.path.exists(VPY) and ok('import sys'): return
    say('처음 실행이라 준비를 합니다. 인터넷 연결이 필요하고 몇 분 걸립니다.')
    shutil.rmtree(VENV, ignore_errors=True)
    if not run([sys.executable, '-m', 'venv', VENV]):
        if shutil.which('apt-get') and sys.stdin.isatty():      # Debian and Ubuntu ship venv as a separate package
            say('python3-venv 패키지를 설치합니다. 관리자 암호를 물을 수 있습니다.'); run(['sudo', 'apt-get', 'install', '-y', 'python3-venv'])
            shutil.rmtree(VENV, ignore_errors=True)
        if not os.path.exists(VPY) and not run([sys.executable, '-m', 'venv', VENV]):
            die('파이썬 가상 환경(.venv)을 만들지 못했습니다. 우분투라면 sudo apt install python3-venv 를 실행한 뒤 다시 켜 주세요.')
    run([VPY, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check', '--upgrade', 'pip'])

def packages():
    pip = [VPY, '-m', 'pip', 'install', '--disable-pip-version-check']; skip = os.path.join(VENV, 'people-install-failed')
    if not ok('import numpy, onnxruntime'):
        say('필요한 패키지를 설치합니다 (numpy, onnxruntime).')
        if not run(pip + ['numpy', 'onnxruntime']): die('패키지를 설치하지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요.')
    if not ok('import ultralytics') and not os.path.exists(skip) and not os.environ.get('VBLUR_NO_PEOPLE'):
        say('"모든 사람" 방식에 쓰는 패키지를 설치합니다 (ultralytics, torch). 용량이 커서 몇 분 걸립니다.')
        if sys.platform.startswith('linux'): run(pip + ['torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu'])   # the default Linux build is the multi-GB GPU one
        if not run(pip + ['ultralytics']):
            open(skip, 'w').close(); say('ultralytics를 설치하지 못했습니다. "모든 얼굴"과 "골라서" 방식은 그대로 쓸 수 있습니다. 다시 시도하려면 .venv 폴더를 지우고 실행하세요.')
    if not ok('import cv2'):      # ultralytics brings OpenCV with it; without it, or where that build cannot load (no display libraries), use the headless one
        say('필요한 패키지를 설치합니다 (opencv).')
        if not run(pip + ['opencv-python-headless']) or not ok('import cv2'): die('OpenCV를 설치하지 못했습니다. 위의 오류를 확인해 주세요.')

def ffmpeg_dirs():
    d = [TOOLS]; la = os.environ.get('LOCALAPPDATA', '')
    if WIN and la: d += [os.path.join(la, 'Microsoft', 'WinGet', 'Links')] + sorted({os.path.dirname(p) for p in glob.glob(os.path.join(la, 'Microsoft', 'WinGet', 'Packages', 'Gyan.FFmpeg*', '*', 'bin', 'ffmpeg.exe'))})
    return d
def have_ffmpeg():
    """True when ffmpeg and ffprobe can be started; folders this script installs into are put on PATH for the app"""
    for d in [None] + ffmpeg_dirs():
        path = os.environ.get('PATH', '') if d is None else d + os.pathsep + os.environ.get('PATH', '')
        if shutil.which('ffmpeg', path=path) and shutil.which('ffprobe', path=path): os.environ['PATH'] = path; return True
    return False

def download_ffmpeg_win():
    say('ffmpeg를 앱 폴더 안의 tools에 내려받습니다 (약 200MB).')
    os.makedirs(TOOLS, exist_ok=True); tmp = os.path.join(HERE, 'tools', 'ffmpeg.zip')
    try:
        with urllib.request.urlopen(urllib.request.Request(FFMPEG_WIN_ZIP, headers={'User-Agent': 'video-blur-app'}), timeout=60) as r, open(tmp, 'wb') as f:
            total = int(r.headers.get('Content-Length') or 0); got = 0; shown = -1
            while True:
                b = r.read(1 << 20)
                if not b: break
                f.write(b); got += len(b); p = got * 10 // total if total else -1
                if p != shown: shown = p; say(f'  {got >> 20} MB' + (f' / {total >> 20} MB' if total else ''))
        with zipfile.ZipFile(tmp) as z:
            for n in z.namelist():
                if n.lower().endswith(('/bin/ffmpeg.exe', '/bin/ffprobe.exe')):
                    with z.open(n) as a, open(os.path.join(TOOLS, os.path.basename(n)), 'wb') as b: shutil.copyfileobj(a, b)
    except Exception as e: say(f'내려받지 못했습니다: {e}')
    finally:
        if os.path.exists(tmp): os.remove(tmp)

def ffmpeg():
    if have_ffmpeg(): return
    say('ffmpeg가 없어 설치합니다.')
    if WIN:
        if shutil.which('winget'): run(['winget', 'install', '-e', '--id', 'Gyan.FFmpeg', '--accept-source-agreements', '--accept-package-agreements'])
        if not have_ffmpeg(): download_ffmpeg_win()
    elif MAC:
        if shutil.which('brew'): run(['brew', 'install', 'ffmpeg'])
    elif sys.stdin.isatty():
        for mgr, cmd in (('apt-get', ['sudo', 'apt-get', 'install', '-y', 'ffmpeg']), ('dnf', ['sudo', 'dnf', 'install', '-y', 'ffmpeg']), ('pacman', ['sudo', 'pacman', '-S', '--noconfirm', 'ffmpeg'])):
            if shutil.which(mgr): say('관리자 암호를 물을 수 있습니다: ' + ' '.join(cmd)); run(cmd); break
    if not have_ffmpeg():
        die('ffmpeg를 자동으로 설치하지 못했습니다. 직접 설치한 뒤 다시 실행해 주세요.\n  윈도우: winget install Gyan.FFmpeg\n  macOS: brew install ffmpeg\n  우분투: sudo apt install ffmpeg')

def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding='utf-8', errors='replace')
        except Exception: pass
    if sys.version_info < (3, 9): die(f'Python 3.9 이상이 필요합니다 (지금은 {sys.version.split()[0]}).')
    os.chdir(HERE); venv(); packages(); ffmpeg()
    try: sys.exit(subprocess.call([VPY, os.path.join(HERE, 'app.py')] + sys.argv[1:]))
    except KeyboardInterrupt: pass

if __name__ == '__main__': main()

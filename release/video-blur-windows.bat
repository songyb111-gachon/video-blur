@echo off
rem video-blur one-file starter for Windows.
rem Double-click it: it installs Python if this PC has none, downloads the app into %LocalAppData%\video-blur,
rem lets the app's launch.py install the Python packages and ffmpeg, and opens the app in the browser.
rem Run it again any time to start the app. To remove everything, delete the folder %LocalAppData%\video-blur.
chcp 65001 >nul
setlocal EnableExtensions
set "VER=v1.1.0"
set "APP=%LocalAppData%\video-blur"
set "URL=https://raw.githubusercontent.com/songyb111-gachon/video-blur/%VER%/launch.py"
if not exist "%APP%" mkdir "%APP%"
call :findpy
if defined PY goto havepy
echo Python을 찾지 못해 설치합니다. 몇 분 걸릴 수 있습니다.
where winget >nul 2>nul
if errorlevel 1 goto nopy
winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
call :findpy
if defined PY goto havepy
:nopy
echo.
echo Python을 자동으로 설치하지 못했습니다.
echo https://www.python.org/downloads/ 에서 Python 3을 설치한 뒤 이 파일을 다시 실행해 주세요.
echo 설치 화면에서 "Add python.exe to PATH"를 체크하세요.
pause
exit /b 1

:havepy
if exist "%APP%\launch.py" if exist "%APP%\installed-%VER%.txt" goto start
echo 앱을 내려받습니다.
"%PY%" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "%URL%" "%APP%\launch.py"
if errorlevel 1 goto nodl
:start
"%PY%" "%APP%\launch.py" --install %VER% %*
if errorlevel 1 pause
exit /b 0

:nodl
echo.
echo 앱을 내려받지 못했습니다. 인터넷 연결을 확인하고 다시 실행해 주세요.
pause
exit /b 1

:findpy
set "PY="
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY exit /b 0
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=py"
if defined PY exit /b 0
for %%V in (313 312 311 310 39) do if not defined PY if exist "%LocalAppData%\Programs\Python\Python%%V\python.exe" set "PY=%LocalAppData%\Programs\Python\Python%%V\python.exe"
for %%V in (313 312 311 310 39) do if not defined PY if exist "%ProgramFiles%\Python%%V\python.exe" set "PY=%ProgramFiles%\Python%%V\python.exe"
exit /b 0

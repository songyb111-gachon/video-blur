@echo off
rem Starts the app. Installs Python first if this PC has none; launch.py then takes care of the packages and ffmpeg.
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
call :findpy
if defined PY goto run
echo Python을 찾지 못해 설치합니다. 몇 분 걸릴 수 있습니다.
where winget >nul 2>nul
if errorlevel 1 goto nopy
winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
call :findpy
if defined PY goto run
:nopy
echo.
echo Python을 자동으로 설치하지 못했습니다.
echo https://www.python.org/downloads/ 에서 Python 3을 설치한 뒤 이 파일을 다시 실행해 주세요.
echo 설치 화면에서 "Add python.exe to PATH"를 체크하세요.
pause
exit /b 1

:run
"%PY%" launch.py %*
if errorlevel 1 pause
exit /b 0

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

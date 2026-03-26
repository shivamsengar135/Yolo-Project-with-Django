@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Missing virtual environment at .venv\Scripts\python.exe
  echo Create it first with:
  echo   python -m venv .venv
  echo   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

echo Installing/Updating PyInstaller...
".venv\Scripts\python.exe" -m pip install --upgrade pyinstaller
if errorlevel 1 (
  echo Failed to install PyInstaller.
  pause
  exit /b 1
)

echo Generating app icon...
".venv\Scripts\python.exe" scripts\create_app_icon.py
if errorlevel 1 (
  echo Failed to generate icon.
  pause
  exit /b 1
)

echo Building single-file executable...
".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name "ObjectDetectorApp" ^
  --icon "assets\app_icon.ico" ^
  "src\app_gui.py"
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo.
echo Build complete.
echo EXE: dist\ObjectDetectorApp.exe
echo.
echo Tip: Keep this EXE in the project root so it can find .venv and src.
pause
exit /b 0

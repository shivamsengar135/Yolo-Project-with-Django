@echo off
setlocal
cd /d "%~dp0"

if not exist "dist\ObjectDetectorApp.exe" (
  echo Missing EXE: dist\ObjectDetectorApp.exe
  echo Build first using:
  echo   build_exe.bat
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$W = New-Object -ComObject WScript.Shell; " ^
  "$Desktop = [Environment]::GetFolderPath('Desktop'); " ^
  "$Shortcut = $W.CreateShortcut((Join-Path $Desktop 'Object Detector App.lnk')); " ^
  "$Shortcut.TargetPath = (Resolve-Path 'dist\ObjectDetectorApp.exe').Path; " ^
  "$Shortcut.WorkingDirectory = (Resolve-Path '.').Path; " ^
  "$Shortcut.IconLocation = (Resolve-Path 'assets\app_icon.ico').Path; " ^
  "$Shortcut.Save();"

if errorlevel 1 (
  echo Failed to create desktop shortcut.
  pause
  exit /b 1
)

echo Desktop shortcut created: Object Detector App.lnk
echo Right-click the shortcut and choose "Pin to taskbar" or "Pin to Start".
pause
exit /b 0

@echo off
rem start "Refl1d Web GUI" 
call "%~dp0env\Scripts\activate.bat"
start "Refl1D Webview" "python.exe" "-m" "refl1d.webview.server"
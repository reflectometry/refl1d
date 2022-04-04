rem start "Refl1d CLI in PowerShell" 
start PowerShell -noexit -Command "Function refl1d() {&'%~dp0\python.exe' '%~dp0\Scripts\refl1d_cli.py' $args}"

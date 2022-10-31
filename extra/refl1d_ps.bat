@REM start powershell.exe -noexit -Command "Function refl1d() {&'C:\Users\bbm\Downloads\Refl1D windows exe latest (NIST)\Refl1D-0.8.14\python.exe' 'C:\Users\bbm\Downloads\Refl1D windows exe latest (NIST)\Refl1D-0.8.14\Scripts\refl1d_cli.py' $args}" 
@REM start powershell.exe -noexit -Command "Function refl1d() {&'%~dp0\python.exe' '%~dp0\Scripts\refl1d_cli.py' $args}"
@REM start PowerShell -noexit -Command "Function refl1d() {&'%~dp0\python.exe' '%~dp0\Scripts\refl1d_cli.py' $args}"
@REM start PowerShell -noexit -Command "$Env:Path = '%~dp0;%~dp0\Scripts;%Path%'; $Env:PYTHONPATH='"
set "PATH=%~dp0;%~dp0\Scripts;%PATH%"
start PowerShell -noexit

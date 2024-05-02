@echo off

rem start "Refl1d Jupyter Session"
start "Refl1D Jupyter Server" "%~dp0\env\python.exe" "-c" "from tkinter import filedialog as fd; import jupyterlab.labapp as lab; start_folder = fd.askdirectory(); lab.main(argv=[f'--notebook-dir={start_folder}'])"
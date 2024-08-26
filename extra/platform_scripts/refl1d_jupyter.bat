@echo off

rem start "Refl1d Jupyter Session"
call "%~dp0env\Scripts\activate.bat"
start "Refl1D Jupyter Server" "python.exe" "-c" "from tkinter import filedialog as fd; import jupyterlab.labapp as lab; start_folder = fd.askdirectory(title='Choose working folder for Jupyter session'); lab.main(argv=[f'--notebook-dir={start_folder}'])"

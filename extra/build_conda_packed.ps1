$ENV_NAME="isolated-base"
$PYTHON_VERSION="3.10"
$DIRNAME="refl1d"

conda activate "base"

conda install -y conda-pack
conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION"
conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"

# unpack the new environment, that contains only python + pip
$destdir="$DIRNAME"
Remove-Item -r "$DIRNAME"
mkdir "$destdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$destdir"

# activate the unpacked environment and install pip packages
conda deactivate
$WORKING_DIRECTORY="$pwd"
pwd
dir ..
dir ...
& "$destdir\python.exe" -m pip install --no-input numba
& "$destdir\python.exe" -m pip install --no-input git+https://github.com/bumps/bumps@webview
& "$destdir\python.exe" -m pip install --no-input git+https://github.com/reflectometry/refl1d@webview
& "$destdir\python.exe" -m pip install -r https://raw.githubusercontent.com/bumps/bumps/webview/webview-requirements

# add our batch script:
Copy-Item .\extra\refl1d_webview.bat .\refl1d\

# zip it back up
tar -czf "$WORKING_DIRECTORY/refl1d-webview-Windows-x86_64.tar.gz" "$DIRNAME"

$ENV_NAME="isolated-base"
$PYTHON_VERSION="3.10"
$DIRNAME="refl1d"

conda activate "base"

conda install -y conda-pack
conda env remove -n "$ENV_NAME"
conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION"
conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"

# unpack the new environment, that contains only python + pip
$tmpdir = Join-Path $Env:Temp "jupyter_webview_build"
$destdir="$tmpdir\$DIRNAME"
Remove-Item -r "$destdir"
mkdir "$destdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$destdir"

# activate the unpacked environment and install pip packages
conda deactivate
$WORKING_DIRECTORY="$pwd"
echo "WORKING_DIRECTORY=$WORKING_DIRECTORY"
dir .
dir ..
# add our batch script:
Copy-Item .\extra\refl1d_webview.bat "$destdir"

& "$destdir\python.exe" -m pip install --no-input --no-compile numba jupyterlab notebook
& "$destdir\python.exe" -m pip install --no-input --no-compile git+https://github.com/bumps/bumps@webview
& "$destdir\python.exe" -m pip install --no-input --no-compile git+https://github.com/reflectometry/refl1d@webview
& "$destdir\python.exe" -m pip install --no-compile -r https://raw.githubusercontent.com/bumps/bumps/webview/webview-requirements

$version=$(& "$destdir\python.exe" -c "import refl1d; print(refl1d.__version__)")
# zip it back up
cd $tmpdir
Rename-Item "$DIRNAME" "$DIRNAME-$version"
tar -czf "refl1d-webview-jupyter-Windows-x86_64.tar.gz" "$DIRNAME-$version"

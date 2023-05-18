$ENV_NAME="isolated-base"
$PYTHON_VERSION="3.10"
$DIRNAME="Refl1D-webview"

conda activate "base"

conda install -y conda-pack
conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION"
conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"

# unpack the new environment, that contains only python + pip
$destdir="$DIRNAME"
Remove-Item -r "$DIRNAME" -ErrorAction SilentlyContinue
mkdir "$destdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$destdir"

# activate the unpacked environment and install pip packages
conda deactivate
$WORKING_DIRECTORY="$pwd"
& "$destdir\python.exe" -m pip install --no-input --no-compile numba
& "$destdir\python.exe" -m pip install --no-input --no-compile git+https://github.com/bumps/bumps@webview
& "$destdir\python.exe" -m pip install --no-input --no-compile git+https://github.com/reflectometry/refl1d@webview
& "$destdir\python.exe" -m pip install --no-compile -r https://raw.githubusercontent.com/bumps/bumps/webview/webview-requirements

# add our batch script:
Copy-Item "$PSScriptRoot\refl1d_webview.bat" "$destdir"

# zip it back up
mkdir "$WORKING_DIRECTORY\dist" -ErrorAction SilentlyContinue
tar -czf "$WORKING_DIRECTORY\dist\Refl1D-webview-Windows-x86_64.tar.gz" "$DIRNAME"

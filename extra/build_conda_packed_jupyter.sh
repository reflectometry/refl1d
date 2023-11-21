#!/bin/bash

ENV_NAME="isolated-base"
PYTHON_VERSION="3.10"
DIRNAME="refl1d"

eval "$(conda shell.bash hook)"
conda activate base || { echo 'failed: conda not installed'; exit 1; }

conda install -y conda-pack
conda remove -n "$ENV_NAME" -y --all
conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION"
conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"

# unpack the new environment, that contains only python + pip
tmpdir=$(mktemp -d)
destdir="$tmpdir/$DIRNAME"
envdir="$destdir/env"
mkdir -p "$envdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$envdir"

# activate the unpacked environment and install pip packages
conda deactivate
WORKING_DIRECTORY=$(pwd)
# add our batch scripts:
case $OSTYPE in 
  darwin*) cp -r ./extra/refl1d_webview.app "$destdir" && \
           cp -r ./extra/refl1d_jupyter.app "$destdir" ;; 
esac

$envdir/bin/python -m pip install --no-input --no-compile numba jupyterlab notebook
$envdir/bin/python -m pip install --no-input --no-compile git+https://github.com/bumps/bumps@webview
$envdir/bin/python -m pip install --no-input --no-compile git+https://github.com/reflectometry/refl1d@webview
$envdir/bin/python -m pip install --no-compile -r https://raw.githubusercontent.com/bumps/bumps/webview/webview-requirements

version=$($envdir/bin/python -c "import refl1d; print(refl1d.__version__)")
mv "$tmpdir/$DIRNAME" "$tmpdir/$DIRNAME-$version"

case $OSTYPE in 
  # darwin*) cd $tmpdir && hdiutil create -srcfolder  "$DIRNAME-$version" -volname "Refl1D_Jupyter" "$WORKING_DIRECTORY/Refl1D_Jupyter.dmg" ;; 
  darwin*) pkgbuild --root $tmpdir --identifier org.reflectometry.refl1d-webview-jupyter --version $version --ownership preserve --install-location /Applications refl1d-webview-jupyter-$(uname -s)-$(uname -m).pkg ;;
esac

cd $tmpdir && tar -czf "$WORKING_DIRECTORY/refl1d-webview-jupyter-$version-$(uname -s)-$(uname -m).tar.gz" "$DIRNAME-$version"
rm -rf $tmpdir

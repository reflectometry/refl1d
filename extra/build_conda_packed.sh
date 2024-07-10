#!/bin/bash

ENV_NAME="isolated-base"
PYTHON_VERSION="3.10"
DIRNAME="refl1d"

eval "$(conda shell.bash hook)"
conda activate base || { echo 'failed: conda not installed'; exit 1; }

conda install -y conda-pack
conda remove -n "$ENV_NAME" -y --all
conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION" "nodejs"
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
# add our batch script:
case $OSTYPE in darwin*) cp -r ./extra/platform_scripts/refl1d_webview.app "$destdir" ;; esac
case $OSTYPE in 
  linux*) cp -r ./extra/platform_scripts/make_linux_desktop_shortcut.sh "$destdir" && \
          cp -r ./extra/platform_scripts/refl1d-webview "$destdir" ;;
esac

$envdir/bin/python -m pip install --no-input --no-compile numba
$envdir/bin/python -m pip install --no-input --no-compile git+https://github.com/bumps/bumps@webview
$envdir/bin/python -m pip install --no-input --no-compile git+https://github.com/reflectometry/refl1d@webview
$envdir/bin/python -m pip install --no-compile -r https://raw.githubusercontent.com/bumps/bumps/webview/webview-requirements

# build the client
cd $envdir/lib/python$PYTHON_VERSION/site-packages/bumps/webview/client
$envdir/bin/npm install
$envdir/bin/npm link

cd $envdir/lib/python$PYTHON_VERSION/site-packages/refl1d/webview/client
$envdir/bin/npm link ../../../bumps/webview/client
$envdir/bin/npm install
$envdir/bin/npm run build
rm -rf $envdir/lib/python$PYTHON_VERSION/site-packages/bumps/webview/client/node_modules
rm -rf $envdir/lib/python$PYTHON_VERSION/site-packages/refl1d/webview/client/node_modules

version=$($envdir/bin/python -c "import refl1d; print(refl1d.__version__)")
mv "$tmpdir/$DIRNAME" "$tmpdir/$DIRNAME-$version"

case $OSTYPE in 
  # darwin*) cd $tmpdir && hdiutil create -srcfolder  "$DIRNAME-$version" -volname "Refl1D_Jupyter" "$WORKING_DIRECTORY/Refl1D_Jupyter.dmg" ;; 
  darwin*) pkgbuild --root $tmpdir --identifier org.reflectometry.refl1d-webview --version $version --ownership preserve --install-location /Applications refl1d-webview-$(uname -s)-$(uname -m).pkg ;;
esac

cd $tmpdir && tar -czf "$WORKING_DIRECTORY/refl1d-webview-$version-$(uname -s)-$(uname -m).tar.gz" "$DIRNAME-$version"
rm -rf $tmpdir

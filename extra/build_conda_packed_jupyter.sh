#!/bin/bash

ENV_NAME="isolated-base"
PYTHON_VERSION="3.10"
PKGNAME="refl1d"
SUBNAME="webview-jupyter"
OUTPUT="artifacts"
WORKING_DIRECTORY=$(pwd)

mkdir -p $OUTPUT

eval "$(conda shell.bash hook)"
conda activate base || { echo 'failed: conda not installed'; exit 1; }

conda install -y conda-pack
if ! test -f "$ENV_NAME.tar.gz"; then
  echo "creating isolated environment"
  conda remove -n "$ENV_NAME" -y --all
  conda create -n "$ENV_NAME" -q --force -y "python=$PYTHON_VERSION" "nodejs"
  conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"
fi

# unpack the new environment, that contains only python + pip
tmpdir=$(mktemp -d)
pkgdir="$tmpdir/$PKGNAME"
envdir="$pkgdir/env"
mkdir -p "$envdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$envdir"

# activate the unpacked environment and install pip packages
# add our batch script:
case $OSTYPE in 
  darwin*) cp -r ./extra/platform_scripts/refl1d_webview.app "$pkgdir" ;
           cp -r ./extra/platform_scripts/refl1d_jupyter.app "$pkgdir" ;;
  msys*) cp ./extra/platform_scripts/refl1d_webview.bat "$pkgdir" ;
         cp ./extra/platform_scripts/refl1d_jupyter.bat "$pkgdir" ;;
  linux*) cp -r ./extra/platform_scripts/make_linux_desktop_shortcut.sh "$pkgdir" ;
          cp -r ./extra/platform_scripts/refl1d-webview "$pkgdir" ;
          cp -r ./extra/platform_scripts/refl1d-jupyter "$pkgdir" ;;
esac

case "$OSTYPE" in
 "msys") bindir=$envdir ;
         sitepackages=$envdir/lib/site-packages ;
         platform="Windows";;
 *) bindir=$envdir/bin ;
    sitepackages=$envdir/lib/python$PYTHON_VERSION/site-packages
    platform="$(uname -s)";;
esac


$bindir/python -m pip install --no-input --no-compile numba jupyterlab notebook
$bindir/python -m pip install --no-input --no-compile git+https://github.com/bumps/bumps@webview
$bindir/python -m pip install --no-input --no-compile git+https://github.com/reflectometry/refl1d@webview
$bindir/python -m pip install --no-compile -r https://raw.githubusercontent.com/bumps/bumps/webview/webview-requirements
$bindir/python -m pip install orsopy

# build the client
cd $sitepackages/bumps/webview/client
$bindir/npm install

cd $sitepackages/refl1d/webview/client
$bindir/npm link ../../../bumps/webview/client
$bindir/npm install
$bindir/npm run build

cd $tmpdir

rm -rf $sitepackages/refl1d/webview/client/node_modules
rm -rf $sitepackages/bumps/webview/client/node_modules

version=$($bindir/python -c "import refl1d; print(refl1d.__version__)")
mv "$tmpdir/$PKGNAME" "$tmpdir/$PKGNAME-$version"

case $OSTYPE in 
  # darwin*) cd $tmpdir && hdiutil create -srcfolder  "$PKGNAME-$version" -volname "Refl1D_Jupyter" "$WORKING_DIRECTORY/Refl1D_Jupyter.dmg" ;; 
  darwin*) pkgbuild --root $tmpdir --identifier org.reflectometry.$PKGNAME-$SUBNAME --version $version --ownership preserve --install-location /Applications "$WORKING_DIRECTORY/$OUTPUT/$PKGNAME-$SUBNAME-$version-$platform-$(uname -m).pkg" ;;
  msys*) conda install -y 7zip ;
         curl -L https://www.7-zip.org/a/7z2106-x64.exe --output 7z_exe ;
         7z e 7z_exe -aoa 7z.sfx ;
         7z a -mhe=on -mx=1 -sfx".\7z.sfx" "$WORKING_DIRECTORY/$OUTPUT/$PKGNAME-$SUBNAME-$version-$platform-$(uname -m)-self-extracting.exe" "$PKGNAME-$version";;
esac

cd $tmpdir && tar -czf "$WORKING_DIRECTORY/$OUTPUT/$PKGNAME-$SUBNAME-$version-$platform-$(uname -m).tar.gz" "$PKGNAME-$version"
cd $WORKING_DIRECTORY
rm -rf $tmpdir
#!/bin/bash

ENV_NAME="refl1d-webview-packed"
PYTHON_VERSION="3.10"
DIRNAME="refl1d"

eval "$(conda shell.bash hook)"
conda activate base || { echo 'failed: conda not installed'; exit 1; }

conda install -y conda-pack
conda create -n "$ENV_NAME" -y "python=$PYTHON_VERSION"
conda-pack -n "$ENV_NAME" -f -o "$ENV_NAME.tar.gz"

# unpack the new environment, that contains only python + pip
tmpdir=$(mktemp -d)
destdir="$tmpdir/$DIRNAME"
mkdir "$destdir"
tar -xzf "$ENV_NAME.tar.gz" -C "$destdir"

# activate the unpacked environment and install pip packages
WORKING_DIRECTORY=$(pwd)
cd "$tmpdir"
source "$DIRNAME/bin/activate"
pip install numba
pip install git+https://github.com/bumps/bumps@dataclass_overlay
pip install git+https://github.com/reflectometry/refl1d@webview
pip install -r https://raw.githubusercontent.com/reflectometry/refl1d/webview/webview-requirements

# zip it back up
zip -y -r "$WORKING_DIRECTORY/refl1d-webview-$(uname -s)-$(uname -m).zip" "$DIRNAME"




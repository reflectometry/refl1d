# Install the application in the embedded python for windows distribution.
# This script fetches the embedded distribution from python.org and installs
# pip. It then uses pip to install the binary wheel from the dist directory.
# Remove dist\*.whl and run "pythoon setup.py bdist_wheel" before running.

# TODO: Use the PYTHON environment variable to infer the version of python.
# Look in https://www.python.org/ftp/python for the latest python version.
$PY_VERSION = "3.8.1"
$PTH_FILE = "python38._pth"
$WHEEL_TAG = "cp38-cp38m-win_amd64"
$INSTALL_TAG = "cp38m-embedded-amd64"
$PACKAGE = "refl1d"
$APP_NAME = "Refl1D"

# Grab version number from the application source.  Assumes it is in the
$APP_VERSION = (Select-String '__version__ *= *"(.*)"' "$PACKAGE\__init__.py" -ca).matches | Select-Object -ExpandProperty Groups | Select-Object -Last 1 -ExpandProperty value

# Fetch embedded python zip file
$AllProtocols = [System.Net.SecurityProtocolType]'Ssl3,Tls,Tls11,Tls12'
[System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PY_VERSION/python-$PY_VERSION-embed-amd64.zip" -OutFile python_embedded.zip
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile get-pip.py

# Expand the zip file into the build directory where it won't collide with the
# package source directory.  Note: The directory name $APP_NAME will appear on
# the users machine when they expand the zipped distribution we are creating.
Expand-Archive -path '.\python_embedded.zip' -destinationpath "build\$APP_NAME"

# Enable pip in the embedded python distribution
cd "build\$APP_NAME"
(Get-Content $PTH_FILE).replace('#import site', 'import site') | Set-Content $PTH_FILE
.\python.exe "..\..\get-pip.py"

# Use pip to install the app from the binary wheel in the dist directory.
cd ..
& "$APP_NAME\Scripts\pip.exe" install "$PSScriptPath\dist\$PACKAGE-$APP_VERSION-$WHEEL_TAG.whl"

# Create the embedded app archive in the dist directory .
Compress-Archive -Path "$APP_NAME" -DestinationPath "$PSScriptPath\dist\$APP_NAME-$APP_VERSION-$INSTALL_TAG.zip"

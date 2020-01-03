# Install the application in the embedded python for windows distribution.
# This script fetches the embedded distribution from python.org and installs
# pip. It then uses pip to install the binary wheel from the dist directory.
# Remove dist\*.whl and run "pythoon setup.py bdist_wheel" before running.

# TODO: Use the PYTHON environment variable to infer the version of python.
# Look in https://www.python.org/ftp/python for the latest python version.
$PY_VERSION = "3.8.1"
$PTH_FILE = "python38._pth"
$WHEEL_TAG = "cp38-cp38-win_amd64"
$INSTALL_TAG = "cp38-embedded-amd64"
$PACKAGE = "refl1d"
$APP_NAME = "Refl1D"

# Grab version number from the application source.  Assumes it is defined as
#     package\__init__.py:__version__ = "..."
$APP_VERSION = (Select-String '__version__ *= *"(.*)"' "$PACKAGE\__init__.py" -ca).matches | Select-Object -ExpandProperty Groups | Select-Object -Last 1 -ExpandProperty value
#Write-Host "Building application $APP_NAME-$APP_VERSION"

# Fetch embedded python zip file
#Write-Host "Fetching python"
$AllProtocols = [System.Net.SecurityProtocolType]'Ssl3,Tls,Tls11,Tls12'
[System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$PY_VERSION/python-$PY_VERSION-embed-amd64.zip" -OutFile python_embedded.zip
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile get-pip.py

# Expand the zip file into the build directory where it won't collide with the
# package source directory.  Note: The directory name $APP_NAME will appear on
# the users machine when they expand the zipped distribution we are creating.
Write-Host "Preparing embedded environment"
Expand-Archive -path 'python_embedded.zip' -destinationpath "build\$APP_NAME"

# Enable pip in the embedded python distribution
$PTH_PATH = "build\$APP_NAME\$PTH_FILE"
(Get-Content "$PTH_PATH").replace('#import site', 'import site') | Set-Content "$PTH_PATH"
& "build\$APP_NAME\python.exe" "get-pip.py"

# Use pip to install the app from the binary wheel in the dist directory.
Write-Host "Installing $APP_NAME-$APP_VERSION into embedded environment"
& "build\$APP_NAME\Scripts\pip.exe" install --no-warn-script-location "$PSScriptRoot\..\dist\$PACKAGE-$APP_VERSION-$WHEEL_TAG.whl"

# Create the embedded app archive in the dist directory .
cd build
$ZIP_PATH = "dist\$APP_NAME-$APP_VERSION-$INSTALL_TAG.zip"
#Write-Host "Compressing into $ZIP_PATH"
Compress-Archive -Path "$APP_NAME" -DestinationPath "$PSScriptRoot\..\$ZIP_PATH"

Write-Host "All done!!  Application is in $ZIP_PATH"

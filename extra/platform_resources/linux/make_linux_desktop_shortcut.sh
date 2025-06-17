#!/bin/bash

# This script creates a desktop shortcut for the Refl1d webview server.
# The shortcut will be created in the user's desktop directory.
# the executable path is in 'env/bin/python'

# Get the user's desktop directory
desktop_dir=~/Desktop

# Get the path to the bumps webview server
script_dir=$(realpath $(dirname $0))

# Create the desktop shortcut
echo "[Desktop Entry]
Name=Refl1d-Webview
Comment=Start the refl1d webview server
Exec='$script_dir/env/bin/python' -m refl1d.webview.server --use-persistent-path
Icon=$script_dir/env/share/icons/refl1d-icon.svg
Terminal=true
Type=Application
Categories=Development;
" > $desktop_dir/Refl1dWebviewServer.desktop

# Make the desktop shortcut executable
chmod +x $desktop_dir/Refl1dWebviewServer.desktop

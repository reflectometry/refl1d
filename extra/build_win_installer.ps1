$AllProtocols = [System.Net.SecurityProtocolType]'Ssl3,Tls,Tls11,Tls12'
[System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.7.4/python-3.7.4-embed-amd64.zip" -OutFile python_embedded.zip
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile get-pip.py

Expand-Archive -path '.\python_embedded.zip' -destinationpath 'build\Refl1D'
cd 'build\Refl1D'

(Get-Content python37._pth).replace('#import site', 'import site') | Set-Content python37._pth
.\python.exe ..\..\get-pip.py
.\Scripts\pip.exe install Refl1D

cd ..
$VERSION = (Select-String '__version__ *= *"(.*)"' ..\refl1d\__init__.py -ca).matches | Select -ExpandProperty Groups | Select -Last 1 -ExpandProperty value
Compress-Archive -Path Refl1D -DestinationPath '..\dist\Refl1D-$VERSION.zip'

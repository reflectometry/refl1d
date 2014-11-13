#!/bin/sh

cat <<EOF
Refl1D Release Process

(0) release dependent packages (bumps, periodictable) if necessary

# for reproducible builds we want a fixed bumps version number, but for
# users who are living without virtualenv, we want to be relaxed about
# the version number.  Maybe separate pip requirements from

(1) check that all changes have been pushed
  [bumps, refl1d, periodictable] X [windows, mac, linux boxen]
   # Note: linux boxen includes ~/dev as well as ~/src on the shared machines

git status

(2) check that all tests pass
  [bumps, refl1d, periodictable] X [windows, mac, linux] X [python 2.6, 2.7, 3.3, 3.4]

git pull
pythonX.Y tests.py
pythonX.Y check_examples.py --chisq

(3) update change log in CHANGES.rst

(cd ../periodictable && git log)
(cd ../bumps && git log)
git log

(4) update version number and requirements and tag release

vi refl1d/__init__.py
vi setup.py
vi rtd-requirements
git commit -a -m "RX.Y.Z"
git tag -a vX.Y.Z -m "Refl1d X.Y.Z"
git push origin vX.Y.Z

(5) build the docs

# even though this should happen automatically in jenkins,
# build the docs locally and examine the error messages that come out
(cd doc && make clean html pdf)

# trigger the build on readthedocs
https://readthedocs.org/builds/refl1d/

(5) build release packages and upload to release server

# This should happen automatically on jenkins & readthedocs.
# Note: don't use anaconda to build release version unless we update all
# web pages to say "built with anaconda"; as convenient as it is, we don't
# want to run afoul of license agreements by missing a page.

[windows]
python setup_py2exe.py
scp dist/refl1d-X.Y.Z-install.exe reflectometry.org:web/danse/downloads

[mac]
python setup_py2app.py
scp dist/refl1d-X.Y.Z.dmg reflectometry.org:web/danse/downloads

(6) update pypi with new release version

python setup.py sdist upload

(7) check that the new pypi version runs (single machine should be fine)

# create virtualenv
cd ~
conda create -n piptest pip numpy matplotlib scipy wxpython pyparsing
activate piptest
pip install refl1d
python -m refl1d.main
pythonw -m refl1d.main --edit
# would be nice to run tests against installed version, but not yet possible
deactivate piptest
rm -r ~/anaconda/envs/piptest

(8) update shared resources with new versions

ssh sparkle  // shimmer
cd ~/src/periodictable && git pull
cd ~/src/bumps && git pull
cd ~/src/refl1d && git pull && python setup.py build

ssh rocks
pip install --user --no-deps periodictable bumps refl1d

(9) announce release

update reflectometry.org web pages
send message to reflectometry announcement list

EOF

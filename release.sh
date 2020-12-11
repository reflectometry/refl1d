#!/bin/sh

cat <<EOF
Refl1D Release Process

(0) release dependent packages (bumps, periodictable) if necessary

# for reproducible builds we want a fixed bumps version number, but for
# users who are living without virtualenv, we want to be relaxed about
# the version number.  Maybe separate pip requirements from binary
# distribution requirements?  Or maybe don't worry about it.

(1) check that all changes have been pushed
  [bumps, refl1d, periodictable] X [windows, mac, linux boxen]
   # Note: linux boxen includes ~/dev as well as ~/src on the shared machines
sh
git status
git pull
git push

(2) check that all tests pass
  [bumps, refl1d, periodictable] X [windows, mac, linux] X [python 2.6, 2.7, 3.3, 3.4]

... look for build badge on github page ...

python setup.py test
python check_examples.py --chisq

(3) update change log in CHANGES.rst

(cd ../periodictable && git log)
(cd ../bumps && git log)
git log

(4) build the docs

# even though this should happen automatically in jenkins we
# do the builds by hand to make sure they are error free.
(cd doc && make clean html pdf)

(5) update version number and requirements

REFLVER=X.Y.Z
vi refl1d/__init__.py
vi rtd-requirements
git commit -a -m "R $REFLVER"
git push

(6) wait for the binaries for windows and mac

# again, release is on jenkins, but this allows us to check the build
# Note: don't use anaconda to build release version unless we update all
# web pages to say "built with anaconda"; as convenient as it is, we don't
# want to run afoul of license agreements by missing a page.
#[windows, mac]
#python master_builder.py

(7) tag release

git tag -a v$REFLVER -m "Refl1d $REFLVER"
git push --tags # OR MAYBE: git push origin v$REFLVER

# to move a tag to the new head
git tag -a v$REFLVER -m "Refl1d $REFLVER" -f
git push --tags -f

# mark an existing commit with a version tag e.g.:
#GIT_COMMITTER_DATE="$(git show -s --format=%cI bd145a46)" git tag -a v0.8.1 bd145a46 -m "Refl1d 0.8.1"

# trigger the build on readthedocs (this should happen automatically now)
https://readthedocs.org/builds/refl1d/

(8) Build source distribution

python setup.py sdist

(8) Update the release page

https://github.com/reflectometry/refl1d/releases/edit/vX.Y.Z

Documentation: http://refl1d.readthedocs.org

Use `pip install refl1d wxpython` to install in your python environment.

For windows, download `Refl1d-X.Y.Z-exe.zip` and extract the archive. Go to
the extracted directory and click on `refl1d_gui.bat`. This will open a dialog
saying that the application is untrusted with a "Don't run" button at the
bottom. Click on "more info" and a "Run anyway" button will appear.

YYYY-M-DD vX.Y.Z
================
... content of CHANGES.rst for most recent release ...

See [CHANGES.rst](https://github.com/reflectometry/refl1d/tree/vX.Y.Z/CHANGES.rst) for complete history.


(9) download wheel files for windows/mac cp27/36/37 from release page

# You will to download the wheels from github with something like:
#
#   curl -OL https://github.com/reflectometry/refl1d/releases/download/v0.8.10/refl1d-0.8.10-cp27-cp27m-macosx_10_13_x86_64.whl
#   ...
#   curl -OL https://github.com/reflectometry/refl1d/releases/download/v0.8.10/refl1d-0.8.10-cp38-cp38-win_amd64.whl
#
# These commands can be generated using the following:
curl -s https://api.github.com/repos/reflectometry/refl1d/releases/tags/v$REFLVER | grep "download_url.*whl" | sed -e's/^.*https:/curl -OL https:/;s/".*$//'

# Copy and paste the commands after changing into the dist directory.
cd dist
<paste>
cd ..

(10) update pypi with new release version

twine upload dist/refl1d-$REFLVER*.whl dist/refl1d-$REFLVER.tar.gz

(11) check that the new pypi version runs (single machine should be fine)

# create virtualenv
cd ~
conda create -n reflpip python==3.8
conda activate reflpip
pip install refl1d
python -m refl1d.main
pythonw -m refl1d.main --edit
# would be nice to run tests against installed version, but not yet possible
deactivate
rm -r ~/anaconda/envs/piptest

(12) update shared resources with new versions

ssh ncnrgpu
sudo su - conda
conda activate refl1d
pip install --upgrade refl1d

*skip*
ssh sparkle  // shimmer
cd ~/src/periodictable && git pull
cd ~/src/bumps && git pull
cd ~/src/refl1d && git pull && python setup.py build

*skip*
ssh rocks
pip install --user --no-deps periodictable bumps refl1d

(13) announce release

update reflectometry.org web pages
send message to reflectometry announcement list

EOF

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

(2) check that all tests pass
  [bumps, refl1d, periodictable] X [windows, mac, linux] X [python 2.6, 2.7, 3.3, 3.4]

git pull
pythonX.Y tests.py
pythonX.Y check_examples.py --chisq

(3) update change log in CHANGES.rst

(cd ../periodictable && git log)
(cd ../bumps && git log)
git log

(4) update version number and requirements

vi refl1d/__init__.py
vi setup.py
vi rtd-requirements
git commit -a -m "R X.Y.Z"
git push

(5) build the docs

# even though this should happen automatically in jenkins we
# do the builds by hand to make sure they are error free.
(cd doc && make clean html pdf)

(6) build the binaries for windows and mac

# again, release is on jenkins, but this allows us to check the build
# Note: don't use anaconda to build release version unless we update all
# web pages to say "built with anaconda"; as convenient as it is, we don't
# want to run afoul of license agreements by missing a page.
[windows]
python master_builder.py
#python setup_py2exe.py

[mac]
python setup_py2app.py


(7) tag release

git tag -a vX.Y.Z -m "Refl1d X.Y.Z"
git push --tags # OR MAYBE: git push origin vX.Y.Z

# to move a tag to the new head
git tag -a vX.Y.Z -f
git push --tags -f


# trigger the build on readthedocs
https://readthedocs.org/builds/refl1d/

(8) update pypi with new release version

python setup.py sdist upload

(9) check that the new pypi version runs (single machine should be fine)

# create virtualenv
cd ~
conda create -n piptest pip numpy matplotlib scipy wxpython pyparsing
activate piptest
pip install refl1d
python -m refl1d.main
pythonw -m refl1d.main --edit
# would be nice to run tests against installed version, but not yet possible
deactivate
rm -r ~/anaconda/envs/piptest

(10) update shared resources with new versions

ssh sparkle  // shimmer
cd ~/src/periodictable && git pull
cd ~/src/bumps && git pull
cd ~/src/refl1d && git pull && python setup.py build

ssh rocks
pip install --user --no-deps periodictable bumps refl1d

(11) announce release

update reflectometry.org web pages
send message to reflectometry announcement list

EOF

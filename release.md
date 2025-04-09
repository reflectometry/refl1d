# Refl1D Release Process

1. Release dependent packages (bumps, periodictable) if necessary

    For reproducible builds we want a fixed bumps version number, but for users who are living without virtualenv, we want to be relaxed about the version number. Maybe separate pip requirements from binary distribution requirements? Or maybe don't worry about it.

    Check dependencies in various locations including:
    - requirements.txt
    - setup.py
    - doc/requirements.txt
    - .readthedocs.yaml
    - .github/workflows/\*

1. Check that all dependencies are up to date and compatible, in particular bumps and periodictable

    Note: linux boxen includes ~/dev as well as ~/src on the shared machines

    ```bash
    git status
    git push
    git pull
    ```

1. Check that all tests pass for bumps, refl1d, and periodictable on Windows, Mac, Linux for all supported python versions

    ```bash
    python setup.py test
    python check_examples.py --chisq
    ```

    and check that the build badge on the github page is green.

1. Update change log in CHANGES.rst

    ```bash
    cd ../periodictable && git log
    cd ../bumps && git log
    git log

1. Build the docs

    This should happen automatically via Github Actions, but we should do the builds by hand to make sure they are error free.

    ```bash
    cd doc && make clean html pdf
    ```

    **Note**: the pdf build requires `latexmk` and `pdflatex` to be installed.  
    If you don't have these installed, you can omit the pdf build.

1. Update version number and requirements

    ```bash
    REFLVER=X.Y.Z
    vi refl1d/**init**.py
    vi rtd-requirements
    git commit -a -m "R $REFLVER"
    git push
    ```

1. Tag release

    ```bash
    git tag -a v$REFLVER -m "Refl1d $REFLVER"
    git push --tags

    # to move a tag to the new head
    git tag -a v$REFLVER -m "Refl1d $REFLVER" -f
    git push --tags -f

    # mark an existing commit with a version tag e.g.:
    GIT_COMMITTER_DATE="$(git show -s --format=%cI bd145a46)" git tag -a v0.8.1 bd145a46 -m "Refl1d 0.8.1"
    
1. Update the release page: https://github.com/reflectometry/refl1d/releases/edit/vX.Y.Z
  
      Format:
      ```markdown
      - list of changes
      - made in this release
      - since the last release
      See [CHANGES.rst](https://github.com/reflectometry/refl1d/tree/vX.Y.Z/CHANGES.rst) for complete history.

1. Trigger the build on readthedocs
    - This step is automated via [readthedocs](https://readthedocs.org/builds/refl1d/)

1. Update pypi with new release version
    - This step is automated via the `test.yml` GitHub Action

1. Check that the new pypi version runs (single machine should be fine)
      
    ```bash
    # create virtualenv
    cd ~
    conda create -n reflpip
    conda activate reflpip

    # install and run refl1d
    pip install refl1d
    python -m refl1d.main
    pythonw -m refl1d.main --edit

    # would be nice to run tests against installed version, but not yet possible

    # clean up
    deactivate
    conda env remove -n reflpip 
    ```

1. Update shared/remote resources to new version(s)

    ```bash
    # uses pip to install refl1d
    ssh ncnrgpu
    sudo su - conda
    conda activate refl1d
    pip install --upgrade refl1d

    # uses source code
    # Skip
    ssh sparkle // shimmer
    cd ~/src/periodictable && git pull
    cd ~/src/bumps && git pull
    cd ~/src/refl1d && git pull && python setup.py build

    # Skip
    ssh rocks
    pip install --user --no-deps periodictable bumps refl1d
    ```

1. Announce release:
    - send message to reflectometry mailing list
    - update web pages

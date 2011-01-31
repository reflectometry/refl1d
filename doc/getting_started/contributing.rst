.. _contributing:

********************
Contributing Changes
********************

.. contents:: :local:

The best way to contribute to the reflectometry package is to work
from a copy of the source tree in the revision control system.

Currently it is hosted on svn, and can be accessed using::

	svn co svn://danse.us/reflectometry/trunk/refl1d
	cd refl1d
	python setup.py develop

By using the *develop* keyword on setup.py, changes to the files in the
package are immediately available without the need to run setup.py
install each time.

Track updates to the original package using::

    svn up

If you find you need to modify the package, please update the documentation 
and add tests for your changes.  We use doctests on all of our examples 
that we know our documentation is correct.  More thorough tests are found 
in test directory.  Using the the nose test package, you can run both sets 
of tests::

    easy_install nose
    python2.5 tests.py
    python2.6 tests.py

When all the tests run, generate a patch and send it to the 
`DANSE <http://danse.us>`_ Project mailing list at danse-dev@cacr.caltech.edu::

    svn diff > refl1d.patch

Windows user can use `TortoiseSvn <http://tortoisesvn.tigris.org/>`_ 
package which provides similar operations.

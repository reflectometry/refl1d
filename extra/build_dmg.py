#!/usr/bin/env python
"""
Build a dmg out of app in the "dist/product version.app" directory, the
html docs in "doc/_build/html" and the examples in "doc/examples".

Call from the project root as::

    python extra/build_dmg.py PRODUCT-NAME PRODUCT-VERSION

"extra/dmgpack.sh" should contain the shell script needed to create a
dmg given a set of directories.
"""

import os
import sys
import shutil

def build_dmg(name, version):
    """DMG builder; should include docs"""
    product = name+" "+version
    productdash = name+"-"+version
    app="dist/%s.app"%product
    dmg="dist/%s.dmg"%productdash
    # Remove previous build if it is still sitting there
    if os.path.exists(app): 
        shutil.rmtree(app)
    if os.path.exists(dmg): 
        os.unlink(dmg)
    print(os.getcwd(), name, app)
    os.rename("dist/%s.app"%name, app)
    os.system('cd dist && ../extra/dmgpack.sh "%s" "%s.app" ../doc/_build/html ../doc/examples'
              % (productdash,product))
    os.system('chmod a+r "%s"'%dmg)

def main():
    if len(sys.argv) != 3:
        print("usage: python build_dmg.py  PRODUCT-NAME PRODUCT-VERSION")
        sys.exit(1)
    name, version = sys.argv[1:3]
    build_dmg(name, version)

if __name__ == "__main__":
    main()


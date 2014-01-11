#!/usr/bin/env python
import sys, os

sys.dont_write_bytecode = True

ROOT = os.path.abspath(os.path.dirname(__file__))
CLI = "%s %s/bin/refl1d_cli.py %%s %%s"%(sys.executable,ROOT)
EXAMPLEDIR = os.path.join(ROOT,'doc','_examples')

# Add the build dir(s) to the system path
from distutils.util import get_platform
platform = '.%s-%s'%(get_platform(),sys.version[:3])
buildpath = os.path.abspath(os.path.join(ROOT,'build','lib'+platform))
packages = [buildpath]
try: import bumps
except: packages.append(os.path.abspath(os.path.join(ROOT,'..','bumps','build','lib'+platform)))
try: import periodictable
except: packages.append(os.path.abspath(os.path.join(ROOT,'..','periodictable')))
if 'PYTHONPATH' in os.environ: packages.append(os.environ['PYTHONPATH'])
os.environ['PYTHONPATH'] = os.pathsep.join(packages)

class Commands(object):
    @staticmethod
    def preview(f):
        return os.system(CLI%(f,'--preview --seed=1'))

    @staticmethod
    def edit(f):
        return os.system(CLI%(f,'--edit --seed=1'))

    @staticmethod
    def chisq(f):
        return os.system(CLI%(f,'--chisq --seed=1'))

examples = [
    "distribution/dist-example.py",
    "ex1/nifilm-web.py",
    "ex1/nifilm-fit-web.py",
    "ex1/nifilm-data-web.py",
    "ex1/nifilm-tof-web.py",
    "freemag/pmf.py",
    "ill_posed/anticor.py",
    "ill_posed/tethered.py",
    "interface/model.py",
    "mixed/mixed-web.py",
    "mixed/mixed_magnetic.py",
    #"peaks/model.py",
    "polymer/tethered-web.py",
    "polymer/freeform.py",
    "profile/model.py",
    "spinvalve/n101G.py",
    "staj/De2_VATR.py",
    "superlattice/freeform-web.py",
    "superlattice/NiTi-web.py",
    "superlattice/PEMU-web.py",
    "thick/nifilm.py",
    "TOF/du53.py",
    "xray/mlayer-lin.staj",
    "xray/model.py",
    "xray/staj.py",
    ]

def main():
    if len(sys.argv) == 1 or not hasattr(Commands, sys.argv[1]):
        print("usage: check_examples.py [preview|edit|chisq]")
    else:
        command = getattr(Commands, sys.argv[1])
        for f in examples:
            print("\n"+f)
            if command(os.path.join(EXAMPLEDIR,f)) != 0:
                #break
                pass

if __name__ == "__main__": main()

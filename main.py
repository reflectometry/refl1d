import sys

if len(sys.argv) == 2:
    execfile(sys.argv[1])
else:
    print "Usage: %s model.py"%sys.argv[0]

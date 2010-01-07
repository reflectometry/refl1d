#!/usr/bin/env python
import sys; sys.path.append('..')

import pylab
import mystic.examples as ex

Po = ex.minimal_circle.start

print ex.minimal_circle(Po)
#ex.minimal_circle.response_surface()
ex.minimal_circle.plot()
pylab.show()

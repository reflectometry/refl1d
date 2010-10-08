import sys; sys.path.extend(('c:/home/pkienzle/danse/refl1d','c:/home/pkienzle/danse/refl1d/dream'))
from math import pi
from refl1d import *
Probe.view = 'log' # log, linear, fresnel, or Q**4

#from refl1d import ncnrdat
#import periodictable

# Compute slits from dT given in the staj file
slits = 0.03
instrument = ncnrdata.XRay(dLoL=0.005, slits_at_Tlo=slits)
probe = instrument.load('e1085009.log')
probe.log10_to_linear()  # data was stored as log_10 (R) rather than R

# Values from staj
Pt = SLD(name='Pt', rho=86.431, irho=42.41/(2*1.54))
Ni80Fe20 = SLD(name="Ni80Fe20", rho=63.121, irho=8.24/(2*1.54))
Pt55Fe45 = SLD(name="Pt55Fe45", rho=93.842, irho=32.2/(2*1.54))
seed = SLD(name="seed", rho=110.404, irho=42.41/(2*1.54))
glass = SLD(name="glass", rho=15.086, irho=1.55/(2*1.54))


sample = (glass%(17.53/2.35)
          + seed/22.9417%(20.72/2.35)
          + Pt55Fe45/146.576%(20.22/2.35)
          + Ni80Fe20/508.784%(29.93/2.35)
          + Pt/31.8477%(25.18/2.35)
          + air)

# Values from fit
if 1:
    sample[0].interface.value = 7.53
    sample[0].material.rho.value = 15.38
    sample[0].material.irho.value = 0.41
    sample[1].thickness.value = 19.6
    sample[1].interface.value = 9.19
    sample[1].material.rho.value = 109.38
    sample[1].material.irho.value = 11.47
    sample[2].thickness.value = 150.06
    sample[2].interface.value = 8.92
    sample[2].material.rho.value = 98.52
    sample[2].material.irho.value = 10.59
    sample[3].thickness.value = 514.27
    sample[3].interface.value = 12.45
    sample[3].material.rho.value = 61.93
    sample[3].material.irho.value = 3.29
    sample[4].thickness.value = 25.01
    sample[4].interface.value = 10.89
    sample[4].material.rho.value = 92.02
    sample[4].material.irho.value = 11.93

elif 1: # grower values
    sample[1].thickness.value = 25
    sample[2].thickness.value = 200
    sample[3].thickness.value = 500
    sample[4].thickness.value = 25
    for i,L in enumerate(sample[1:-2]):
        L.interface.value = 20

# Fit parameters
#probe.theta_offset.dev(radians(0.01)/sqrt(12))  # accurate to 0.01 degrees

store = "T1"
if 1: # Open set
    for i,L in enumerate(sample[0:-1]):
        if i>0: L.thickness.range(0,1000)
        L.interface.range(0,50)
        L.material.rho.range(0,200)
        L.material.irho.range(0,200)
elif 0: # jiggle
    for i,L in enumerate(sample[0:-1]):
        if i>0: L.thickness.pmp(0,10)
        L.interface.pmp(0,10)
        L.material.rho.pmp(10)
        L.material.irho.pmp(10)
elif 0: # grower
    sample[1].thickness.range(15,35)
    sample[2].thickness.range(150,250)
    sample[3].thickness.range(300,700)
    sample[4].thickness.range(15,35)
    for i,L in enumerate(sample[1:-2]):
        L.interface.pmp(100)
        L.material.rho.pmp(10)
        L.material.irho.pmp(10)

elif 0: # d2 X d3
    #sample[2].thickness.range(0,400)
    #sample[3].thickness.range(0,1000)
    sample[2].thickness.range(50,400)
    sample[3].thickness.range(50,1000)
    sample[2].thickness.value = 400
    sample[3].thickness.value = 1000

M = Experiment(probe=probe, sample=sample)

# Needed by dream fitter
problem = FitProblem(M)
problem.dream_opts = dict(chains=20,draws=1000,burn=3000)
problem.name = "Example"
problem.title = "xray"
problem.store = store

#Probe.view = 'log'
#from refl1d.mystic.parameter import randomize, varying
#randomize(varying(M.parameters()))


# Do the fit
if 1:
    result = preview(models=M)
elif 1:
    result = fit(models=[M], npop=10)
    result.show()
elif 0:
    import dream
    mc = dream.load_state('grower2')
    mc.show(portion=1)
elif 0:
    result = fit(models=M).show()
elif 0:
    from dream.corrplot import COLORMAP
    import pylab
    from numpy import min,exp
    x,y,image = mesh(models=M,
                 vars=[sample[2].thickness,sample[3].thickness],
                 n=200)
    vmax = 100*min(image)
    image[image>vmax] = vmax
    pylab.pcolormesh(y,x,-0.5*image.T, cmap=COLORMAP)
    pylab.colorbar()
    pylab.show()
elif 0:
    mc = draw_samples(models=M, chains=10, draws=5000, burn=1000)
    output = "xray"
    sys.stdout = open(output+".out","w")
    model.show()

    # Plot
    import pylab
    model.plot(fignum=6, figfile=output)
    pylab.suptitle(":".join((model.store,model.title)))
    state.show(figfile=output)
    if not "--noplot" in fit_options: pylab.show()

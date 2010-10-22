from refl1d import *
from refl1d.stajconvert import load_mlayer
Probe.view = 'fresnel'

# Load neutron model and data from staj file
Mnr = load_mlayer("De2_VATR.staj")

if True:
    Mxr = load_mlayer("De2_VATR.staj")
    for Lxr,Lnr in zip(Mxr.sample,Mnr.sample):
        Lxr.thickness = Lnr.thickness
        Lxr.interface = Lnr.interface

    for Lxr,Lnr in zip(Mxr.sample,Mnr.sample):
        Lxr.material.rho.pmp(5)
        Lnr.material.rho.pmp(5)
        Lnr.thickness.pmp(20)
        Lnr.interface.pmp(20)

if False:
    # Construct xray layers from neutron layers with shared thickness
    # and roughness, but independent SLD, and no absorption
    xray_slds = [SLD(rho=L.material.rho.value) for L in Mnr.sample]
    layers = [Slab(sld,thickness=L.thickness,interface=L.interface)
          for sld,L in zip(xray_slds, Mnr.sample)]

    for sld in xray_slds:
        sld.rho.range(0,20)
    for L in layers:
        L.thickness.pmp(5)

    # Load xray data
    instrument = ncnrdata.XRay(slits_at_Tlo=0.2)
    xray_probe = instrument.load('n5hl2_Short.refl')

    # Associate xray model and xray data
    Mxr = Experiment(probe=xray_probe, sample=Stack(layers))

#preview(models=[M, Mxray])
fit(models=[Mnr,Mxr], npop=100)



for i in range(1,len(M.sample)-1):
    M.sample[i].thickness.pmp(30)
    M.sample[i].interface.pmp(30)
    M.sample[i].material.rho.pmp(10)
    #M.sample[i].material.irho.pmp(10)
if 1:
    refl1d.preview(models=M)
else:
    result = refl1d.fit(models=M) #, fitter=None)
    result.resample(samples=100, fitter=refl1d.DEfit)
    result.save('De2_VATR_staj')
    result.show()
    result.show_stats()

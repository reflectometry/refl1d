import numpy
from .util import next_color

# TODO: we should just keep a certain number of evaluations as a matter
# of course during sampling rather than recomputing them after the fact.
# TODO: want similar code for covariance matrix based forward analysis
# TODO: need to delegate accumulation of models and plotting to Fitness

def calc_distribution_from_state(problem, state, nshown=50):
    points, logp = state.sample()
    if points.shape[0] < nshown: nshown = points.shape[0]
    return calc_distribution(problem, points[-nshown:])

def calc_distribution(problem, points):
    original = problem.getp()
    try:
        ret = _calc_distribution(problem, points)
    except:
        import traceback
        print "error calculating distribution on model"
        traceback.print_exc()
        ret = None
    finally:
        problem.setp(original)
    return ret

def _calc_distribution(problem, points):
    from .experiment import ExperimentBase

    # Grab the individual samples
    if hasattr(problem, 'models'):
        models = [m.fitness for m in problem.models]
    else:
        models = [problem.fitness]
    # Hack: this only works for refl!
    if not isinstance(models[0],ExperimentBase): return None
    experiments = []
    for m in models:
        if hasattr(m,'parts'):
            experiments.extend(m.parts)
        else:
            experiments.append(m)
    #probes = []
    #for m in experiments:
    #    if hasattr(m.probe,'probes'):
    #        probes.extend(m.probe.probes)
    #    elif hasattr(m.probe,'xs'):
    #        probes.extend([p for p in m.probe if p])
    #    else:
    #        probes.append(p)

    # Find Q
    def residQ(m):
        if m.probe.polarized:
            return numpy.hstack([xs.Q
                                 for xs in m.probe.xs
                                 if xs is not None])
        else:
            return m.probe.Q
    Q = dict((m, residQ(m)) for m in experiments)

    profiles = dict((m,[]) for m in experiments)
    residuals = dict((m,[]) for m in experiments)
    def record_point():
        chisq = problem.chisq()
        for m in experiments:
            D = m.residuals()
            residuals[m].append(D+0)
            if m.ismagnetic:
                z,rho,irho,rhoM,thetaM = m.magnetic_profile()
                profiles[m].append((z+0,rho+0,irho+0,rhoM+0,thetaM+0))
            else:
                z,rho,irho = m.smooth_profile()
                profiles[m].append((z+0,rho+0,irho+0))
    record_point() # Put best at slot 0
    for p in points:
        problem.setp(p)
        record_point()

    # Align profiles
    _align_profiles(profiles)

    residuals = dict((m,numpy.array(residuals[m]).T)
                     for m in experiments)
    return profiles, Q, residuals

def show_distribution(profiles, Q, residuals):
    import pylab
    pylab.subplot(211)
    for m,p in profiles.items():
        if len(p[0]) == 3:
            color = next_color()
            for z,rho,_ in p[1:]:
                pylab.plot(z,rho,'-',hold=True,color=color,alpha=0.4)
            z,rho,_ = p[0]
            pylab.plot(z,rho,'-k',hold=True) # best
        else:
            rho_color = next_color()
            rhoM_color = next_color()
            for z, rho,_,rhoM,_ in p[1:]:
                pylab.plot(z,rho,'-',hold=True,color=rho_color,alpha=0.4)
                pylab.plot(z,rhoM,'-',hold=True,color=rhoM_color,alpha=0.4)
            z,rho,_,rhoM,_ = p[0]
            pylab.plot(z,rho,'-k',hold=True) # best
            pylab.plot(z,rhoM,'-k',hold=True) # best

    pylab.subplot(212)
    shift = 0
    for m,r in residuals.items():
        color = next_color()
        pylab.plot(Q[m], shift+residuals[m][:,1:],'.',color=color, alpha=0.4)
        pylab.plot(Q[m], shift+residuals[m][:,0],'.k', hold=True) # best
        shift += 5

def _align_profiles(profiles):
    """
    Align profiles for each sample
    """
    for m in profiles.keys():
        profiles[m] = _align_profile_set(profiles[m])

def _align_profile_set(profiles):
    """
    Align all profiles to the first profile.
    """
    p1 = profiles[0]
    offsets = [0]
    for p2 in profiles[1:]:
        offsets.append(_align_profile_pair(p1[0],p1[1],p2[0],p2[1]))
    profiles = [tuple([p[0]+offset]+list(p[1:]))
                for offset,p in zip(offsets,profiles)]
    return profiles

def _align_profile_pair(z1,r1,z2,r2):
    """
    Use autocorrelation to align r1 and r2.
    """
    # Assume z1,z2 have the same step size
    n1,n2 = len(r1),len(r2)
    idx = numpy.argmax(numpy.correlate(r1,r2,'full'))
    if idx < n2:
        offset = z2[n2-idx-1] - z1[0]
    else:
        offset = z2[0] - z1[idx-n2+1]
    return offset

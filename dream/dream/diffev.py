from __future__ import division
from numpy import zeros, dot, cov, eye, sqrt, all, where, sum
from numpy.linalg import cholesky, LinAlgError
from . import util

def de_step(x,CR,max_pairs=2,eps=0.05):
    """
    Generates offspring using METROPOLIS HASTINGS monte-carlo markov chain
    """
    Npop, Nvar = x.shape

    # Initialize the delta update to zero
    delta_x = zeros( (Npop,Nvar) )

    # Full differential evolution 80% of the time
    used = util.RNG.rand(Npop) < 4/5
    
    # Chains evolve using information from other chains to create offspring
    for qq in range(Npop):

        # Generate a random permutation of individuals excluding the current
        perm = util.RNG.permutation(Npop-1)
        perm[perm>=qq] += 1

        if used[qq]:
            
            # Select to number of vector pair differences to use in update
            # using k ~ discrete U[1,max pairs]
            k = util.RNG.randint(max_pairs)+1
            # [PAK: same as k = DEversion[qq,1] in the old code]
            
            # Select 2*k members at random different from the current member
            r1,r2 = perm[:k],perm[k:2*k]
        
            # Select the dims to update based on the crossover ratio, making
            # sure at least one dim is selected
            vars = where(util.RNG.rand(Nvar) > (1-CR[qq]))[0]
            if len(vars) == 0: vars = [util.RNG.randint(Nvar)]

            # Weight the size of the jump inversely proportional to the 
            # number of contributions, both from the parameters being
            # updated and from the population defining the step direction.
            gamma = 2.38/sqrt(2 * len(vars) * k)
            # [PAK: same as F=Table_JumpRate[len(vars),k] in the old code]

            # Find and average step from the selected pairs
            delta = sum(x[r1]-x[r2], axis=0)

            # Apply that step with F scaling and noise
            noise = 1 + eps * (2 * util.RNG.rand(*delta.shape) - 1)
            delta_x[qq,vars] = (noise*gamma*delta)[vars]

        else:  # 20% of the time, just use one pair and all dimensions

            # Note that there is no F scaling, dimension selection or noise
            delta_x[qq,:] = x[perm[0],:] - x[perm[1],:]

        # If no step was specified (exceedingly unlikely!), then 
        # select a delta at random from a gaussian approximation to the 
        # current population
        if all(delta_x[qq] == 0):
            try:
                # Compute the Cholesky Decomposition of x_old
                R = (2.38/sqrt(Nvar)) * cholesky(cov(x.T) + 1e-5*eye(Nvar))
                # Generate jump using multinormal distribution
                delta_x[qq] = dot(util.RNG.randn(*(1,Nvar)), R)
            except LinAlgError:
                print "Bad cholesky"
                delta_x[qq] = util.RNG.randn(Nvar)


    # Update x_old with delta_x and noise
    x_new = x + delta_x + 1e-6*util.RNG.randn(*x.shape)

    return x_new, used

def test():
    from numpy import array,arange
    x = 100*arange(8*10).reshape((8,10))   # pop 10, vars 8
    x = x + RNG.rand(*x.shape)*1e-6
    CR = array([0,0,.2,.2,.8,.8,1,1])
    x_new, used = de_step(x,CR,max_pairs=2,eps=0.05)
    print """\
The following table shows the expected portion of the dimensions that 
are changed and the rounded value of the change for each point in the 
population.
"""
    for r,i,u in zip(CR,range(8),used):
        rstr = ("%3d%%"%(r*100)) if u else "full"
        vstr = " ".join("%4d"%(int(v/100+0.5)) for v in x_new[i]-x[i])
        print rstr, vstr

if __name__ == "__main__":
    test()
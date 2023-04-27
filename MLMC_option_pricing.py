import numpy as np

"""
    It simulates with Euler-Maruyama approximation the stochastic process given by the equation:

    dS(t) = r * S(t) * dt + sigma * S(t) * dW(t)

    It returs the collection of S(t) over a uniformly discretizatized time domain (0, T].


Parameters:

    S_init = initial condition of S
    r = coefficient of deterministic part of the option pricing equation
    sigma = coefficient of the stochastic part of the option pricing equation
    dW = collection of standard normal realizations, \sim N(0, 1)
    T = final time
"""
def EM_option_pricing(S_init, r, sigma, h, dW, T):
    
    M = dW.shape[-1]
    S = np.ones(shape = (*dW.shape[:-1], M+1)) * S_init
    
    sqrt_h = np.sqrt(h)

    for m in range(1, M+1):
        S[...,m] = (1. + r * h + sigma * sqrt_h * dW[...,m-1])  * S[...,m-1]

    return S



"""
    S_init = initial condition of S
    r = coefficient of deterministic part of the option pricing equation
    sigma = coefficient of the stochastic part of the option pricing equation
    f = function to estimate the mean f(S(T))
    T = final time of the simulation
    h0: initial time step
    nb_samples(np.array): number of samples for each level 
    
    return_samples: return Y_l - Y_{l-1}
    return_biases: return weak errors abs(mean(Y_l - Y_{l-1}))
    return_variances: return strong errors np.var(Y_l - Y_{l-1})
"""
def MLMC_option_pricing(S_init, r, sigma, T, f, h0, nb_samples, return_samples = False, return_biases = False, return_variances = False):
    
    # number of levels
    L = len(nb_samples) - 1

    # determine steps
    steps = np.array([h0 * 2**(-l) for l in range(L+1)])

    # number of time steps for each level
    M = np.round(T / steps).astype(np.int64)

    # estimated means
    level_means = np.zeros(L+1)

    # intermediate samples of Y
    dY = []
    
    #  loop on levels
    for l in range(L+1):

        # exponential decay browian motion sampling
        dW_actual = np.random.standard_normal(size = M[l] * nb_samples[l]).reshape(( nb_samples[l], M[l] ))
        
        Y_prev = np.zeros(nb_samples[l])
        
        if l > 0:
            # twice the time step brownian motion
            dW_prev = (dW_actual[:,0::2] + dW_actual[:,1::2]) / np.sqrt(2.)
            
            # previous step computation 
            S_prev = EM_option_pricing(S_init, r, sigma, steps[l-1], dW_prev, T)
            Y_prev = f(S_prev, h = steps[l-1])
        
        # actual step computation
        S_actual = EM_option_pricing(S_init, r, sigma, steps[l], dW_actual, T)
        Y_actual = f(S_actual, h = steps[l])
        
        # store intermediate samples
        dY.append(Y_actual - Y_prev)
        
        # average over samples
        level_means[l] = np.mean(Y_actual - Y_prev)

    # collect outputs
    outputs = [np.sum(level_means)]
    
    # return forward model samples
    if return_samples:
        outputs.append(dY)
    
    # return estimated biases
    if return_biases:
        biases = [np.abs(np.mean(dy)) for dy in dY]
        outputs.append(biases)
    
    # return estimated variances
    if return_variances:
        variances = [np.var(dy, ddof=1) for dy in dY]
        outputs.append(variances)
    
    return tuple(outputs) if len(outputs) > 1 else outputs[0]

import numpy as np
import pandas as pd
import pandas as pd
import scipy.optimize
import scipy.stats as st
import warnings
import multiprocessing
# from tqdm.auto import tqdm
import tqdm
import bebi103


def gen_model(params, size):
    '''Returns a list that is the same length as the empirical data set and contains values sampled from our model pdf'''
    beta1, beta2 = params
    
    rg = np.random.default_rng()
    
    if abs(beta2 - beta1) < 0.000001 or beta2 < beta1:
        return rg.gamma(2, 1 / beta1, size=size)
    return rg.exponential(1/beta1, size=size) + rg.exponential(1/beta2, size=size)

def model_log_like(beta1, beta2, point):
    return np.log((beta1 * beta2)/(beta2 - beta1)) - beta1 * point + np.log(1 - np.exp(point*(beta1 - beta2)))

def log_like_iid_model(params, n):
    """Log likelihood for i.i.d. model measurements, parametrized
    by beta1, beta2."""
    beta1, beta2 = params

    if beta1 <= 0 or beta2 <= 0 or beta2 < beta1:
        return -np.inf
    
    if abs(beta1-beta2) < 0.0000001:
        return np.sum(st.gamma.logpdf(n, 2, loc=0, scale=1/beta1))

    return np.sum(model_log_like(beta1, beta2, point) for point in n)


def mle_iid_model(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    model measurements, parametrized by beta1, beta2"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_model(params, n),
            x0=np.array([3, 3]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        print("failed in mle_iid_model", res.x)
        raise RuntimeError('Convergence failed with message', res.message)
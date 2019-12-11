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


def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. gamma measurements."""
    
    alpha, beta = params

    if alpha <= 0 or beta <= 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(n, alpha, scale=1/beta))

def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    gamma measurements, parametrized by alpha, beta"""
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=np.array([3, 3]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)


def calculate(df, conc):
    '''
        df: tidy dataframe with "concentration" and "time to catastrophe (s)" columns
        conc: concentration of interest
    '''
    df_conc = df.loc[df['concentration']==conc]
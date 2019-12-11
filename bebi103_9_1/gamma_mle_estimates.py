import numpy as np
import pandas as pd
import pandas as pd
import scipy.optimize
import scipy.stats as st
import warnings
import bebi103

def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    gamma measurements, parametrized by alpha, b=1/beta"""
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
        print(res)
        raise RuntimeError('Convergence failed with message', res.message)
        
def gen_gamma(alpha, beta, size):
    return rg.gamma(alpha, scale=1/beta, size=size)  

def get_mles():
    alphas = []
    betas = []
    conf = []
    for key in df.concentration.unique():
        df_temp = df.loc[df['concentration']==key, 'time to catastrophe (s)'].values
        df_temp = df_temp[~np.isnan(df_temp)]
        alpha, beta = mle_iid_gamma(df_temp)
        alphas.append(alpha)
        betas.append(beta)    
        print(key, "estimates | ", 'alpha: ', alpha, ' | beta: ', beta)        
    return alphas, betas

def draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    params = mle_fun(data, *args)

    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [mle_fun(gen_fun(*params, size=len(data), *args)) for _ in iterator]
    )

def draw_bs_all(df):
    '''
    Draw bootstrap replicates for all concentrations
    '''
    conf_ints = []
    for key in df.concentration.unique():
        data = df.loc[df['concentration']==key, 'time to catastrophe (s)'].values
        data = data[~np.isnan(data)]
        bs_reps = draw_parametric_bs_reps_mle(mle_iid_gamma, gen_gamma, data, size=5000, progress_bar=True)
        conf_ints.append(np.percentile(bs_reps, [2.5, 97.5], axis=0))
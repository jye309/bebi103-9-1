import numpy as np
import pandas as pd
import pandas as pd
import scipy.optimize
import scipy.stats as st
import warnings
import multiprocessing
# from tqdm.auto import tqdm
import tqdm

import bokeh.io
import bokeh_catplot
import holoviews as hv
import matplotlib.pyplot as plt

import bebi103

hv.extension('bokeh')
import bokeh.io
bokeh.io.output_notebook()

def draw_gamma(alpha, beta, size=1):
    rg = np.random.default_rng()
    return rg.gamma(alpha, scale=1/beta, size=size)

def draw_model(beta1, beta2, size=1):
    rg = np.random.default_rng()
    return rg.exponential(1/beta1, size=size) + rg.exponential(1/beta2, size=size)

def make_qq(n, alpha, beta, beta1, beta2):
    ''' 
    Make a QQ plot of (1) the gamma distribution and (2) the successive poisson distribution
        n: data from df
        alpha: MLE estimate for gamma distriubution
        beta: MLE estimate for rate of arrival in gamma distribution
        beta1: MLE estimate for rate of arrival of chemical reaction 1 successive poissson distribution
        beta2: MLE estimate for rate of arrival of chemical reaction 2 in successive poissson distribution
    '''
    
    samples_gamma = np.array([draw_gamma(alpha, beta, size=len(n)) for _ in range(5000)])
    samples_model = np.array([draw_model(beta1, beta2, size=len(n)) for _ in range(5000)])

    p_gamma_qq = bebi103.viz.qqplot(
        data=n,
        gen_fun=draw_gamma,
        args=(alpha, beta),
        x_axis_label="Time to Catastropne (s)",
        y_axis_label="Time to Catastropne (s)",
        title='Gamma Distribution QQ Plot'
    )

    p_model_qq = bebi103.viz.qqplot(
        data=n,
        gen_fun=draw_model,
        args=(beta1, beta2),
        x_axis_label="Time to Catastropne (s)",
        y_axis_label="Time to Catastropne (s)",
        title='Successive Poisson QQ Plot'
    )
    
    return p_gamma_qq, p_model_qq

def make_pred_ecdf(n, alpha, beta, beta1, beta2):
    ''' 
    Make a predictive ECDF of (1) the gamma distribution and (2) the successive poisson distribution
        n: data from df
        alpha: MLE estimate for gamma distriubution
        beta: MLE estimate for rate of arrival in gamma distribution
        beta1: MLE estimate for rate of arrival of chemical reaction 1 successive poissson distribution
        beta2: MLE estimate for rate of arrival of chemical reaction 2 in successive poissson distribution
    '''
    samples_gamma = np.array([draw_gamma(alpha, beta, size=len(n)) for _ in range(5000)])
   
    p_gamma_predECDF = bebi103.viz.predictive_ecdf(
        samples=samples_gamma, 
        data=n, 
        discrete=True, 
        x_axis_label="n",
        title='Gamma Distribution Predictive ECDF'
    )
    
    samples_model = np.array([draw_model(beta1, beta2, size=len(n)) for _ in range(5000)])

    p_model_predECDF = bebi103.viz.predictive_ecdf(
        samples=samples_model, 
        data=n, 
        discrete=True, 
        x_axis_label="n",
        title='Successive Poisson Predictive ECDF'
    )
    
    return p_gamma_predECDF, p_model_predECDF

def make_ecdf_diff(n, alpha, beta, beta1, beta2):
    ''' 
    Make a difference of predictive ECDF of (1) the gamma distribution and (2) the successive poisson distribution
        n: data from df
        alpha: MLE estimate for gamma distriubution
        beta: MLE estimate for rate of arrival in gamma distribution
        beta1: MLE estimate for rate of arrival of chemical reaction 1 successive poissson distribution
        beta2: MLE estimate for rate of arrival of chemical reaction 2 in successive poissson distribution
    '''
    
    samples_gamma = np.array([draw_gamma(alpha, beta, size=len(n)) for _ in range(5000)])
    
    p_gamma_diff = bebi103.viz.predictive_ecdf(
        samples=samples_gamma, 
        data=n, 
        diff=True, 
        discrete=True, 
        x_axis_label="n",
        title='Gamma Distribution ECDF Difference'
    )

    samples_model = np.array([draw_model(beta1, beta2, size=len(n)) for _ in range(5000)])

    p_model_diff = bebi103.viz.predictive_ecdf(
        samples=samples_model, 
        data=n, 
        diff=True, 
        discrete=True, 
        x_axis_label="n",
        title='Successive Poisson ECDF Difference'
    )
    
    return p_gamma_diff, p_model_diff


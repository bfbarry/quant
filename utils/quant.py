import pandas as pd
import numpy as np
from math import log, floor
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""More mathematically oriented functions and helper functions"""

norm1 =lambda x: (np.array(x) - min(np.array(x))) / (max(np.array(x)) - min(np.array(x)))

exp_growth = lambda a,r,x: a*(1+r)**x 

def detrend1(data):
    """Basic time series detrending based on some mlmastery page"""
    diff = list()
    for i in range(1, len(data)):
        value = data[i] - data[i - 1]
        diff.append(value)
    return diff

def draw_phase_space(x,y,z=None):
    i_c = 0
    colors = list(iter(plt.cm.inferno(np.linspace(0, 1, len(x)))))
    if z is None:
        xprev,yprev = x[0],y[0]
        plt.figure(figsize=(10,10))
        
        for xn,yn in list(zip(x,y))[1:]:
            plt.plot([xprev,xn],[yprev,yn],color=colors[i_c],alpha=0.5, linestyle='solid'); i_c+=1
            xprev,yprev=xn,yn
    else:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        xprev, yprev, zprev = x[0], y[0], z[0]
        
        for xn,yn,zn in list(zip(x,y,z))[1:]:
            ax.plot([xprev,xn],[yprev,yn],[zprev,zn],color=colors[i_c],alpha=0.5, linestyle='solid'); i_c+=1
            xprev,yprev,zprev = xn,yn,zn

def delay_embed(data, norm, m, step = 1):
    """m is window length"""
    data = np.array(data)
    nt = data.shape[0] #length of original vector
    n = int((nt-m+1)/step) #number of lags
    traj_mat = np.empty((0,m)) #initialize delay embedded trajectory matrix
    for i in range(n):
        window_i = data[i*step:i*step + m]
        traj_mat = np.append(traj_mat, [window_i], axis=0 )
    
    if norm:
        traj_mat = (1/(nt**0.5)) * traj_mat #normalize by 1/sqrt(n)
    return traj_mat

def SSA(data, m, step, norm = True, return_traj = True):
    """Performs SSA on time series with window size m and inter-window step size"""
    if type(data) != np.ndarray:
        data = np.array(data)
    nt = data.shape[0] #length of original vector
    n = int((nt-m+1)/step) #number of lags
    
    traj_mat = delay_embed(data, norm, m, step)
    pca = PCA(n_components = n)
    pca.fit(traj_mat.T)
    components = pca.components_
    e_spectrum = pca.explained_variance_ratio_
    
    if not return_traj:
        return components, e_spectrum
    else:
        return components, e_spectrum, traj_mat

def mass_SSA(tickers, period, window, step):
    """Period is a tuple, e.g. ('2019-01-01','2020-01-01') for extracting data from that time period
    Window and step are SSA parameters """
    ssa_results = {}
    for ticker in tickers:
        data = yf.download(ticker, period[0], period[1])
        ssa_results[ticker] = SSA(data, m = window, step = step, return_traj = True)
        
    return ssa_results
        
def kmeans(data, k):
    ...
    
### DFA from https://raphaelvallat.com/entropy/build/html/generated/entropy.detrended_fluctuation.html#entropy.detrended_fluctuation
###

def _linear_regression(x, y):
    """Fast linear regression using Numba.
    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables
    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept

def _log_n(min_n, max_n, factor):
    """
    Creates a list of integer values by successively multiplying a minimum
    value min_n by a factor > 1 until a maximum value max_n is reached.
    Used for detrended fluctuation analysis (DFA).
    Function taken from the nolds python package
    (https://github.com/CSchoel/nolds) by Christopher Scholzel.
    Parameters
    ----------
    min_n (float):
        minimum value (must be < max_n)
    max_n (float):
        maximum value (must be > min_n)
    factor (float):
       factor used to increase min_n (must be > 1)
    Returns
    -------
    list of integers:
        min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
        without duplicates
    """
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

def dfa(x):
    """
    Utility function for detrended fluctuation analysis
    """
    N = len(x)
    nvals = _log_n(4, 0.1 * N, 1.2)
    walk = np.cumsum(x - x.mean())
    fluctuations = np.zeros(len(nvals))

    for i_n, n in enumerate(nvals):
        d = np.reshape(walk[:N - (N % n)], (N // n, n))
        ran_n = np.array([float(na) for na in range(n)])
        d_len = len(d)
        trend = np.empty((d_len, ran_n.size))
        for i in range(d_len):
            slope, intercept = _linear_regression(ran_n, d[i])
            trend[i, :] = intercept + slope * ran_n
        # Calculate root mean squares of walks in d around trend
        # Note that np.mean on specific axis is not supported by Numba
        flucs = np.sum((d - trend) ** 2, axis=1) / n
        # https://github.com/neuropsychology/NeuroKit/issues/206
        fluctuations[i_n] = np.sqrt(np.mean(flucs))

    # Filter zero
    nonzero = np.nonzero(fluctuations)[0]
    fluctuations = fluctuations[nonzero]
    nvals = nvals[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        dfa = np.nan
    else:
        dfa, _ = _linear_regression(np.log(nvals), np.log(fluctuations))
    return dfa, fluctuations

### end of DFA ###
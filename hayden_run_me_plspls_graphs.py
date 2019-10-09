#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')
import time
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from graspy.simulations import er_np, sbm
from graspy.embed import AdjacencySpectralEmbed
from graspy.inference import LatentDistributionTest


# In[ ]:


def fit_avantis_plug_in(X):
    '''
    Estimates the variance using the plug-in estimator.
    RDPG Survey, equation 10.
    '''
    n = len(X.T)
    delta = 1 / (n) * (X @ X.T)
    delta_inverse = np.linalg.inv(delta)
    
    def avantis_plug_in(x):
        if x.ndim < 2:
            undo_dimensions = True
            x = x.reshape(-1, 1)
        else: 
            undo_dimensions = False
        middle_term = np.sum(x.T @ X - (x.T @ X)**2, axis=1) / (n)
        middle_term = np.outer(middle_term, delta)
        if undo_dimensions:
            middle_term = middle_term.reshape(delta.shape)
        else:
            middle_term = middle_term.reshape(-1, *delta.shape)
        return delta_inverse @ middle_term @ delta_inverse
    
    return avantis_plug_in


# In[ ]:


def sample_noisy_points(X, Y):
    n = len(X.T)
    m = len(Y.T)
    two_samples = np.concatenate([X, Y], axis=1)
    get_sigma = fit_avantis_plug_in(two_samples)
    sigma_X = get_sigma(X) / m
    sigma_Y = get_sigma(Y) / n
    X_sampled = np.zeros(X.shape)
    for i in range(n):
        X_sampled[:,i] = X[:, i] + stats.multivariate_normal.rvs(cov=sigma_X[i]).T
    Y_sampled = np.zeros(Y.shape)
    for i in range(m):
        Y_sampled[:,i] = Y[:, i] + stats.multivariate_normal.rvs(cov=sigma_Y[i]).T
    return X_sampled, Y_sampled


# In[ ]:


def mc_iter(n, m, p, q, tilde, i=1):
    X_graph = er_np(n, p)
    ase = AdjacencySpectralEmbed(n_components=1)
    X = ase.fit_transform(X_graph).T

    Y_graph = er_np(m, q)
    ase = AdjacencySpectralEmbed(n_components=1)
    Y = ase.fit_transform(Y_graph).T

    if tilde:
        X_new, Y_new = sample_noisy_points(X, Y)
    else:
        X_new, Y_new = X, Y

    ldt = LatentDistributionTest()
    pval = ldt.fit(X_new.T, Y_new.T, pass_graph=False)#, median_heuristic=False)
    return pval

def mc_iter_wrapper(i, n, m, p, q, tilde):
    np.random.seed(int(time.time() * i) % 100000000)
    return i, mc_iter(n, m, p, q, tilde, i)

def monte_carlo(n, m, p, q, tilde=False, mc_iters=200):
    pool = Pool(cpu_count() - 2)
    pvals = np.zeros(mc_iters)
    
    pbar = tqdm(total=mc_iters)
    def update(tup):
        i, ans = tup
        pvals[i] = ans  # put answer into correct index of result list
        pbar.update()
    
    results = [None] * mc_iters
    for i in range(mc_iters):
        results[i] = pool.apply_async(mc_iter_wrapper,
                         args = (i, n, m, p, q, tilde),
                         callback=update)
    for r in results:
        r.get()
        
    pool.close()
    pool.join()
    pbar.close()

    return np.array(pvals) < 0.05


# In[ ]:


mc_iters = 200
ns = [25, 50, 100, 200, 400, 800]
cs= [1,3,4,5,6,7,8,9,10]


# In[ ]:


data = {}
for c in cs:
    print('current c: {}'.format(c))
    tests_size_er_xhat = [monte_carlo(n=i, m=c*i, p=0.8*0.8, q=0.8*0.8,
                                      mc_iters=mc_iters, tilde=False)
                          for i in ns]
    size_er_xhat = np.array([np.sum(i)/mc_iters for i in tests_size_er_xhat])
    tests_size_er_xtilde = [monte_carlo(n=i, m=c*i, p=0.8*0.8, q=0.8*0.8,
                                        mc_iters=mc_iters, tilde=True)
                            for i in ns]
    size_er_xtilde = np.array([np.sum(i)/mc_iters for i in tests_size_er_xtilde])
    data[c] = (size_er_xhat, size_er_xtilde)


# In[ ]:


data


# In[ ]:


pkl.dump(data, open( "nonpar_results_graphs.pkl", "wb" ) )


#!/usr/bin/env python
# coding: utf-8

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline
import time
import pickle as pkl
import numpy as np
from tqdm import tqdm
from scipy import stats
# import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from graspy.simulations import er_np, sbm
from graspy.embed import AdjacencySpectralEmbed
from graspy.inference import LatentDistributionTest

def sample_noisy_points(X, Y):
    X, Y = X.flatten(), Y.flatten()
    n, m = len(X), len(Y)
    var = (np.var(X) * (n * (n - 1)) + np.var(Y) * (m * (m - 1)))/(n+m-2)
    X_sampled = X + np.random.normal(0, np.sqrt(var/m**3), n)
    Y_sampled = Y + np.random.normal(0, np.sqrt(var/n**3), m)
    return X_sampled.reshape(1, -1), Y_sampled.reshape(1, -1)

def mc_iter(n, m, p, q, tilde, i=1):
    X = np.random.normal(p, np.sqrt(1/n**3), n).reshape(1, -1)

    Y = np.random.normal(q, np.sqrt(1/m**3), m).reshape(1, -1)

    if tilde:
        X_new, Y_new = sample_noisy_points(X, Y)
    else:
        X_new, Y_new = X, Y

    ldt = LatentDistributionTest()
    pval = ldt.fit(X_new.T, Y_new.T, pass_graph=False, median_heuristic=False)
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

mc_iters = 100
ns = [800, 1200, 1600, 2000, 3000]
cs= [4]

data = {}
for c in cs:
    print('current c: {}'.format(c))
    tests_size_er_xhat = [monte_carlo(n=i, m=c*i, p=0, q=0,
                                      mc_iters=mc_iters, tilde=False)
                          for i in ns]
    size_er_xhat = np.array([np.sum(i)/mc_iters for i in tests_size_er_xhat])
#    tests_size_er_xtilde = [monte_carlo(n=i, m=c*i, p=0, q=0,
#                                        mc_iters=mc_iters, tilde=True)
#                            for i in ns]
#    size_er_xtilde = np.array([np.sum(i)/mc_iters for i in tests_size_er_xtilde])
    data[c] = (size_er_xhat)

pkl.dump(data, open( "gaussians_xhat_c5.pkl", "wb" ) )


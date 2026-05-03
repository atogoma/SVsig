#!/usr/bin/env python
"""
mvnmf初始化
"""
import numpy as np
import scipy as sp
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as sch
import warnings


def nnls(X, W):
    H = []  
    for x in X.T:  
        h, _ = sp.optimize.nnls(W, x)  
        H.append(h)  
    H = np.array(H)  
    H = H.T  
    return H  

def initialize_nmf(X, n_components, init='cluster', init_normalize_W=None,
                   init_refit_H=None,
                   init_cluster_metric='cosine',
                   init_cluster_linkage='average',
                   init_cluster_max_ncluster=100, init_cluster_min_nsample=1):

    if (type(X) != np.ndarray) or (not np.issubdtype(X.dtype, np.floating)):
        X = np.array(X).astype(float)
    n_features, n_samples = X.shape

    W, H, _ = _init_cluster(X, n_components, metric=init_cluster_metric,
                            linkage=init_cluster_linkage,
                            max_ncluster=init_cluster_max_ncluster,
                            min_nsample=init_cluster_min_nsample)
    return W, H


def _init_cluster(X, n_components, metric='cosine', linkage='average',
                  max_ncluster=100, min_nsample=1):

    n_features, n_samples = X.shape  
    XT_norm = normalize(X, norm='l1', axis=0).T  
    d = sp.spatial.distance.pdist(XT_norm, metric=metric)  
    d = d.clip(0) 
    linkage = sch.linkage(d, method=linkage)  
    for ncluster in range(n_components, np.min([n_samples, max_ncluster]) + 1):
        cluster_membership = sch.fcluster(linkage, ncluster, criterion='maxclust')  
        if len(set(cluster_membership)) != ncluster:
            cluster_membership = sch.cut_tree(linkage, n_clusters=ncluster).flatten() + 1
            if len(set(cluster_membership)) != ncluster:
                warnings.warn('Number of clusters output by cut_tree or fcluster is not equal to the specified number of clusters',
                              UserWarning)
        W = []
        for i in range(1, ncluster + 1):
            if np.sum(cluster_membership == i) >= min_nsample:
                W.append(np.mean(XT_norm[cluster_membership == i, :], 0))  
        W = np.array(W).T
        if W.shape[1] == n_components:
            break
    if W.shape[1] != n_components:
        raise RuntimeError('Initialization with init=cluster failed.')  
    W = normalize(W, norm='l1', axis=0)  
    H = nnls(X, W)  
    return W, H, cluster_membership  

def beta_divergence(A, B, square_root=False):

    A_data = A.ravel()
    B_data = B.ravel()
    indices = A_data > 0
    A_data = A_data[indices]
    B_data_remaining = B_data[~indices]
    B_data = B_data[indices]

    klres = np.sum(A_data*np.log(A_data/B_data) - A_data + B_data)
    klres = klres + np.sum(B_data_remaining)
    
    fres = np.linalg.norm(A - B, ord=None)
    fres = fres**2 / 2
    if square_root:
        klres = np.sqrt(2*klres)
        fres = np.sqrt(2*fres)

    return klres, fres


def normalize_WH(W, H):
    normalization_factor = np.sum(W, 0)
    return W/normalization_factor, H*normalization_factor[:, None]
####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# Based on: https://ieeexplore.ieee.org/document/9177681

import numpy as np
from scipy.spatial.distance import cdist


def MM_sampling(S, h):
    """
        S: input dataset, n x d
        h: # iterations
    """
    F_idx = (-1 * np.ones(h)).astype(np.int32)
    F = np.zeros((h, S.shape[1]))
    # random initialization
    F_idx[0] = np.random.choice(S.shape[0])
    F[0] = S[F_idx[0]]
    Z = cdist(S, [F[0]]).ravel()   # current min dist to all selected centers
    all_maximin_dists = np.zeros(h)
    for t in range(1, h):
        d2 = cdist(S, [F[t-1]]).ravel()
        Z = np.minimum(Z, d2)
        new_idx = np.argmax(Z)
        new_min_dist = Z[new_idx]
        all_maximin_dists[t] = new_min_dist
        F_idx[t] = new_idx
        F[t] = S[new_idx]
    return F_idx, F, all_maximin_dists

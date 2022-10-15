####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

import numpy as np
from detecta import detect_peaks


def determine_threshold(neighbor_recorder):
    unique_counts = np.unique(neighbor_recorder.data)  # sorted unique elements
    unique_counts = np.concatenate(([0], unique_counts))
    if len(unique_counts) <= 1:
        return -1
    diff_counts = np.array([unique_counts[i] - unique_counts[i - 1] for i in range(1, len(unique_counts))])
    peak = detect_peaks(diff_counts, mpd=len(diff_counts) / 2)
    if len(diff_counts) == 0:
        return -1
    max_gap = np.max(diff_counts)
    if max_gap == 1 or len(peak) == 0:  # no significant difference
        return -1
    else:
        peak_idx = peak[0]
        return unique_counts[peak_idx] - 1


def find_clusters(neighbor_recorder, only_cluster=True, threshold=None):
    # neighbor: CSC matrix
    if threshold is None or threshold < 0:
        threshold = 0
    seen = set()
    nnz_idx_row, nnz_idx_col = neighbor_recorder.nonzero()
    nnz_idx = np.unique(nnz_idx_row)
    groups = []
    def _dfs(i, visited):
        if i in visited:
            return
        visited.add(i)
        tmp = neighbor_recorder.getrow(i).toarray().ravel()
        nbs = np.where(tmp > threshold)[0]
        if len(nbs) == 0:
            return
        for nn_idx in nbs:
            _dfs(nn_idx, visited)

    for j in nnz_idx:
        if j not in seen:
            vst = set()
            _dfs(j, vst)
            if only_cluster:
                if len(vst) > 1:
                    groups.append(vst)
            else:
                groups.append(vst)
            seen = seen.union(vst)
    return groups




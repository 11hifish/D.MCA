####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# This script contains two versions of the proposed DMCA algorithm:
# - DMCA_0: single stage, no hyperensemble
# - DMCA (Improved DMCA_0, robust to hyperparameters): the ultimate two-stage algorithm: hyperensemble + DMCA_0

import numpy as np
from scipy.spatial.distance import cdist
from src.maximin_sampling import MM_sampling
from kneed import KneeLocator
from src.find_clusters import find_clusters, determine_threshold
from src.inne_python import pyinne_single
from scipy.sparse import csr_matrix
from detecta import detect_peaks

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


# util functions
def compute_area(sorted_dist, sorted_score, max_rad, avg_line=None):
    # sorted dist: dist to ano center sorted
    # sorted score: sorted avg. neighbor anomalous score
    # avg line: average line such that score < avg line will incur an area of 0
    area = 0
    if max_rad < 0:
        raise Exception('value max_radius must >= 0!')
    valid_idx = np.where(sorted_dist <= max_rad)[0]  # can't be []
    idx_threshold = np.max(valid_idx)
    sorted_dist_valid = sorted_dist[:idx_threshold+1]
    sorted_score_valid = sorted_score[:idx_threshold+1]
    # append score up to max rad (for fair comparison)
    sorted_score_valid = np.append(sorted_score_valid, sorted_score_valid[-1])
    sorted_dist_valid = np.append(sorted_dist_valid, max_rad)

    if avg_line is not None:
        below_avg_idx = np.where(sorted_score_valid < avg_line)[0]
        if len(below_avg_idx) > 0:
            first_below_avg_idx = below_avg_idx[0]
            sorted_score_valid[first_below_avg_idx:] = 0

    for i in range(1, len(sorted_dist_valid)):
        dist_diff = sorted_dist_valid[i] - sorted_dist_valid[i - 1]
        weight = 0.5 * (sorted_dist_valid[i] + sorted_score_valid[i-1])
        sc = sorted_score[i - 1]
        area += dist_diff * sc * weight
    return area


def get_top_ano_score_points(ano_scores, num_prune):
    sorted_idx = np.argsort(ano_scores)
    max_score = ano_scores[sorted_idx[-1]]
    max_score_id = np.where(ano_scores == max_score)[0]
    check_pt_id = sorted_idx[-num_prune:]
    if len(max_score_id) > num_prune:
        all_ano_pts = np.union1d(max_score_id, check_pt_id)
        check_pt_id = np.random.choice(all_ano_pts, size=num_prune, replace=False)
    return check_pt_id


def get_F2_representative_set(X, check_pt_id):
    check_pt = X[check_pt_id]
    # get representative set F2
    # make sure we do need MM sampling to remove redundant points to gain efficiency,
    # otherwise we can just include them all
    if len(check_pt) >= 10:
        # Maximin sampling
        F2_idx, F2, maximin_dists = MM_sampling(check_pt, len(check_pt))
        kneedle = KneeLocator(np.arange(len(maximin_dists[1:])), maximin_dists[1:],
                              S=1, online=False, curve="convex", direction="decreasing")
        if kneedle.elbow is None:
            cthres = len(F2_idx)
        else:
            cthres = round(kneedle.elbow)

        ano_candidate_idx_relative = F2_idx[:cthres]
        F2_id = check_pt_id[ano_candidate_idx_relative]  # idx in X
        if len(F2_id) == 0:
            F2_id = check_pt_id
    else:
        F2_id = check_pt_id
    return F2_id


def get_max_rad(F2_pt, center_pts):
    candidate_to_centers = cdist(F2_pt, center_pts, metric='euclidean')
    max_rad = np.max(np.min(candidate_to_centers, axis=1))
    return max_rad


def compute_neighbors_threshold(sorted_dist, max_rad):
    valid_idx = np.where(sorted_dist <= max_rad)[0]
    sorted_dist_2 = sorted_dist[valid_idx]
    diff_dist = np.array([sorted_dist_2[j] - sorted_dist_2[j - 1] for j in range(1, len(sorted_dist_2))])
    diff_dist = np.concatenate(([0], diff_dist))
    peaks = detect_peaks(diff_dist, mpd=30)
    if len(peaks) == 0:  # no local peaks, mark self
        max_peak_idx = 1
    else:
        max_peak_idx = peaks[0]
    return max_peak_idx


def find_TP_and_neighbors(X, F2_id, center_id, psi, ano_scores, max_rad):
    alpha = min(psi, len(F2_id))  # number of additional points to be included in area
    if alpha < psi:
        normal_rep_id = np.random.choice(center_id, size=alpha, replace=False)
    else:
        normal_rep_id = center_id
    all_areas = np.zeros(len(F2_id) + alpha)

    neighbors = {}
    U = np.concatenate((F2_id, normal_rep_id))  # F2 union selected normal samples
    for i, sample_id in enumerate(U):
        dist_to_all = cdist(X, [X[sample_id]], metric='euclidean').ravel()
        dist_to_all_argsorted = np.argsort(dist_to_all)
        dist_to_all_sorted = dist_to_all[dist_to_all_argsorted]

        # ready for computing weighted area and finding FPs
        iscore_sorted = ano_scores[dist_to_all_argsorted]
        avg_neighbor_iscore = np.cumsum(iscore_sorted) / (np.arange(len(iscore_sorted)) + 1)
        # compute area under the curve
        area = compute_area(dist_to_all_sorted, avg_neighbor_iscore, max_rad)
        all_areas[i] = area

        # mark neighbors (only for points in F2, not in normal representatives)
        if i < len(F2_id):
            max_peak_idx = compute_neighbors_threshold(dist_to_all_sorted, max_rad)
            # get neighbors' id
            nn_id = dist_to_all_argsorted[:max_peak_idx]
            neighbors[sample_id] = nn_id

    # determine TP, FP
    mean_area = np.mean(all_areas)
    prune_pt_idx = np.where(all_areas > mean_area)
    TP_id = U[prune_pt_idx]
    return TP_id, neighbors


def prune_pts_and_record_neighbors(n, TP_id, neighbors, neighbor_recorder):
    # neighbor recorder: sparse kernel matrix
    prune_pt_id = []
    for tpid in TP_id:
        if tpid in neighbors:
            all_neighbors_fd = neighbors[tpid]
            prune_pt_id.append(all_neighbors_fd)
            tmp = np.zeros(n)
            tmp[all_neighbors_fd] = 1
            tmp_sparse = csr_matrix(tmp)
            outer = tmp_sparse.reshape(n, 1).dot(tmp_sparse)
            neighbor_recorder += outer
        else:
            neighbor_recorder[tpid, tpid] = 1
    if len(prune_pt_id) == 0:
        R_idx = np.arange(n)
    else:
        prune_pt_id = np.unique(np.concatenate(prune_pt_id))
        R_idx = np.setdiff1d(np.arange(n), prune_pt_id)
    return R_idx, neighbor_recorder


def DMCA_single_model_subroutine(X, center_id, ano_scores, num_prune, psi, neighbor_recorder):
    # get F2: the representative set
    check_pt_id = get_top_ano_score_points(ano_scores, num_prune)
    F2_id = get_F2_representative_set(X, check_pt_id)
    max_rad = get_max_rad(X[F2_id], X[center_id])
    TP_id, neighbors = find_TP_and_neighbors(X, F2_id, center_id, psi, ano_scores, max_rad)
    R_idx, neighbor_recorder = prune_pts_and_record_neighbors(X.shape[0], TP_id, neighbors, neighbor_recorder)
    return R_idx, neighbor_recorder


## DMCA-0
def DMCA_0(X, t=100, psi=128, p=0.1):
    """
        Neighor anomalous scores + change point detection
        X: training and testing data
        t: # ensembles
        psi: subsample size
        p: percentage of data with the highest anomalous score to be considered in F
    """
    n = X.shape[0]
    num_prune = int(p * n)
    iscore_all = np.zeros((n, t))
    R_idx = np.arange(n)
    neighbor_recorder = csr_matrix((n, n), dtype=np.int32)

    for model_idx in range(t):
        print('[Model {}]'.format(model_idx))
        R = X[R_idx]
        center_id, center_1nn_id, center_1nn_dist, iscore, all_dists = pyinne_single(R, X, psi)
        iscore_all[:, model_idx] = iscore
        # get mean iscore upto date
        iscore_upto_current = np.mean(iscore_all[:, :model_idx + 1], axis=1)
        # decide on TPs and mark neighbors
        R_idx, neighbor_recorder = DMCA_single_model_subroutine(X=X, center_id=center_id,
                                                                ano_scores=iscore_upto_current,
                                                                num_prune=num_prune, psi=psi,
                                                                neighbor_recorder=neighbor_recorder)
    return np.mean(iscore_all, axis=1), neighbor_recorder


## The ULTIMATE DMCA ALGORITHM with 2-phase hyperensemble !!!!
def split_two_phases(t, psi):
    phase1_t = int(t / 2)
    phase1_psi_LB, phase1_psi_UB = 2, psi
    if phase1_psi_UB - phase1_psi_LB + 1 <= phase1_t:
        phase1_t = phase1_psi_UB - phase1_psi_LB + 1
        phase1_psis = np.arange(phase1_psi_LB, phase1_psi_UB + 1)
    else:
        delta = (phase1_psi_UB + 1 - phase1_psi_LB) / phase1_t
        phase1_psis = np.arange(phase1_psi_LB, phase1_psi_UB + 1, delta)
        phase1_psis = np.unique(np.rint(phase1_psis)).astype(np.int32)
        phase1_t = len(phase1_psis)
    assert(len(phase1_psis) == phase1_t)
    print('# phase 1 iterations: ', phase1_t)
    return phase1_t, phase1_psis


def DMCA(X, t, psi, p=0.1):
    n = X.shape[0]
    num_prune = int(p * n)
    # split t into 2 phases and determine the psi values for phase 1
    phase1_t, phase1_psis = split_two_phases(t, psi)
    neighbor_recorder = csr_matrix((n, n), dtype=np.int32)

    ## Phase 1: vary psi, only to get cluster info, no pruning
    for model_idx in range(phase1_t):
        current_psi = phase1_psis[model_idx]
        print('[Model {}], current psi: {}'.format(model_idx, current_psi))
        center_id, center_1nn_id, center_1nn_dist, iscore, all_dists = pyinne_single(X, X, current_psi)
        _, neighbor_recorder = DMCA_single_model_subroutine(X=X, center_id=center_id, ano_scores=iscore,
                                                            num_prune=num_prune, psi=current_psi,
                                                            neighbor_recorder=neighbor_recorder)
    # end of phase 1, get clusters/groups and ready to prune in phase 2
    threshold = determine_threshold(neighbor_recorder)
    if threshold < 0:
        groups = {}
    else:
        groups = find_clusters(neighbor_recorder=neighbor_recorder, only_cluster=True, threshold=threshold)
    print('groups: ', groups)
    if len(groups) == 0:
        R_idx = np.arange(n)
    else:
        prune_pt_id = np.array(list(set.union(*groups)))
        R_idx = np.setdiff1d(np.arange(n), prune_pt_id)
    ## Phase 2: fix psi, start pruning
    iscore_phase2 = np.zeros((n, t - phase1_t))
    for model_idx in range(phase1_t, t):
        print('[Model {}], phase 2 psi: {}'.format(model_idx, psi))
        R = X[R_idx]
        center_id, center_1nn_id, center_1nn_dist, iscore, all_dists = pyinne_single(R, X, psi)
        iscore_phase2[:, model_idx - phase1_t] = iscore
        # get mean iscore upto date and compute F
        iscore_upto_current = np.mean(iscore_phase2[:, :model_idx + 1], axis=1)
        R_idx, neighbor_recorder = DMCA_single_model_subroutine(X=X, center_id=center_id,
                                                                ano_scores=iscore_upto_current,
                                                                num_prune=num_prune, psi=psi,
                                                                neighbor_recorder=neighbor_recorder)
    # end of phase 2
    return np.mean(iscore_phase2, axis=1), neighbor_recorder

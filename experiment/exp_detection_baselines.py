####################################
# Author: Shuli Jiang              #
# Email	: shulij@andrew.cmu.edu    #
####################################

# Compute baseline detectors' detection performance in AUC / AP.

import argparse
import numpy as np
from src.utils import load_dataset, convert_to_binary_label
from src.inne_python import pyinne
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from isotree import IsolationForest
from gen2out.gen2out import gen2Out
from experiment.hyperparameters import get_hyperparameter


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--method', type=str, default='copod',
                    help='Outlier detection method')
parser.add_argument('--dataset', type=str, default='synthetic_1_6_2',
                    help='Dataset')
parser.add_argument('--num-exp', type=int, default=5,
                    help='# random runs (only for non-deterministic methods)')
args = parser.parse_args()

# prepare dataset
X, y = load_dataset(args.dataset)
bin_y = convert_to_binary_label(y)

# hyperparameters
psi_list, k_list = get_hyperparameter(args.method, X.shape[0])
num_params = max(1, len(psi_list), len(k_list))

if args.method.lower() in ['copod', 'hbos', 'lof', 'knn']:  # determinisitc
    num_exp = 1
else:
    num_exp = args.num_exp

print('num exp: ', num_exp)
# start training
all_auc = np.zeros((num_exp, num_params))
all_ap = np.zeros((num_exp, num_params))

for exp_idx in range(num_exp):
    if args.method.lower() == 'copod':
        model = COPOD().fit(X)
        ano_score = model.decision_scores_
        auc_sc = roc_auc_score(bin_y, ano_score)
        ap_sc = average_precision_score(bin_y, ano_score)
        all_auc[exp_idx, 0] = auc_sc
        all_ap[exp_idx, 0] = ap_sc
    elif args.method.lower() == 'hbos':
        model = HBOS(n_bins='auto').fit(X)
        ano_score = model.decision_scores_
        auc_sc = roc_auc_score(bin_y, ano_score)
        ap_sc = average_precision_score(bin_y, ano_score)
        all_auc[exp_idx, 0] = auc_sc
        all_ap[exp_idx, 0] = ap_sc
    elif args.method.lower() in ['iforest', 'inne', 'iforest_star', 'sciforest', 'sciforest_star']:
        print('psi list: ', psi_list)
        print('exp no: {}'.format(exp_idx))
        for i, psi_ in enumerate(psi_list):
            if args.method.lower() in ['iforest', 'iforest_star']:
                print('[TREE] method: {}'.format(args.method.lower()))
                model = IForest(n_estimators=100, max_samples=psi_).fit(X)
                ano_score = model.decision_scores_
            elif args.method.lower() == 'inne':
                ano_score = pyinne(X, X, t=100, psi=psi_)
            elif args.method.lower() in ['sciforest', 'sciforest_star']:
                print('[SC-TREE] method: {}'.format(args.method.lower()))
                model = IsolationForest(ndim=2, sample_size=int(psi_), max_depth=None,
                                        ntrees=100, missing_action="fail",
                                        coefs="normal", ntry=10,
                                        prob_pick_avg_gain=1, penalize_range=True).fit(X)
                ano_score = model.predict(X)
                # print('SCiForest ano score: ', ano_score)
            else:
                raise Exception('method should not go here!')
            param_auc = roc_auc_score(bin_y, ano_score)
            param_ap = average_precision_score(bin_y, ano_score)
            all_auc[exp_idx, i] = param_auc
            all_ap[exp_idx, i] = param_ap
            print('i: {}, psi_: {}, AUC: {:.4f}, AP: {:.4f}'
                  .format(i, psi_, param_auc, param_ap))
    elif args.method.lower() in ['lof', 'knn']:
        print('k_list: ', k_list)
        param_auc = np.zeros(len(k_list))
        param_ap = np.zeros(len(k_list))
        print('exp no: {}'.format(exp_idx))
        for i, k_ in enumerate(k_list):
            if args.method.lower() == 'lof':
                model = LOF(n_neighbors=k_).fit(X)
            else:
                model = KNN(n_neighbors=k_).fit(X)
            ano_score = model.decision_scores_
            param_auc = roc_auc_score(bin_y, ano_score)
            param_ap = average_precision_score(bin_y, ano_score)
            all_auc[exp_idx, i] = param_auc
            all_ap[exp_idx, i] = param_ap
            print('i: {}, k_: {}, AUC: {:.4f}, AP: {:.4f}'
                  .format(i, k_, param_auc, param_ap))
    elif args.method.lower() == 'gen2out':
        model = gen2Out().fit(X)
        ano_score = model.decision_function(X)
        all_auc[exp_idx, 0] = roc_auc_score(bin_y, ano_score)
        all_ap[exp_idx, 0] = average_precision_score(bin_y, ano_score)
    else:
        raise Exception('What method???')

print('all auc: ')
print(all_auc)
print('all ap: ')
print(all_ap)
print('[Model] {}, [Data] {}: AUC {:.4f} ({:.4f}), AP {:.4f} ({:.4f})'
      .format(args.method, args.dataset,
              np.mean(all_auc), np.std(all_auc), np.mean(all_ap), np.std(all_ap)))

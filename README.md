# D.MCA: Outlier Detection with Explicit Micro-Cluster Assignments

by
Shuli Jiang, 
Robson L. F. Cordeiro, 
and Leman Akoglu

## To Cite
    @INPROCEEDINGS{DMCA2022,
      author={Shuli Jiang, Robson L. F. Cordeiro, Leman Akoglu},
      booktitle={Twenty Second IEEE International Conference on Data Mining}, 
      title={D.MCA: Outlier Detection with Explicit Micro-Cluster Assignments}, 
      year={2022}
    }

## Abstract

> How can we detect outliers, both scattered and clustered, and also explicitly assign them to respective micro-clusters, without knowing apriori how many micro-clusters exist?
    How can we perform both tasks in-house, i.e., without any post-hoc processing, so that both detection and assignment can benefit simultaneously from each other?
    Presenting outliers in separate micro-clusters is informative to analysts in many real-world applications.
    However, a naive solution based on post-hoc clustering of the outliers detected by any existing method suffers from two main drawbacks:
    (a) appropriate hyperparameter values are commonly unknown for clustering, and most algorithms struggle with clusters of varying shapes and densities;
    (b) detection and assignment cannot benefit from one another.
    In this paper, we propose D.MCA to <ins>D</ins>etect outliers with explicit <ins>M</ins>icro-<ins>C</ins>luster <ins>A</ins>ssignment. 
    Our method performs both detection and assignment iteratively, and in-house, by using a novel strategy that prunes entire micro-clusters out of the training set to improve the performance of the detection.
    It also benefits from a novel strategy that avoids clustered outliers to mask each other, which is a well-known problem in the literature.
    Also, D.MCA is designed to be robust to a critical hyperparameter by employing a hyperensemble "warm up" phase.
    Experiments performed on 16 real-world and synthetic datasets demonstrate that D.MCA outperforms 8 state-of-the-art competitors, especially on the explicit outlier micro-cluster assignment task.


## Datasets

Datasets used in the experiments fall into three categories:
* ``2D synthetic dataset``
* ``Semi-synthetic dataset``
* ``Real-world dataset``

All datasets used in the experiments are available [here on Google Drive](https://drive.google.com/drive/folders/1GIqMDtVjGpYicZEtP1T06AuZc2b8hpPF?usp=sharing)


## Dependencies

Python 3.7+ is preferred. See ``requirements.txt`` for all dependencies. You can do

    pip install -r requirements.txt

to install all dependencies.

## A brief overview of the repository structure
* ``src/`` contains all source files, including:
  * ``DMCA.py``: scipts of the proposed D.MCA/D.MCA-0 algorithm.
  * ``find_clusters.py``, ``maximin_sampling.py``, ``utils.py``: subroutines used by D.MCA/D.MCA-0.
  * ``inne_python.py``: implements a python version of the base iNNE detector.
  * ``cluster_metric.py``: to compute the F1 score used for outlier micro-clusters assignment evaluation
  * ``DMCA_extract_masking_effect.py``: a script based on ``DMCA.py`` used for evaluating the cumulative masking effects of iNNE, D.MCA and D.MCA-0.
* ``experiment/`` contains all the files to perform the experiments:
  * ``hyperparameters.py``: to get the set of hyperparameters for each baseline and D.MCA/D.MCA-0 used in the experiments.
  * ``exp_DMCA.py``: to run experiments with D.MCA/D.MCA-0.
  * ``exp_assignment_baselines.py``, ``exp_assignment_DMCA.py``: to evaluate the assignment performance of the baselines and D.MCA/D.MCA-0.
  * ``exp_detection_baselines.py``, ``exp_detection_DMCA.py``: to evaluate the detection performance of the baselines and D.MCA/D.MCA-0.
* ``gen2out/``: Source code of the Gen2Out algorithm (one of our baselines) from https://github.com/mengchillee/gen2Out.

## Running experiments

Before running the experiments:

Make a ``data/`` folder and put all downloaded datasets in this folder.

Then run:

    pip install -e .

To run D.MCA and save the results into a folder ``results/``,

    python experiment/exp_DMCA.py --method DMCA --dataset <dataset name> --num-exp 1 --save-path results

Similarly, to run D.MCA-0 and save the results into a folder ``results/``, 

    python experiment/exp_DMCA.py --method DMCA_0 --dataset <dataset name> --num-exp 1 --save-path results

Note that D.MCA and D.MCA-0 can take a while to run -- they are not optimized for parallel execution.  

The saved result in ``results/`` for each run and 
each hyperparameter (hyperparameters are determined by ``src/hyperparameters.py``) 
contains: 1) outlier scores for each sample in the dataset, 
and 2) the weighted neighbor graph G.

Once the results are in place, to evaluate the detection performance of D.MCA / D.MCA-0, 

    python experiment/exp_detection_DMCA.py --method <DMCA or DMCA_0> --dataset <dataset name> --num-exp 1 --load-path results

To evaluate the assignment performance of D.MCA / D.MCA-0,

    python experiment/exp_assignment_DMCA.py --method <DMCA or DMCA_0> --dataset <dataset name> --num-exp 1 --load-path results

To run and evaluate the detection performance of one baseline,

    python experiment/exp_detection_baselines.py --method <baseline name> --dataset <dataset name> --num-exp 1

To run and evaluate the assignment performance of one baseline,

    python experiment/exp_assignment_baselines.py --method <baseline name> --dataset <dataset name> --num-exp 1


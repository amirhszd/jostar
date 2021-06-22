# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:57:23 2021

@author: Amirh
"""

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import r2_score
from jostar.algorithms import ACO, GA, SA, PSO, PlusLMinusR, DE, NSGA2, SBS, SFS
from sklearn.svm import SVR, SVC
from sklearn.base import is_classifier, is_regressor
from sklearn.model_selection import KFold
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib



def eval_opt_model_output_regression(opt_model, n_f):
    rank_models = ["GA", "SA", "PSO", "ACO", "DE"]
    seq_models = ["PlusLMinusR", "SFS", "SBS"]
    if opt_model._name_ in rank_models:
        assert len(opt_model.best_fits) == 1
        assert len(opt_model.best_sol) == n_f
        assert is_regressor(opt_model.model_best)
        assert len(opt_model.rankings) == 2
        assert opt_model.display_results() is not None
    elif opt_model._name_ in seq_models:
        if opt_model._name_ != "SBS":
            assert len(opt_model.best_fits) == n_f
            assert len(opt_model.best_sol) == n_f
            assert is_regressor(opt_model.model_best)
            assert opt_model.display_results() is not None
        else:
            assert len(opt_model.best_sol) == n_f
            assert is_regressor(opt_model.model_best)
            assert opt_model.display_results() is not None
    else:
        result_df = opt_model.res_df
        assert isinstance(result_df, pd.core.frame.DataFrame)
        assert opt_model.display_results(0) is not None
    plt.close('all')
    pbar.update(1)


def eval_opt_model_output_classification(opt_model, n_f):
    rank_models = ["GA", "SA", "PSO", "ACO", "DE"]
    seq_models = ["PlusLMinusR", "SFS", "SBS"]
    if opt_model._name_ in rank_models:
        assert len(opt_model.best_fits) == 1
        assert len(opt_model.best_sol) == n_f
        assert is_classifier(opt_model.model_best)
        rank_models = ["GA", "SA", "PSO", "ACO", "DE"]
        if opt_model._name_ in rank_models:
            assert len(opt_model.rankings) == 2
        assert opt_model.display_results() is not None
    elif opt_model._name_ in seq_models:
        if opt_model._name_ != "SBS":
            assert len(opt_model.best_fits) == n_f
            assert len(opt_model.best_sol) == n_f
            assert is_classifier(opt_model.model_best)
            assert opt_model.display_results() is not None
        else:
            assert len(opt_model.best_sol) == n_f
            assert is_classifier(opt_model.model_best)
            assert opt_model.display_results() is not None
    else:
        result_df = opt_model.res_df
        assert isinstance(result_df, pd.core.frame.DataFrame)
        assert opt_model.display_results(0) is not None
    plt.close('all')
    pbar.update(1)


def test_all_regression():
    global pbar
    cv = KFold(5)
    n_f = 5
    x, y = make_regression(100, 10)
    model = SVR()

    # regression
    # with CV
    pbar = tqdm(total=18)
    ga_opt_model = GA(model, n_f, +1, r2_score, n_gen=1,
                      n_pop=20, cv=cv, verbose=False)
    sa_opt_model = SA(model, n_f, +1, r2_score, n_iter=1,
                      n_sub_iter=20, cv=cv, verbose=False)
    de_opt_model = DE(model, n_f, +1, r2_score, n_iter=1,
                      n_pop=20, cv=cv, verbose=False)
    aco_opt_model = ACO(model, n_f, +1, r2_score, n_iter=1,
                        n_ant=20, cv=cv, verbose=False)
    pso_opt_model = PSO(model, n_f, +1, r2_score, n_iter=1,
                        n_pop=20, cv=cv, verbose=False)
    lrs_opt_model = PlusLMinusR(model, n_f, +1, r2_score, cv=cv, verbose=False)
    nsga_opt_model = NSGA2(model, n_f, (+1, -1), r2_score,
                           n_gen=1, n_pop=20, cv=cv, verbose=False)
    sbs_opt_model = SBS(model, n_f, +1, r2_score, cv=cv, verbose=False)
    sfs_opt_model = SFS(model, n_f, +1, r2_score, cv=cv, verbose=False)

    ga_opt_model.fit(x, y, decor=0.95, scale=True)
    sa_opt_model.fit(x, y, decor=0.95, scale=True)
    de_opt_model.fit(x, y, decor=0.95, scale=True)
    aco_opt_model.fit(x, y, decor=0.95, scale=True)
    pso_opt_model.fit(x, y, decor=0.95, scale=True)
    lrs_opt_model.fit(x, y, decor=0.95, scale=True)
    nsga_opt_model.fit(x, y, decor=0.95, scale=True)
    sbs_opt_model.fit(x, y, decor=0.95, scale=True)
    sfs_opt_model.fit(x, y, decor=0.95, scale=True)

    eval_opt_model_output_regression(ga_opt_model, n_f)
    eval_opt_model_output_regression(sa_opt_model, n_f)
    eval_opt_model_output_regression(de_opt_model, n_f)
    eval_opt_model_output_regression(aco_opt_model, n_f)
    eval_opt_model_output_regression(pso_opt_model, n_f)
    eval_opt_model_output_regression(lrs_opt_model, n_f)
    eval_opt_model_output_regression(nsga_opt_model, n_f)
    eval_opt_model_output_regression(sbs_opt_model, n_f)
    eval_opt_model_output_regression(sfs_opt_model, n_f)

    # with test size
    ga_opt_model = GA(model, n_f, +1, r2_score, n_gen=1,
                      n_pop=20, cv=None, verbose=False)
    sa_opt_model = SA(model, n_f, +1, r2_score, n_iter=1,
                      n_sub_iter=20, cv=None, verbose=False)
    de_opt_model = DE(model, n_f, +1, r2_score, n_iter=1,
                      n_pop=20, cv=None, verbose=False)
    aco_opt_model = ACO(model, n_f, +1, r2_score, n_iter=1,
                        n_ant=20, cv=None, verbose=False)
    pso_opt_model = PSO(model, n_f, +1, r2_score, n_iter=1,
                        n_pop=20, cv=None, verbose=False)
    lrs_opt_model = PlusLMinusR(
        model, n_f, +1, r2_score, cv=None, verbose=False)
    nsga_opt_model = NSGA2(model, n_f, (+1, -1), r2_score,
                           n_gen=1, n_pop=20, cv=None, verbose=False)
    sbs_opt_model = SBS(model, n_f, +1, r2_score, cv=None, verbose=False)
    sfs_opt_model = SFS(model, n_f, +1, r2_score, cv=None, verbose=False)

    ga_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    sa_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    de_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    aco_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    pso_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    lrs_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    nsga_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    sbs_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    sfs_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)

    eval_opt_model_output_regression(ga_opt_model, n_f)
    eval_opt_model_output_regression(sa_opt_model, n_f)
    eval_opt_model_output_regression(de_opt_model, n_f)
    eval_opt_model_output_regression(aco_opt_model, n_f)
    eval_opt_model_output_regression(pso_opt_model, n_f)
    eval_opt_model_output_regression(lrs_opt_model, n_f)
    eval_opt_model_output_regression(nsga_opt_model, n_f)
    eval_opt_model_output_regression(sbs_opt_model, n_f)
    eval_opt_model_output_regression(sfs_opt_model, n_f)


def test_all_classification():
    global pbar
    cv = KFold(5)
    n_f = 5
    x, y = make_classification(100, 10)
    model = SVC(probability=True)

    # regression
    # with CV
    pbar = tqdm(total=18)
    ga_opt_model = GA(model, n_f, +1, r2_score, n_gen=1,
                      n_pop=20, cv=cv, verbose=False)
    sa_opt_model = SA(model, n_f, +1, r2_score, n_iter=1,
                      n_sub_iter=20, cv=cv, verbose=False)
    de_opt_model = DE(model, n_f, +1, r2_score, n_iter=1,
                      n_pop=20, cv=cv, verbose=False)
    aco_opt_model = ACO(model, n_f, +1, r2_score, n_iter=1,
                        n_ant=20, cv=cv, verbose=False)
    pso_opt_model = PSO(model, n_f, +1, r2_score, n_iter=1,
                        n_pop=20, cv=cv, verbose=False)
    lrs_opt_model = PlusLMinusR(model, n_f, +1, r2_score, cv=cv, verbose=False)
    nsga_opt_model = NSGA2(model, n_f, (+1, -1), r2_score,
                           n_gen=1, n_pop=20, cv=cv, verbose=False)
    sbs_opt_model = SBS(model, n_f, +1, r2_score, cv=cv, verbose=False)
    sfs_opt_model = SFS(model, n_f, +1, r2_score, cv=cv, verbose=False)

    ga_opt_model.fit(x, y, decor=0.95, scale=True)
    sa_opt_model.fit(x, y, decor=0.95, scale=True)
    de_opt_model.fit(x, y, decor=0.95, scale=True)
    aco_opt_model.fit(x, y, decor=0.95, scale=True)
    pso_opt_model.fit(x, y, decor=0.95, scale=True)
    lrs_opt_model.fit(x, y, decor=0.95, scale=True)
    nsga_opt_model.fit(x, y, decor=0.95, scale=True)
    sbs_opt_model.fit(x, y, decor=0.95, scale=True)
    sfs_opt_model.fit(x, y, decor=0.95, scale=True)

    eval_opt_model_output_classification(ga_opt_model, n_f)
    eval_opt_model_output_classification(sa_opt_model, n_f)
    eval_opt_model_output_classification(de_opt_model, n_f)
    eval_opt_model_output_classification(aco_opt_model, n_f)
    eval_opt_model_output_classification(pso_opt_model, n_f)
    eval_opt_model_output_classification(lrs_opt_model, n_f)
    eval_opt_model_output_classification(nsga_opt_model, n_f)
    eval_opt_model_output_classification(sbs_opt_model, n_f)
    eval_opt_model_output_classification(sfs_opt_model, n_f)

    # with test size
    ga_opt_model = GA(model, n_f, +1, r2_score, n_gen=1,
                      n_pop=20, cv=None, verbose=False)
    sa_opt_model = SA(model, n_f, +1, r2_score, n_iter=1,
                      n_sub_iter=20, cv=None, verbose=False)
    de_opt_model = DE(model, n_f, +1, r2_score, n_iter=1,
                      n_pop=20, cv=None, verbose=False)
    aco_opt_model = ACO(model, n_f, +1, r2_score, n_iter=1,
                        n_ant=20, cv=None, verbose=False)
    pso_opt_model = PSO(model, n_f, +1, r2_score, n_iter=1,
                        n_pop=20, cv=None, verbose=False)
    lrs_opt_model = PlusLMinusR(
        model, n_f, +1, r2_score, cv=None, verbose=False)
    nsga_opt_model = NSGA2(model, n_f, (+1, -1), r2_score,
                           n_gen=1, n_pop=20, cv=None, verbose=False)
    sbs_opt_model = SBS(model, n_f, +1, r2_score, cv=None, verbose=False)
    sfs_opt_model = SFS(model, n_f, +1, r2_score, cv=None, verbose=False)

    ga_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    sa_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    de_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    aco_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    pso_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    lrs_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    nsga_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    sbs_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)
    sfs_opt_model.fit(x, y, decor=0.95, scale=True, test_size=0.3)

    eval_opt_model_output_classification(ga_opt_model, n_f)
    eval_opt_model_output_classification(sa_opt_model, n_f)
    eval_opt_model_output_classification(de_opt_model, n_f)
    eval_opt_model_output_classification(aco_opt_model, n_f)
    eval_opt_model_output_classification(pso_opt_model, n_f)
    eval_opt_model_output_classification(lrs_opt_model, n_f)
    eval_opt_model_output_classification(nsga_opt_model, n_f)
    eval_opt_model_output_classification(sbs_opt_model, n_f)
    eval_opt_model_output_classification(sfs_opt_model, n_f)


if __name__ == '__main__':
    test_all_regression()
    test_all_classification()
    print(" ")
    print("All tests passed!")

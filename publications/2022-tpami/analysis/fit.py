import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def get_num_par(model_id):
    if model_id == 'last1':
        return 1
    if model_id in ['pow2', 'log2', 'exp2', 'lin2', 'ilog2']:
        return 2
    if model_id in ['pow3', 'exp3', 'vap3', 'expp3', 'expd3', 'logpower3']:
        return 3
    if model_id in ['mmf4', 'wbl4', 'exp4', 'pow4']:
        return 4


def fit_model(sizes, scores, sizes_extrapolation, model_id, use_jac=False):
    # print(sizes)
    # print(scores)
    # print(sizes_extrapolation)
    # print(model_id)

    def get_J(beta):
        num_par = get_num_par(model_id)
        if num_par == 2:
            a, b = beta[0], beta[1]
        if num_par == 3:
            a, b, c = beta[0], beta[1], beta[2]
        if num_par == 4:
            a, b, c, d = beta[0], beta[1], beta[2], beta[3]
        X = sizes

        if model_id == 'pow2':
            J = np.array([-X ** (-b), a * X ** (-b) * np.log(X)])
        if model_id == 'pow3':
            J = np.array([np.ones_like(X), -X ** (-c), b * X ** (-c) * np.log(X)])
        if model_id == 'log2':
            J = np.array([-np.log(X), np.ones_like(X)])
        if model_id == 'exp3':
            J = np.array([np.exp(-b * X), -a * X * np.exp(-b * X), np.ones_like(X)])
        if model_id == 'exp2':
            J = np.array([np.exp(-b * X), -a * X * np.exp(-b * X)])
        if model_id == 'lin2':
            J = np.array([X, np.ones_like(X)])
        if model_id == 'vap3':
            J = np.array([np.exp(a + b / X + c * np.log(X)), np.exp(a + b / X + c * np.log(X)) / X,
                          np.exp(a + b / X + c * np.log(X)) * np.log(X)])
        if model_id == 'mmf4':
            J = np.array(
                [b / (b + X ** d), a / (b + X ** d) - (a * b + c * X ** d) / (b + X ** d) ** 2, X ** d / (b + X ** d),
                 c * X ** d * np.log(X) / (b + X ** d) - X ** d * (a * b + c * X ** d) * np.log(X) / (b + X ** d) ** 2])
        if model_id == 'wbl4':
            J = np.array([b * X ** d * np.exp(-a * X ** d), -np.exp(-a * X ** d), np.ones_like(X),
                          a * b * X ** d * np.exp(-a * X ** d) * np.log(X)])
        if model_id == 'exp4':
            J = np.array([X ** d * np.exp(-a * X ** d + b), -np.exp(-a * X ** d + b), np.ones_like(X),
                          a * X ** d * np.exp(-a * X ** d + b) * np.log(X)])
        if model_id == 'expp3':
            J = np.array([-(-b + X) ** a * np.exp((-b + X) ** a) * np.log(-b + X),
                          a * (-b + X) ** a * np.exp((-b + X) ** a) / (-b + X), np.ones_like(X)])
        if model_id == 'pow4':
            J = np.array([np.ones_like(X), -(d + X) ** (-c), b * (d + X) ** (-c) * np.log(d + X),
                          b * c * (d + X) ** (-c) / (d + X)])
        if model_id == 'ilog2':
            J = np.array([-1 / np.log(X), np.ones_like(X)])
        if model_id == 'expd3':
            J = np.array([np.exp(-b * X), -X * (a - c) * np.exp(-b * X), np.ones_like(X) - np.exp(-b * X)])
        if model_id == 'logpower3':
            J = np.array([((X * np.exp(-b)) ** c + np.ones_like(X)) ** (-1.0),
                          a * c * (X * np.exp(-b)) ** c / ((X * np.exp(-b)) ** c + np.ones_like(X)) ** 2,
                          -a * (X * np.exp(-b)) ** c * np.log(X * np.exp(-b)) / (
                                      (X * np.exp(-b)) ** c + np.ones_like(X)) ** 2])
        return J.T

    def get_fun(beta):
        num_par = get_num_par(model_id)
        fun = None

        # unpack parameters
        if num_par == 1:
            a = beta[0]
        if num_par == 2:
            a, b = beta[0], beta[1]
        if num_par == 3:
            a, b, c = beta[0], beta[1], beta[2]
        if num_par == 4:
            a, b, c, d = beta[0], beta[1], beta[2], beta[3]

        # define curve models
        if model_id == 'pow2':
            fun = lambda x: -a * x ** (-b)
        if model_id == 'pow3':
            fun = lambda x: a - b * x ** (-c)
        if model_id == 'log2':
            fun = lambda x: -a * np.log(x) + b
        if model_id == 'exp3':
            fun = lambda x: a * np.exp(-b * x) + c
        if model_id == 'exp2':
            fun = lambda x: a * np.exp(-b * x)
        if model_id == 'lin2':
            fun = lambda x: a * x + b
        if model_id == 'vap3':
            fun = lambda x: np.exp(a + b / x + c * np.log(x))
        if model_id == 'mmf4':
            fun = lambda x: (a * b + c * x ** d) / (b + x ** d)
        if model_id == 'wbl4':
            fun = lambda x: (c - b * np.exp(-a * (x ** d)))
        if model_id == 'exp4':
            fun = lambda x: c - np.exp(-a * (x ** d) + b)
        if model_id == 'expp3':
            # fun = lambda x: a * np.exp(-b*x) + c
            fun = lambda x: c - np.exp((x - b) ** a)
        if model_id == 'pow4':
            fun = lambda x: a - b * (x + d) ** (-c)  # has to closely match pow3
        if model_id == 'ilog2':
            fun = lambda x: b - (a / np.log(x))
        if model_id == 'expd3':
            fun = lambda x: c - (c - a) * np.exp(-b * x)
        if model_id == 'logpower3':
            fun = lambda x: a / (1 + (x / np.exp(b)) ** c)
        if model_id == 'last1':
            fun = lambda x: (a + x) - x  # casts the prediction to have the correct size
        return fun

    def objective(beta):  # objective function to minimize
        fun = get_fun(beta)
        return fun(sizes) - scores

    # we dp multiple repititions and collect best results in lists below
    beta_list = []
    trn_error = []

    # this model requires no optimization
    if model_id == 'last1':
        a = scores[-1]
        return np.array([a]), get_fun(np.array([a])), 0, 0

    # failure statistics
    rep = 5
    fails_fit = 0
    fails_init = 0
    i = 0
    while i <= rep:  # try repeatedly
        num_par = get_num_par(model_id)
        # print('parameters %d' % num_par)

        beta = None
        error = True
        first = True
        # keep trying initial points until a suitable one is found
        while (error):

            if not first:
                fails_init += 1
                print('initial value failed, retrying for ', model_id)
            init = np.random.rand(num_par)

            if model_id == 'pow4':  # this init works well for pow4
                best_beta, _, _, _ = fit_model(sizes, scores, sizes_extrapolation, 'pow3')
                # print(best_beta)
                init[0:3] = best_beta

            # check for errors in initial point
            trn_error_init = np.mean(objective(init) ** 2)
            fun_init = get_fun(init)
            sizes_all = np.hstack((sizes, sizes_extrapolation))
            hat_all = fun_init(sizes_all)
            nan_error1 = np.isnan(hat_all).any()
            inf_error1 = np.isinf(hat_all).any()
            nan_error2 = np.isnan(trn_error_init).any()
            inf_error2 = np.isinf(trn_error_init).any()
            error = nan_error1 or inf_error1 or nan_error2 or inf_error2

            first = False

        # start fitting
        # todo: fitting will be much faster if the hessian and gradient are provided analytically

        if use_jac:
            beta = sp.optimize.least_squares(objective, init, method="lm", jac=get_J).x
        else:
            beta = sp.optimize.least_squares(objective, init, method="lm").x

        # check if fit extrapolates well to unseen sizes
        fun = get_fun(beta)
        extrapolations = fun(sizes_extrapolation)
        nan_error = np.isnan(extrapolations).any()
        inf_error = np.isinf(extrapolations).any()

        if nan_error or inf_error:
            pass  # redo's the optimization since extrapolations failed
            fails_fit += 1
            print('fit failed, nan error?', nan_error, 'inf error?', inf_error, 'model?', model_id)
        else:
            i += 1
            pass  # save the parameter values and objective function
            beta_list.append(beta)
            trn_error.append(np.mean(objective(beta) ** 2))

    # select the best one 
    trn_error = np.array(trn_error)
    best_i = np.argmin(trn_error)
    # print('train error')
    # print(trn_error)
    # print('best index')
    # print(best_i)
    best_beta = beta_list[best_i]
    # print(best_beta)
    return best_beta, get_fun(best_beta), fails_init, fails_fit


def get_multiple_extrapolations_mean_curve_robust(df):
    model_names = ['pow4', 'pow3', 'pow2', 'log2', 'exp2', 'exp3', 'lin2', 'last1', 'vap3', 'mmf4', 'wbl4', 'exp4',
                   'expp3', 'ilog2', 'expd3', 'logpower3']
    # model_names = ['pow3', 'pow2', 'log2', 'exp2', 'exp3', 'lin2', 'last1', 'vap3','mmf4','wbl4','exp4','ilog2','expd3','logpower3']
    rows = []
    pbar = tqdm(total=len(pd.unique(df["openmlid"])) * len(pd.unique(df["learner"])) * len(model_names), smoothing=0,
                miniters=1)
    for openmlid, df_dataset in tqdm(df.groupby("openmlid")):
        for learner, df_learner in df_dataset.groupby("learner"):
            sizes = None
            scores = []
            for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                if sizes is None:
                    sizes = sizes_seed
                scores.append(scores_seed)
            scores = np.array(scores)
            if len(scores.shape) != 2:
                print(f"Skipping {learner}")
                continue
            mean_scores = np.mean(scores, axis=0)
            # sizes, scores = df_seeded["size_train"].values, df_seeded["score_valid"].values
            for i in range(0, len(model_names)):
                model_name = model_names[i]
                # print(model_name)
                for offset in range(4, len(sizes)):
                    beta, model, fails_init, fails_fit = fit_model(np.array(sizes[:offset]),
                                                                   np.array(mean_scores[:offset]),
                                                                   np.array(sizes[offset:]), model_name)
                    predictions = np.round(model(sizes), 4)
                    assert (len(predictions) == len(sizes))
                    rows.append([openmlid, learner, sizes[offset - 1], predictions, model_names[i], beta, fails_init,
                                 fails_fit])
                pbar.update(1)
    pbar.close()
    return pd.DataFrame(rows, columns=["openmlid", "learner", "max_anchor_seen", "prediction", "curve_model", "beta",
                                       "fails_init", "fails_fit"])


def get_anchors_and_scores_mean_curve(df):
    rows = []
    for openmlid, df_dataset in tqdm(df.groupby("openmlid")):
        for learner, df_learner in df_dataset.groupby("learner"):
            sizes = None
            scores = []
            for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                if sizes is None:
                    sizes = sizes_seed
                scores.append(scores_seed)
            scores = np.array(scores)
            if len(scores.shape) != 2:
                print(f"Skipping {learner}")
                continue
            mean_scores = np.mean(scores, axis=0)
            rows.append([openmlid, learner, sizes, mean_scores])
    return pd.DataFrame(rows, columns=["openmlid", "learner", "anchor_prediction", "score"])


def metrics_per_row(row, score, anchor_prediction):
    max_anchor_seen = row.max_anchor_seen
    prediction = row.prediction
    max_anchor = np.max(anchor_prediction)
    percentage_train = np.round(max_anchor_seen / max_anchor * 100) / 100

    trn_ind = np.argwhere(max_anchor_seen == anchor_prediction)[0][0]  # recover offset
    trn_indices = range(0, (trn_ind + 1))
    tst_indices = range(trn_ind + 1, len(anchor_prediction))
    n_trn = len(trn_indices)

    y_trn_hat = prediction[trn_indices]
    y_trn = score[trn_indices]
    y_tst_hat = prediction[tst_indices]
    y_tst = score[tst_indices]

    MSE_trn = np.mean((y_trn - y_trn_hat) ** 2)
    MSE_tst = np.mean((y_tst - y_tst_hat) ** 2)
    MSE_tst_last = (y_tst[-1] - y_tst_hat[-1]) ** 2
    L1_trn = np.mean((y_trn - y_trn_hat) ** 2)
    L1_tst = np.mean((y_tst - y_tst_hat) ** 2)
    L1_tst_last = (y_tst[-1] - y_tst_hat[-1]) ** 2

    return [MSE_trn, MSE_tst, MSE_tst_last, L1_trn, L1_tst, L1_tst_last, max_anchor_seen, percentage_train, n_trn,
            row.curve_model]


def get_info_mean_curve(df_info, openmlid, learner):
    q = df_info.query('openmlid==@openmlid and learner==@learner')
    q = q.iloc[0, :]
    return [q.anchor_prediction, q.score]


def df_compute_metrics_mean_curve(df, df_info):
    pbar = tqdm(total=len(df))
    rows_metrics = []
    for i in range(0, len(df)):
        row = df.iloc[i, :]
        anchor_prediction, score = get_info_mean_curve(df_info, row.openmlid, row.learner)
        rows_metrics.append(metrics_per_row(row, score, anchor_prediction))
        pbar.update(1)
    pbar.close()
    df_metrics = pd.DataFrame(rows_metrics,
                              columns=['MSE trn', 'MSE tst', 'MSE tst last', 'L1 trn', 'L1 tst', 'L1 tst last',
                                       'max anchor seen', 'percentage', 'n', 'curve_model'])
    return df_metrics


def filter_nan_predictions(df_extrapolations):
    rows_filtered = []
    pbar = tqdm(total=len(df_extrapolations))
    for i in range(0, len(df_extrapolations)):
        row = df_extrapolations.iloc[i, :]
        if not np.isnan(row.prediction).any():
            rows_filtered.append(row)
        pbar.update(1)
    pbar.close()
    df_extrapolations_filtered = pd.DataFrame(rows_filtered)
    return df_extrapolations_filtered


def select_part(part, df_all, datasets):
    num = 20
    indices = range(part * num, part * num + num)
    if part == 9:
        indices = range(part * num, len(datasets))
    datasets_todo = []
    for i in indices:
        datasets_todo.append(datasets[i])
    df_selected = df_all.loc[df_all['openmlid'].isin(datasets_todo)]
    return df_selected


def do_job(part):
    print('starting part %d' % part)

    df_all = pd.read_csv("lcdb_new.csv")
    np.random.seed(42)
    datasets = df_all['openmlid'].unique()
    np.random.shuffle(datasets)

    df_selected = select_part(part, df_all, datasets)
    # df_selected = df_all.loc[df_all['openmlid'] == 6]
    print('computing extrapolations...')
    df_extrapolations = get_multiple_extrapolations_mean_curve_robust(df_selected)
    df_extrapolations.to_csv('extrapolations%d.csv' % part)
    print('computing anchors and scores...')
    df_anchors_and_scores = get_anchors_and_scores_mean_curve(df_selected)
    df_anchors_and_scores.to_csv('anchors_scores%d.csv' % part)
    print('computing metrics....')
    df_metrics = df_compute_metrics_mean_curve(df_extrapolations, df_anchors_and_scores)
    df_metrics.to_csv('metrics_tmp%d.csv' % part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", type=int)
    args = parser.parse_args()
    part = args.part
    do_job(part)


if __name__ == "__main__":
    main()

import scipy as sp
import scipy.optimize
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from hashlib import sha256


def get_num_par(model_id):
    if model_id == 'last1':
        return 1
    if model_id in ['pow2', 'log2', 'exp2', 'lin2', 'ilog2']:
        return 2
    if model_id in ['pow3', 'exp3', 'vap3', 'expp3', 'expd3', 'logpower3']:
        return 3
    if model_id in ['mmf4', 'wbl4', 'exp4', 'pow4']:
        return 4


def fit_model(sizes, scores, sizes_extrapolation, model_id, rep=5, verbose=True):
    sizes = np.array(sizes)
    scores = np.array(scores)

    bad_score = np.isnan(scores)

    sizes = sizes[bad_score == False]
    scores = scores[bad_score == False]

    # this defines the curve model
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

    def objective(beta):  # this returns the residuals of the fit on the training points
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
    #rep = 5
    fails_fit = 0
    fails_init = 0
    i = 0
    init = None

    while i <= rep:  # try repeatedly to fit a model
        num_par = get_num_par(model_id)

        beta = None
        error = True
        first = True
        # keep trying initial points until a suitable one is found
        while (error):

            if fails_init > 1000 or fails_fit > 200:  # give up
                best_beta = np.zeros(num_par)
                if verbose:
                    print('giving up...')
                return best_beta, get_fun(best_beta), fails_init, fails_fit

            if not first:
                fails_init += 1
                if verbose:
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
        beta = sp.optimize.least_squares(objective, init, method="lm").x

        # check if fit extrapolates well to unseen sizes
        fun = get_fun(beta)
        extrapolations = fun(sizes_extrapolation)
        nan_error = np.isnan(extrapolations).any()
        inf_error = np.isinf(extrapolations).any()

        if nan_error or inf_error:
            pass  # redo's the optimization since extrapolations failed
            fails_fit += 1
            if verbose:
                print('fit failed, nan error?', nan_error, 'inf error?', inf_error, 'model?', model_id)
        else:
            i += 1
            pass  # save the parameter values and objective function
            beta_list.append(beta)
            trn_error.append(np.mean(objective(beta) ** 2))

    # select the best one
    trn_error = np.array(trn_error)
    best_i = np.argmin(trn_error)

    best_beta = beta_list[best_i]
    return best_beta, get_fun(best_beta), fails_init, fails_fit


def get_multiple_extrapolations_mean_curve_robust(df, rep, verbose):
    model_names = ['pow4', 'pow3', 'pow2', 'log2', 'exp2', 'exp3', 'lin2', 'last1', 'vap3', 'mmf4', 'wbl4', 'exp4',
                   'expp3', 'ilog2', 'expd3', 'logpower3']
    rows = []
    pbar = tqdm(total=len(pd.unique(df["openmlid"])) * len(pd.unique(df["learner"])) * len(model_names), smoothing=0,
                miniters=1)
    for openmlid, df_dataset in df.groupby("openmlid"):
        for learner, df_learner in df_dataset.groupby("learner"):
            sizes = sorted(pd.unique(df_learner["size_train"]))
            scores = []
            for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                scores.append([scores_seed[list(sizes_seed).index(s)] if s in sizes_seed else np.nan for s in sizes])
            scores = np.array(scores)
            mean_scores = np.nanmean(scores, axis=0)
            # sizes, scores = df_seeded["size_train"].values, df_seeded["score_valid"].values

            for i in range(0, len(model_names)):
                model_name = model_names[i]
                # print(model_name)
                for offset in range(4, len(sizes)):
                    experiment_id = '%d-%s-%s-%d' % (openmlid, learner, model_name, offset)

                    # this hash makes sure that we will get exactly the same results if we
                    # run this experiment again
                    hash = sha256(experiment_id.encode())
                    seed = np.frombuffer(hash.digest(), dtype='uint32')
                    np.random.seed(seed)

                    beta, model, fails_init, fails_fit = fit_model(np.array(sizes[:offset]),
                                                                   np.array(mean_scores[:offset]),
                                                                   np.array(sizes[offset:]), model_name, rep=rep, verbose=verbose)
                    sizes = np.array(sizes)
                    predictions = model(sizes)
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
            sizes = sorted(pd.unique(df_learner["size_train"]))
            scores = []
            for (inner, outer), df_seeded in df_learner.groupby(["inner_seed", "outer_seed"]):
                sizes_seed, scores_seed = df_seeded["size_train"].values, df_seeded["score_valid"].values
                scores.append([scores_seed[list(sizes_seed).index(s)] if s in sizes_seed else np.nan for s in sizes])
            scores = np.array(scores)
            mean_scores = np.nanmean(scores, axis=0)
            rows.append([openmlid, learner, sizes, mean_scores])
    return pd.DataFrame(rows, columns=["openmlid", "learner", "anchor_prediction", "score"])


def metrics_per_row(row, score, anchor_prediction):
    max_anchor_seen = row.max_anchor_seen
    prediction = row.prediction
    max_anchor = np.max(anchor_prediction)
    percentage_train = max_anchor_seen / max_anchor

    trn_ind = np.argwhere(max_anchor_seen == anchor_prediction)[0][0]  # recover offset
    trn_indices = range(0, (trn_ind + 1))
    tst_indices = range(trn_ind + 1, len(anchor_prediction))
    n_trn = len(trn_indices)

    y_trn_hat = prediction[trn_indices]
    y_trn = score[trn_indices]
    y_tst_hat = prediction[tst_indices]
    y_tst = score[tst_indices]

    bad_score_trn = np.isnan(y_trn)
    bad_score_tst = np.isnan(y_tst)

    y_trn = y_trn[bad_score_trn == False]
    y_trn_hat = y_trn_hat[bad_score_trn == False]

    y_tst = y_tst[bad_score_tst == False]
    y_tst_hat = y_tst_hat[bad_score_tst == False]

    MSE_trn = np.mean((y_trn - y_trn_hat) ** 2)
    MSE_tst = np.mean((y_tst - y_tst_hat) ** 2)
    MSE_tst_last = (y_tst[-1] - y_tst_hat[-1]) ** 2
    L1_trn = np.mean(np.abs(y_trn - y_trn_hat))
    L1_tst = np.mean(np.abs(y_tst - y_tst_hat))
    L1_tst_last = np.abs(y_tst[-1] - y_tst_hat[-1])

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


def select_part(part, df_all, datasets):
    num = 20
    max_num = part * num + num
    print('number of datasets: %d' % len(datasets))
    if max_num > len(datasets):
        max_num = len(datasets)
    indices = range(part * num, max_num)
    print('this job will do the fitting for datasets:')
    print(indices)
    datasets_todo = []
    for i in indices:
        datasets_todo.append(datasets[i])
    df_selected = df_all.loc[df_all['openmlid'].isin(datasets_todo)]
    return df_selected


def do_job(part, rep, verbose):
    np.seterr(all='ignore')

    print('starting part %d with %d repitions of the fitting' % (part, rep))

    df_all = pd.read_csv("database-accuracy.csv")
    np.random.seed(42)
    datasets = df_all['openmlid'].unique()
    np.random.shuffle(datasets)

    df_selected = select_part(part, df_all, datasets)

    print('computing extrapolations...')
    df_extrapolations = get_multiple_extrapolations_mean_curve_robust(df_selected, rep, verbose)
    df_extrapolations.to_pickle('extrapolations%d.gz' % part, protocol=3)

    print('computing anchors and scores...')
    df_anchors_and_scores = get_anchors_and_scores_mean_curve(df_selected)
    df_anchors_and_scores.to_pickle('anchors_scores%d.gz' % part, protocol=3)

    print('computing metrics....')
    df_metrics = df_compute_metrics_mean_curve(df_extrapolations, df_anchors_and_scores)
    df_metrics.to_pickle('metrics%d.gz' % part, protocol=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("part", type=int, help="a value in [0,12], which indicates which datasets should be fitted.")
    parser.add_argument("--rep", type=int, default=5, help="how many fits should be performed, the one with the best performance on the training anchors is taken, which extrapolates well.")
    parser.add_argument("-v","--verbose", action="store_true", help="if true, shows errors or giving up during fitting")
    args = parser.parse_args()
    part = args.part
    rep = args.rep
    verbose = args.verbose
    do_job(part, rep, verbose)


# usage:
# python fit_database.py <part>
# where <part> is in [0,12]
# we split the database in parts for fitting (each part fits 20 datasets), because otherwise fitting takes very long
# each part can be processed in parallel seperately
# each part takes approximately 30 min - 1 hour
# and requires 512 MB of memory for processing
# jupyter notebook 3b combines, preprocesses and analyses the results
if __name__ == "__main__":
    main()

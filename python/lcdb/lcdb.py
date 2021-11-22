import sklearn.model_selection
import time
from directencoder import DirectEncoder
from compressingencoder import CompressingEncoder

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

import openml

from func_timeout import func_timeout, FunctionTimedOut

import matplotlib.pyplot as plt

from tqdm import tqdm

import traceback


def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    df = ds.get_data()[0]
    num_rows = len(df)
    
    # impute missing values
    cateogry_columns=df.select_dtypes(include=['object', 'category']).columns.tolist()
    obsolete_cols = []
    for column in df:
        if df[column].isnull().any():
            if(column in cateogry_columns):
                if (all(df[column].values == None)):
                    obsolete_cols.append(column)
                else:
                    df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                df[column].fillna(df[column].median(), inplace=True)
    df = df.drop(columns=obsolete_cols)
    
    # prepare label column as numpy array
    y = np.array(list(df[ds.default_target_attribute].values))
    
    print(f"Read in data frame. Size is {len(df)} x {len(df.columns)}.")
    
    categorical_attributes = df.select_dtypes(exclude=['number']).columns
    expansion_size = 1
    for att in categorical_attributes:
        expansion_size *= len(pd.unique(df[att]))
        if expansion_size > 10**5:
            break
    
    if expansion_size < 10**5:
        X = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]]).values.astype(float)
    else:
        print("creating SPARSE data")
        dfSparse = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]], sparse=True)
        
        print("dummies created, now creating sparse matrix")
        X = lil_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        print("Done. shape is" + str(X.shape))
        
    if X.shape[0] != num_rows:
        raise Exception("Number of rows not ok!")
    return X, y


def get_class( kls ):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_outer_split(X, y, seed):
    test_samples_at_90_percent_training = int(X.shape[0] * 0.1)
    if test_samples_at_90_percent_training <= 5000:
        return sklearn.model_selection.train_test_split(X, y, train_size = 0.9, random_state=seed, stratify=y)
    else:
        return sklearn.model_selection.train_test_split(X, y, train_size = X.shape[0] - 5000, test_size = 5000, random_state=seed, stratify=y)

def get_inner_split(X, y, outer_seed, inner_seed):
    X_learn, X_test, y_learn, y_test = get_outer_split(X, y, outer_seed)
    
    validation_samples_at_90_percent_training = int(X_learn.shape[0] * 0.1)
    if validation_samples_at_90_percent_training <= 5000:
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X_learn, y_learn, train_size = 0.9, random_state=inner_seed, stratify=y_learn)
    else:
        print("Creating sample with instances:", X_learn.shape[0] - 5000)
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X_learn, y_learn, train_size = X_learn.shape[0] - 5000, test_size = 5000, random_state=inner_seed, stratify=y_learn)
                                                                                      
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed):
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_inner_split(X, y, outer_seed, inner_seed)
    if anchor > X_train.shape[0]:
        raise ValueError(f"Invalid anchor {anchor} when available training instances are only {X_train.shape[0]}.")
    return X_train[:anchor], X_valid, X_test, y_train[:anchor], y_valid, y_test


def get_truth_and_predictions(learner_inst, X, y, anchor, outer_seed = 0, inner_seed = 0, timeout = None, verbose=False):
    deadline = None if timeout is None else time.time() + timeout

    # create a random split based on the seed
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed)

    # fit the model
    start_time = time.time()
    if verbose:
        print(f"Training {learner_inst} on data of shape {X_train.shape} using outer seed {outer_seed} and inner seed {inner_seed}")
    if deadline is None:
        learner_inst.fit(X_train, y_train)
    else:
        func_timeout(deadline - time.time(), learner_inst.fit, (X_train, y_train))
    train_time = time.time() - start_time

    if verbose:
        print("Training ready. Obtaining predictions for " + str(X_test.shape[0]) + " instances.")

    # compute predictions on train data
    start_time = time.time()
    y_hat_train = learner_inst.predict(X_train)
    predict_time_train = time.time() - start_time
    start_time = time.time()
    try:
        y_prob_train = learner_inst.predict_proba(X_train)
    except:
        y_prob_train = None
    predict_proba_time_train = time.time() - start_time
    
    # compute predictions on validation data
    start_time = time.time()
    y_hat_valid = learner_inst.predict(X_valid)
    predict_time_valid = time.time() - start_time
    start_time = time.time()
    try:
        y_prob_valid = learner_inst.predict_proba(X_valid)
    except:
        y_prob_valid = None
    predict_proba_time_valid = time.time() - start_time

    # compute predictions on test data
    start_time = time.time()
    y_hat_test = learner_inst.predict(X_test)
    predict_time_test = time.time() - start_time
    start_time = time.time()
    try:
        y_prob_test = learner_inst.predict_proba(X_test)
    except:
        y_prob_test = None
    predict_proba_time_test = time.time() - start_time
    
    # return all information
    return y_train, y_valid, y_test, y_hat_train, y_prob_train, y_hat_valid, y_prob_valid, y_hat_test, y_prob_test, learner_inst.classes_, train_time, predict_time_train, predict_proba_time_train, predict_time_valid, predict_proba_time_valid, predict_time_test, predict_proba_time_test


def get_entry(learner_name, learner_params, X, y, anchor, outer_seed, inner_seed, encoder = DirectEncoder(), verbose=False):
    
    # get learner
    learner_class = get_class(learner_name)
    learner_inst = learner_class(**learner_params)
    
    # run learner
    y_train, y_valid, y_test, y_hat_train, y_prob_train, y_hat_valid, y_prob_valid, y_hat_test, y_prob_test, known_labels, train_time, predict_time_train, predict_proba_time_train, predict_time_valid, predict_proba_time_valid, predict_time_test, predict_proba_time_test = get_truth_and_predictions(learner_inst, X, y, anchor, outer_seed, inner_seed, verbose=verbose)
    
    # compute info entry
    info = {
        "size_train": anchor,
        "size_test": len(y_test),
        "outer_seed": outer_seed,
        "inner_seed": inner_seed,
        "traintime": np.round(train_time, 4),
        "labels": [str(l) for l in known_labels],
        "y_train": encoder.encode_label_vector(y_train),
        "y_valid": encoder.encode_label_vector(y_valid),
        "y_test": encoder.encode_label_vector(y_test),
        "y_hat_train": encoder.encode_label_vector(y_hat_train),
        "y_hat_valid": encoder.encode_label_vector(y_hat_valid),
        "y_hat_test": encoder.encode_label_vector(y_hat_test),
        "predictproba_train": encoder.encode_distribution(y_prob_train) if y_prob_train is not None else None,
        "predictproba_valid": encoder.encode_distribution(y_prob_valid) if y_prob_valid is not None else None,
        "predictproba_test": encoder.encode_distribution(y_prob_test) if y_prob_test is not None else None,
        "predicttime_train": np.round(predict_time_train, 4),
        "predicttimeproba_train": np.round(predict_proba_time_train, 4),
        "predicttime_valid": np.round(predict_time_valid, 4),
        "predicttimeproba_valid": np.round(predict_proba_time_valid, 4),
        "predicttime_test": np.round(predict_time_test, 4),
        "predicttimeproba_test": np.round(predict_proba_time_test, 4)
    }
    
    # show stats if desired            
    if verbose and info["predictproba_test"] is not None:
        y_test_after_compression = encoder.decode_distribution(info["predictproba_test"])
        log_loss_orig = np.round(sklearn.metrics.log_loss(y_test, y_prob_test, labels=known_labels), 3)
        log_loss_after_compression = np.round(sklearn.metrics.log_loss(y_test, y_test_after_compression, labels=known_labels), 3)
        diff = np.round(log_loss_orig - log_loss_after_compression, 3)
        gap = np.linalg.norm(y_test_after_compression - y_prob_test)
        print(f"Proba-Matrix Gap: {gap}")
        print(f"Change in log-loss due to compression: {diff} = {log_loss_orig} - {log_loss_after_compression}")
        
        if False:
            fig, ax = plt.subplots(1, y_prob_test.shape[1], figsize=(15, 5))
            for i, col in enumerate(y_prob_test.T):
                ax[i].scatter(col, y_test_after_compression[:,i], s=20)
                ax[i].plot([0,1], [0,1], linestyle="--", linewidth=1, color="black")
                ax[i].set_xscale("log")
                ax[i].set_yscale("log")
                ax[i].set_xlim([0.0001, 10])
                ax[i].set_ylim([0.0001, 10])
            plt.show()
    
    return info

def get_curve_by_metric(json_curve_descriptor, metric, encoder = DirectEncoder(), error="raise"):
    
    # gather data
    anchors_tmp = []
    values_train_tmp = []
    values_valid_tmp = []
    values_test_tmp = []
    for entry in json_curve_descriptor:
        try:
            m_train, m_valid, m_test = get_metric_of_entry(entry, metric, encoder)
            anchors_tmp.append(entry["size_train"])
            values_train_tmp.append(m_train)
            values_valid_tmp.append(m_valid)
            values_test_tmp.append(m_test)
        except:
            if error == "message":
                print("Ignoring entry with exception!")
            else:
                raise

    
    # convert all three lists in numpy arrays
    anchors_tmp = np.array(anchors_tmp)
    values_train_tmp = np.array(values_train_tmp)
    values_valid_tmp = np.array(values_valid_tmp)
    values_test_tmp = np.array(values_test_tmp)
    
    # prepare data
    anchors = sorted(list(np.unique(anchors_tmp)))
    values_train = [list(values_train_tmp[anchors_tmp == v]) for v in anchors]
    values_valid = [list(values_valid_tmp[anchors_tmp == v]) for v in anchors]
    values_test = [list(values_test_tmp[anchors_tmp == v]) for v in anchors]
    return anchors, values_train, values_valid, values_test
    
        
def get_metric_of_entry(entry, metric, encoder = DirectEncoder()):
    y_train = encoder.decode_label_vector(entry["y_train"])
    y_valid = encoder.decode_label_vector(entry["y_valid"])
    y_test = encoder.decode_label_vector(entry["y_test"])
    
    if type(metric) == str:
        if metric == "accuracy":
            y_hat_train = encoder.decode_label_vector(entry["y_hat_train"])
            y_hat_valid = encoder.decode_label_vector(entry["y_hat_valid"])
            y_hat_test = encoder.decode_label_vector(entry["y_hat_test"])
            return sklearn.metrics.accuracy_score(y_train, y_hat_train), sklearn.metrics.accuracy_score(y_valid, y_hat_valid), sklearn.metrics.accuracy_score(y_test, y_hat_test)
        
        elif metric == "logloss":
            if entry["predictproba_train"] is not None:
                y_prob_train = encoder.decode_distribution(entry["predictproba_train"])
                y_prob_valid = encoder.decode_distribution(entry["predictproba_valid"])
                y_prob_test = encoder.decode_distribution(entry["predictproba_test"])
            else:
                y_hat_train = encoder.decode_label_vector(entry["y_hat_train"])
                y_hat_valid = encoder.decode_label_vector(entry["y_hat_valid"])
                y_hat_test = encoder.decode_label_vector(entry["y_hat_test"])
                labels = entry["labels"]
                y_prob_train = np.zeros((len(y_hat_train), len(labels)))
                for i, label in enumerate(y_hat_train):
                    y_prob_train[i,labels.index(label)] = 1
                    y_prob_test = np.zeros((len(y_hat_test), len(labels)))
                for i, label in enumerate(y_hat_test):
                    y_prob_test[i,labels.index(label)] = 1
            m = sklearn.metrics.log_loss
            return m(y_train, y_prob_train, labels=entry["labels"]), m(y_valid, y_prob_valid, labels=entry["labels"]), m(y_test, y_prob_test, labels=entry["labels"])
    else:
        raise Exception("Currently only pre-defined metrics are supported.")

        
def compute_full_curve(learner_name, learner_params, dataset, outer_seeds=range(10), inner_seeds=range(10), show_progress=False, error="raise", verbose=False, encoder=DirectEncoder()):
       
    if type(dataset) == int: # openmlid
        
        # load data
        print("Reading dataset")
        X, y = get_dataset(dataset)
        labels = list(np.unique(y))
        is_binary = len(labels) == 2
        minority_class = labels[np.argmin([np.count_nonzero(y == label) for label in labels])]
        print(f"Labels are: {labels}")
        print(f"minority_class is {minority_class}")
        print("ready. Now building the learning curve")
    elif type(dataset) == tuple:
        if len(dataset) == 2:
            X, y = dataset[0], dataset[1]
        else:
            raise Exception("If data is provided as tuple, it must be a 2-tuple (X, y)")
    else:
        raise Exception("Data must be provided as an openmlid or as a pair (X,y) of instances and labels.")
    
    # configure anchors
    min_exp = 4
    max_train_size = int(max(int(X.shape[0] * 0.9) * 0.9, X.shape[0] - 10000))
    max_exp = np.log(max_train_size) / np.log(2)
    max_exp_int = (int(np.floor(max_exp)) - min_exp) * 2
    anchors = [int(np.round(2**(min_exp + 0.5 * exp))) for exp in range(0, max_exp_int + 1)]
    if anchors[-1] != max_train_size:
        anchors.append(max_train_size)
    print(anchors)
    
    # compute curve
    out = []
    if show_progress:
        pbar = tqdm(total = len(outer_seeds) * len(inner_seeds) * len(anchors))
    for outer_seed in outer_seeds:
        for inner_seed in inner_seeds:
            for anchor in anchors:
                try:
                    out.append(get_entry(learner_name, learner_params, X, y, anchor, outer_seed, inner_seed, encoder=encoder, verbose=verbose))
                except Exception as e:
                    if error == "raise":
                        raise
                    elif error == "message":
                        print(f"AN ERROR OCCURED! Here are the details.\n{type(e)}\n{e}")
                        traceback.print_exc()
                if show_progress:
                    pbar.update(1)
    if show_progress:
        pbar.close()
    return out


def plot_curve(anchors, points, ax, color):
    ax.plot(anchors, [np.median(v) for v in points], color=color)
    ax.plot(anchors, [np.mean(v) for v in points], linestyle="--", color=color)
    ax.fill_between(anchors, [np.percentile(v, 0) for v in points], [np.percentile(v, 100) for v in points], alpha=0.1, color=color)
    ax.fill_between(anchors, [np.percentile(v, 25) for v in points], [np.percentile(v, 75) for v in points], alpha=0.2, color=color)

def plot_train_and_test_curve(curve, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    anchors = curve[0]
    plot_curve(anchors, curve[1], ax, "C0") # train curve
    plot_curve(anchors, curve[2], ax, "C1") # validation curve
    plot_curve(anchors, curve[3], ax, "C2") # test curve
    
    ax.plot(anchors, [(np.mean(v_train) + np.mean(curve[2][a])) / 2 for a, v_train in enumerate(curve[1])], linestyle="--", color="black",linewidth=1)
    
    ax.axhline(np.mean(curve[2][-1]), linestyle="dotted", color="black",linewidth=1)
    ax.fill_between(anchors, np.mean(curve[2][-1]) - 0.0025, np.mean(curve[2][-1]) + 0.0025, color="black", alpha=0.1, hatch=r"//")
    
    if fig is not None:
        plt.show()
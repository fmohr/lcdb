import sklearn.model_selection
import time
from lcdb.directencoder import DirectEncoder

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

import openml

from func_timeout import func_timeout, FunctionTimedOut

import matplotlib.pyplot as plt

from tqdm import tqdm

import traceback

import shutil, tarfile

import json

from os import path
import importlib.resources as pkg_resources
from io import StringIO

import logging

logger = logging.getLogger('lcdb')

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    logger.info(f"Reading in full dataset from openml API.")
    df = ds.get_data()[0]
    num_rows = len(df)
    logger.info(f"Finished to read original data frame. Size is {len(df)} x {len(df.columns)}.")
    
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
    logger.info(f"Data frame is ready for extraction of attributes. Size is {len(df)} x {len(df.columns)}.")
    
    # prepare label column as numpy array
    y = np.array(list(df[ds.default_target_attribute].values))
    
    # identify categorical attributes
    categorical_attributes = df.select_dtypes(exclude=['number']).columns
    expansion_size = 1
    for att in categorical_attributes:
        expansion_size *= len(pd.unique(df[att]))
        if expansion_size > 10**5:
            break
    
    # create dummies
    logger.info(f"Creating dummies for {len(categorical_attributes)} categorical attributes.")
    if expansion_size < 10**5:
        X = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]]).values.astype(float)
    else:
        logger.info("creating SPARSE data")
        dfSparse = pd.get_dummies(df[[c for c in df.columns if c != ds.default_target_attribute]], sparse=True)
        
        logger.info("dummies created, now creating sparse matrix")
        X = lil_matrix(dfSparse.shape, dtype=np.float32)
        for i, col in enumerate(dfSparse.columns):
            ix = dfSparse[col] != 0
            X[np.where(ix), i] = 1
        
    if X.shape[0] != num_rows:
        raise Exception("Number of rows not ok!")
    logger.info(f"X, y are now completely processed. Shape of X is {X.shape}")
    return X, y

def get_names_of_meta_features():
    return ['AutoCorrelation', 'ClassEntropy', 'Dimensionality', 'EquivalentNumberOfAtts', 'MajorityClassSize', 'MaxAttributeEntropy', 'MaxKurtosisOfNumericAtts', 'MaxMeansOfNumericAtts', 'MaxMutualInformation', 'MaxNominalAttDistinctValues', 'MaxSkewnessOfNumericAtts', 'MaxStdDevOfNumericAtts', 'MeanAttributeEntropy', 'MeanKurtosisOfNumericAtts', 'MeanMeansOfNumericAtts', 'MeanMutualInformation', 'MeanNoiseToSignalRatio', 'MeanNominalAttDistinctValues', 'MeanSkewnessOfNumericAtts', 'MeanStdDevOfNumericAtts', 'MinAttributeEntropy', 'MinKurtosisOfNumericAtts', 'MinMeansOfNumericAtts', 'MinMutualInformation', 'MinNominalAttDistinctValues', 'MinSkewnessOfNumericAtts', 'MinStdDevOfNumericAtts', 'MinorityClassPercentage', 'MinorityClassSize', 'NaiveBayesAUC', 'NaiveBayesErrRate', 'NaiveBayesKappa', 'NumberOfBinaryFeatures', 'NumberOfClasses', 'NumberOfFeatures', 'NumberOfInstances', 'NumberOfInstancesWithMissingValues', 'NumberOfMissingValues', 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'PercentageOfBinaryFeatures', 'PercentageOfInstancesWithMissingValues', 'PercentageOfMissingValues', 'PercentageOfNumericFeatures', 'PercentageOfSymbolicFeatures', 'Quartile1AttributeEntropy', 'Quartile1KurtosisOfNumericAtts', 'Quartile1MeansOfNumericAtts', 'Quartile1MutualInformation', 'Quartile1SkewnessOfNumericAtts', 'Quartile1StdDevOfNumericAtts', 'Quartile2AttributeEntropy', 'Quartile2KurtosisOfNumericAtts', 'Quartile2MeansOfNumericAtts', 'Quartile2MutualInformation', 'Quartile2SkewnessOfNumericAtts', 'Quartile2StdDevOfNumericAtts', 'Quartile3AttributeEntropy', 'Quartile3KurtosisOfNumericAtts', 'Quartile3MeansOfNumericAtts', 'Quartile3MutualInformation', 'Quartile3SkewnessOfNumericAtts', 'Quartile3StdDevOfNumericAtts']

def compute_meta_data_of_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    qualities = ds.qualities
    X, y = get_dataset(openmlid)
    return [openmlid, ds.name] + [qualities[c] if c in qualities else np.nan for c in get_names_of_meta_features()] +  [X.shape[1]]

def compute_meta_data_of_dataset_collection(openmlids):
    
    # get current database
    try:
        df_cached = pd.read_csv(StringIO(pkg_resources.read_text('lcdb', 'datasets.csv')))
        logger.info(f"Found cache file with {len(df_cached)} entries for {list(df_cached['openmlid'])}")
        known_ids = pd.unique(df_cached["openmlid"])
    except:
        df_cached = None
        known_ids = []
    
    # compute new rows
    rows = []
    for i in tqdm(openmlids):
        if not i in known_ids:
            logger.info(f"Computing meta-features for dataset {i}")
            rows.append(compute_meta_data_of_dataset(i))
        else:
            logger.info(f"Re-using cached result for dataset {i}")
    df_new = pd.DataFrame(rows, columns=["openmlid", "Name"] + get_names_of_meta_features() + ["NumberOfFeaturesAfterBinarization"])
    for c in df_new.columns:
        if "NumberOf" == c[:8]:
            try:
                df_new = df_new.astype({c: int})
            except:
                logger.info(f"Could not cast column {c} to int!")
    return pd.concat([df_cached, df_new]).sort_values("openmlid")

def get_meta_features(dataset = None):
    df_cached = pd.read_csv(StringIO(pkg_resources.read_text('lcdb', 'datasets.csv')))
    if dataset is None:
        return df_cached
    
    openmlid = get_openmlid_from_descriptor(dataset)
    df_match = df_cached[df_cached["openmlid"] == openmlid]
    return df_match.iloc[0].to_dict()

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
        logger.info(f"Creating sample with instances: {X_learn.shape[0] - 5000}")
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
        logger.info(f"Training {learner_inst} on data of shape {X_train.shape} using outer seed {outer_seed} and inner seed {inner_seed}")
    if deadline is None:
        learner_inst.fit(X_train, y_train)
    else:
        func_timeout(deadline - time.time(), learner_inst.fit, (X_train, y_train))
    train_time = time.time() - start_time

    logger.info(f"Training ready. Obtaining predictions for {X_test.shape[0]} instances.")

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
        logger.info(f"Proba-Matrix Gap: {gap}")
        logger.info(f"Change in log-loss due to compression: {diff} = {log_loss_orig} - {log_loss_after_compression}")
        
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

def get_curve_for_metric_as_dataframe(json_curve_descriptor, metric, encoder = DirectEncoder(), error="raise", precision=4):
    
    cols = ['size_train', 'size_test', 'outer_seed', 'inner_seed', 'traintime']
    rows = []
    for entry in json_curve_descriptor:
        try:
            m_train, m_valid, m_test = get_metric_of_entry(entry, metric, encoder)
            row = [entry["size_train"], entry["size_test"], entry["outer_seed"], entry["inner_seed"], entry["traintime"], np.round(m_train, precision), np.round(m_valid, precision), np.round(m_test, precision)]
            rows.append(row)
        except:
            if error == "message":
                logger.info("Ignoring entry with exception!")
            else:
                raise
    return pd.DataFrame(rows, columns=cols + ["score_" + i for i in ["train", "valid", "test"]])


def get_curve_from_dataframe(curve_df):
    
    # sanity check to see that, if we know the learner, that there is only one of them
    if "learner" in curve_df.columns and len(pd.unique(curve_df["learner"])) > 1:
        raise Exception("pass only dataframes with entries for a single learner.")
    if "openmlid" in curve_df.columns and len(pd.unique(curve_df["openmlid"])) > 1:
        raise Exception("pass only dataframes with entries for a single openmlid.")
    
    # gather data
    anchors = sorted(list(pd.unique(curve_df["size_train"])))
    values_train = []
    values_valid = []
    values_test = []
    
    # extract curve
    for anchor in anchors:
        curve_df_anchor = curve_df[curve_df["size_train"] == anchor]
        values_train.append(list(curve_df_anchor["score_train"]))
        values_valid.append(list(curve_df_anchor["score_valid"]))
        values_test.append(list(curve_df_anchor["score_test"]))
        
    return anchors, values_train, values_valid, values_test

def get_curve_by_metric_from_json(json_curve_descriptor, metric, encoder = DirectEncoder(), error="raise"):
    
    return get_curve_from_dataframe(get_curve_for_metric_as_dataframe(json_curve_descriptor, metric, encoder, error))

        
        
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
                y_prob_valid = np.zeros((len(y_hat_valid), len(labels)))
                for i, label in enumerate(y_hat_valid):
                    y_prob_valid[i,labels.index(label)] = 1
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
        logger.info("Reading dataset")
        X, y = get_dataset(dataset)
        labels = list(np.unique(y))
        is_binary = len(labels) == 2
        minority_class = labels[np.argmin([np.count_nonzero(y == label) for label in labels])]
        logger.info(f"Labels are: {labels}")
        logger.info(f"minority_class is {minority_class}")
        logger.info("ready. Now building the learning curve")
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
    logger.info(anchors)
    
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
                        logger.info(f"AN ERROR OCCURED! Here are the details.\n{type(e)}\n{e}")
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
    else:
        fig = None
    anchors = curve[0]
    plot_curve(anchors, curve[1], ax, "C0") # train curve
    plot_curve(anchors, curve[2], ax, "C1") # validation curve
    plot_curve(anchors, curve[3], ax, "C2") # test curve
    
    ax.plot(anchors, [(np.mean(v_train) + np.mean(curve[2][a])) / 2 for a, v_train in enumerate(curve[1])], linestyle="--", color="black",linewidth=1)
    
    ax.axhline(np.mean(curve[2][-1]), linestyle="dotted", color="black",linewidth=1)
    ax.fill_between(anchors, np.mean(curve[2][-1]) - 0.0025, np.mean(curve[2][-1]) + 0.0025, color="black", alpha=0.1, hatch=r"//")
    
    if fig is not None:
        plt.show()
        
def get_train_times(dataset, learner):
    df_curve = get_curve_as_dataframe(dataset, learner)
    
    # gather data
    anchors = sorted(list(pd.unique(df_curve["size_train"])))
    train_times = []
    
    # extract curve
    for anchor in anchors:
        df_curve_anchor = df_curve[df_curve["size_train"] == anchor]
        train_times.append(list(df_curve_anchor["traintime"]))
        
    return anchors, train_times
        
def plot_train_times(dataset, learner, ax = None, color = None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    anchors, times = get_train_times(dataset, learner)
    
    if color is None:
        num_lines = len(ax.lines)
        color = "C" + str(num_lines)
    plot_curve(anchors, times, ax, color)
    
    if fig is not None:
        plt.show()

        
'''
    this recovers the database table from a tar ball from the disk.
    
    The tar ball is unpacked, read, and the unpacked content is removed again
'''
def get_raw_df(openmlid):
    
    # extract db file
    fname = f"database-original/lcdb-{openmlid}.tar.gz"
    path = "./tmp/conversions"
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(path=path)
    tar.close()
    
    # read in data frame
    df = pd.read_csv(f"{path}/lcdb-{openmlid}.csv", delimiter=";")
    
    # delete folder
    shutil.rmtree(path)
    
    # return dataframe
    return df

'''
    this takes a dataframe modeling the MySQL table and turns it into the final format (result column is resolved)
'''
def convert_raw_df_into_final_df(df_raw, metric):
    cols = ["openmlid", "learner", "size_train", "size_test", "outer_seed", "inner_seed", "traintime", "score_train", "score_valid", "score_test"]
    df_results = pd.DataFrame([], columns=cols)
    for i, row in df_raw.iterrows():
        curve_raw = json.loads(row["result"])
        curve_df = get_curve_for_metric_as_dataframe(curve_raw, metric)
        curve_df["learner"] = row["learner"]
        curve_df["openmlid"] = row["openmlid"]
        df_results = pd.concat([df_results, curve_df[cols]])
    return df_results

'''
    this takes a series of dataset descriptors and computes a full dataframe with all learning curves from the tar balls
    
    for every openmlid, there must be the file "database-original/lcdb-<id>.tar.gz" available
'''
def compile_dataframe_for_all_datasets_on_metric(openmlids, metric):
    df_results = None
    for openmlid in tqdm(openmlids):
        logger.info(openmlid)
        try:
            df_results_tmp = convert_raw_df_into_final_df(get_raw_df(openmlid), metric)
            df_results = df_results_tmp if df_results is None else pd.concat([df_results, df_results_tmp])
        except KeyboardInterrupt:
            raise
        except:
            logger.error("An error occurred.")
    return df_results


'''
    gets the full local database of learning curves as a dataframe
'''
def get_all_curves(metric = "accuracy"):
    supported_metrics = ["accuracy", "logloss"]
    if not metric in supported_metrics:
        raise ValueError(f"Unsupported metric {metric}. Supported metrics are {supported_metrics}")
    return pd.read_csv(StringIO(pkg_resources.read_text('lcdb', f'database-{metric}.csv')))

def get_openmlid_from_name(name):
    df_meta = get_meta_features()
    df_match = df_meta[df_meta["Name"] == name]
    if len(df_match) == 0:
        raise Exception(f"No dataset found with name {name}")
    elif len(df_match) > 1:
        raise Exception(f"Ambigous dataset name found {name}. There are {len(df_match)} datasets with this name.")
    else:
        return int(df_meta[df_meta["Name"] == name].iloc[0]["openmlid"])

def get_openmlid_from_descriptor(dataset):
    if type(dataset) == str:
        return get_openmlid_from_name(dataset)
    elif type(dataset) == int:
        return dataset
    else:
        raise ValuerError(f"Unsupported datatype {type(dataset)} for first positional argument.")

def get_curve_as_dataframe(dataset, learner, metric = "accuracy"):
    openmlid = get_openmlid_from_descriptor(dataset)
    
    df = get_all_curves(metric)
    df = df[(df["openmlid"] == openmlid)]
    if len(df) == 0:
        raise Exception(f"No curves found for openmlid {openmlid}")
    df = df[(df["learner"] == learner)]
    if len(df) == 0:
        raise Exception(f"No curves found for learner {learner} on openmlid {openmlid}")
    return df
        
'''
    gets the anchors and train, validation, and test performances of the learner on the given dataset.
    It is just a shortcut for retrieval from the dataframe.
'''
def get_curve(dataset, learner, metric = "accuracy"):
    

    return get_curve_from_dataframe(get_curve_as_dataframe(dataset, learner, metric))
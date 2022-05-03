import numpy as np
import pandas as pd
import itertools as it
from tqdm.notebook import tqdm


# define some constants for the evaluation
LEARNERS = ['SVC_linear',
 'SVC_poly',
 'SVC_rbf',
 'SVC_sigmoid',
 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis',
 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis',
 'sklearn.ensemble.ExtraTreesClassifier',
 'sklearn.ensemble.GradientBoostingClassifier',
 'sklearn.ensemble.RandomForestClassifier',
 'sklearn.linear_model.LogisticRegression',
 'sklearn.linear_model.PassiveAggressiveClassifier',
 'sklearn.linear_model.Perceptron',
 'sklearn.linear_model.RidgeClassifier',
 'sklearn.linear_model.SGDClassifier',
 'sklearn.naive_bayes.BernoulliNB',
 'sklearn.naive_bayes.MultinomialNB',
 'sklearn.neighbors.KNeighborsClassifier',
 'sklearn.neural_network.MLPClassifier',
 'sklearn.tree.DecisionTreeClassifier',
 'sklearn.tree.ExtraTreeClassifier']

LEARNER_NAMES = [
    "SVC (linear)",
    "SVC (poly)",
    "SVC (rbf)",
    "SVC (sigm.)",
    "LDA", "QDA",
    "Extra Trees", "Grad. Boost",
    "RF", "LR", "PA", "Perceptron", "Ridge", "SGD",
    "Bernouli NB", "Multin. NB", "kNN",
    "MLP", "DT", "Extra Tree"
    
]
if len(LEARNERS) != len(LEARNER_NAMES):
    raise Exception()



def get_mean_curves(df):
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
            rows.append([openmlid, learner, sizes, np.round(mean_scores, 4), np.round(1 - mean_scores, 4)])
    return pd.DataFrame(rows, columns=["openmlid", "learner", "sizes", "mean_accuracies", "mean_errorrates"])
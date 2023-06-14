from lcdb.data import load_task, random_split_from_array
from lcdb.workflow._base_workflow import BaseWorkflow
from sklearn.preprocessing import StandardScaler
from evalutils import *
from time import time

class WorkflowRunner:

    @staticmethod
    def run(
            openmlid: int,
            workflow_class: BaseWorkflow,
            hyperparameters: dict,
            outer_seed: int,
            inner_seed: int,
            anchor: int,
            monotonic: bool,
            logger=None
    ):

        if logger is None:
            logger = logging.getLogger("lcdb.exp")

        # CPU
        logger.info("CPU Settings:")
        for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS"]:
            logger.info(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")

        # load data
        binarize_sparse = openmlid in [1111, 41147, 41150, 42732, 42733]
        logger.info(f"Reading dataset. Will be binarized sparsely: {binarize_sparse}")
        X, y = get_dataset(openmlid)
        logger.info(f"ready. Dataset shape is {X.shape}, label column shape is {y.shape}. Now running the algorithm")
        if X.shape[0] <= 0:
            raise Exception("Dataset size invalid!")
        if X.shape[0] != len(y):
            raise Exception("X and y do not have the same size.")

        # get the data for this experiment
        labels = sorted(np.unique(y))
        X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(X, y, anchor, outer_seed, inner_seed, monotonic)
        preprocessing_steps = get_mandatory_preprocessing(X, y, binarize_sparse=binarize_sparse, drop_first=True)
        if preprocessing_steps:
            pl = Pipeline(preprocessing_steps).fit(X_train, y_train)
            X_train, X_valid, X_test = pl.transform(X_train), pl.transform(X_valid), pl.transform(X_test)


        # create the configured workflow
        # TODO: alternatively, one could be lazy and not pass the training data here.
        #       Then the workflow might have to do some setup routine at the beginning of `fit`
        #       Or as a middle-ground solution: We pass the dimensionalities of the task but not the data itself
        workflow = workflow_class(X_train, y_train, hyperparameters)

        # train the workflow
        ts_fit_start = time()
        workflow.fit((X_train, y_train), (X_valid, y_valid))
        ts_fit_end = time()
        fit_time = ts_fit_end - ts_fit_start

        logger.info(f"SVM fitted after {np.round(fit_time, 2)}s. Now obtaining predictions.")

        # compute confusion matrices
        start = time()
        y_hat_train = workflow.predict(X_train)
        predict_time_train = time() - start
        start = time()
        y_hat_valid = workflow.predict(X_valid)
        predict_time_valid = time() - start
        start = time()
        y_hat_test = workflow.predict(X_test)
        predict_time_test = time() - start
        cm_train = sklearn.metrics.confusion_matrix(y_train, y_hat_train, labels=labels)
        cm_valid = sklearn.metrics.confusion_matrix(y_valid, y_hat_valid, labels=labels)
        cm_test = sklearn.metrics.confusion_matrix(y_test, y_hat_test, labels=labels)

        # ask workflow to update its summary information (post-processing hook)
        workflow.update_summary()

        return labels, cm_train, cm_valid, cm_test, fit_time, predict_time_train, predict_time_valid, predict_time_test, workflow.summary
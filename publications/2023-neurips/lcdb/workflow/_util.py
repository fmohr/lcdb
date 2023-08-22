import base64
import gzip
import importlib
import itertools as it
import logging
import warnings
import os
from time import time
from typing import Dict, List
import psutil
import contextlib
import io


import ConfigSpace.read_and_write.json
import numpy as np
import openml
import pandas as pd
import sklearn.metrics
from ConfigSpace import Configuration, ConfigurationSpace
from py_experimenter.experimenter import PyExperimenter
from py_experimenter.experimenter import utils as pyexp_utils
from sklearn.pipeline import Pipeline
from scipy.stats.qmc import LatinHypercube
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    IntegerHyperparameter,
    NumericalHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import ForbiddenValueError, deactivate_inactive_hyperparameters


from ..data.split import (
    get_splits_for_anchor2,
)
from ._base_workflow import BaseWorkflow
from pynisher import limit, WallTimeoutException


def get_schedule_for_number_of_instances(num_instances, val_fold_size, test_fold_size):
    max_training_set_size = int(
        (1 - test_fold_size) * (1 - val_fold_size) * num_instances
    )
    anchors_for_dataset = []
    # for exp in list(range(8, 2 * (1 + int(np.log2(max_training_set_size))) - 1)):
    #    anchors_for_dataset.append(int(np.round(2 ** (exp / 2))))

    # sample size = 2^((7+k)/2) = 2^exponent
    max_k = int(2 * np.log2(max_training_set_size) - 7)
    for k in list(range(1, max_k + 1)):
        exponent = (7 + k) / 2
        sample_size = int(np.round(2**exponent))
        anchors_for_dataset.append(sample_size)
    if sample_size != max_training_set_size:
        anchors_for_dataset.append(max_training_set_size)
    return anchors_for_dataset


def get_experimenter(config_file, executor_name="") -> PyExperimenter:
    return PyExperimenter(
        experiment_configuration_file_path=config_file, name=executor_name
    )


def get_technical_experiment_grid(
    config_file="config/experiments.cfg",
    max_num_anchors_per_row=3,
):
    config = pyexp_utils.load_config(path=config_file)

    # get all combinations for the keyfields that are independent of the dataset properties and the learner
    keyfield_domains = pyexp_utils.get_keyfield_data(config=config)
    relevant_keyfield_names = [
        "valid_prop",
        "test_prop",
        "seed_outer",
        "seed_inner",
        "monotonic",
        "maxruntime",
        "measure_memory",
    ]
    keyfield_combinations = list(
        it.product(*[keyfield_domains[kf] for kf in relevant_keyfield_names])
    )

    # Note: it is not supported to have multiple valid / test props!
    val_fold_size = float(keyfield_domains.get("valid_prop", 0.1)[0])
    test_fold_size = float(keyfield_domains.get("test_prop", 0.1)[0])

    rows = []
    for openmlid in keyfield_domains["openmlid"]:
        print("trying to download openml dataset %d" % openmlid)
        num_instances = openml.datasets.get_dataset(
            dataset_id=openmlid,
            download_data=True,
            download_qualities=True,
        ).qualities["NumberOfInstances"]
        schedule = get_schedule_for_number_of_instances(
            num_instances, val_fold_size, test_fold_size
        )

        schedule_new = []
        schedule_tmp = []
        for cur_trainsize in schedule:
            schedule_tmp.append(cur_trainsize)
            if len(schedule_tmp) == max_num_anchors_per_row:
                schedule_new.append(schedule_tmp.copy())
                schedule_tmp = []
        if len(schedule_tmp) > 0:
            schedule_new.append(schedule_tmp.copy())

        for my_schedule in schedule_new:
            for combo in keyfield_combinations:
                combo = list(combo)
                rows.append(
                    [openmlid]
                    + combo[:4]  # valid_prop, test_prop, seed_outer, seed_inner
                    + [my_schedule]
                    + combo[-3:]
                )

    sampled_configurations = pd.DataFrame(
        rows,
        columns=["openmlid"]
        + relevant_keyfield_names[:4]
        + ["train_sizes"]
        + relevant_keyfield_names[-3:],
    )

    return config, sampled_configurations


# adapted from here https://github.com/automl/SMAC3/blob/e64e1918eeb88e93f9f201ece343624fb2943e9d/smac/initial_design/latin_hypercube_design.py
# and here https://github.com/automl/SMAC3/blob/e64e1918eeb88e93f9f201ece343624fb2943e9d/smac/initial_design/abstract_initial_design.py
# TODO: SMAC3 is BSD 3-Clause License need to respect this and handle this properly
class LHSGenerator:
    def __init__(
        self,
        configuration_space: ConfigurationSpace,
        n: int,
        seed: int = 0,
    ):
        self.configuration_space = configuration_space
        self.n = n
        self.seed = seed

        if len(self.configuration_space.get_hyperparameters()) > 21201:
            raise ValueError(
                "The default Latin Hypercube generator can only handle up to 21201 dimensions."
            )

    def generate(self) -> list[Configuration]:
        params = self.configuration_space.get_hyperparameters()

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        dim = len(params) - constants
        lhd_gen = LatinHypercube(d=dim, seed=self.seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lhd = lhd_gen.random(self.n)

        return self._transform_continuous_designs(
            design=lhd,
            origin="Latin Hypercube sequence",
            configuration_space=self.configuration_space,
        )

    def _transform_continuous_designs(
        self, design: np.ndarray, origin: str, configuration_space: ConfigurationSpace
    ) -> list[Configuration]:
        params = configuration_space.get_hyperparameters()
        for idx, param in enumerate(params):
            if isinstance(param, IntegerHyperparameter):
                design[:, idx] = param._inverse_transform(
                    param._transform(design[:, idx])
                )
            elif isinstance(param, NumericalHyperparameter):
                continue
            elif isinstance(param, Constant):
                design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
                design_[:, :idx] = design[:, :idx]
                design_[:, idx + 1 :] = design[:, idx:]
                design = design_
            elif isinstance(param, CategoricalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.choices), dtype=int)
            elif isinstance(param, OrdinalHyperparameter):
                v_design = design[:, idx]
                v_design[v_design == 1] = 1 - 10**-10
                design[:, idx] = np.array(v_design * len(param.sequence), dtype=int)
            else:
                raise ValueError(
                    "Hyperparameter not supported when transforming a continuous design."
                )

        configs = []
        for vector in design:
            try:
                conf = deactivate_inactive_hyperparameters(
                    configuration=None,
                    configuration_space=configuration_space,
                    vector=vector,
                )
            except ForbiddenValueError:
                continue

            conf.origin = origin
            configs.append(conf)

        return configs


def unserialize_config_space(json_filename) -> ConfigSpace.ConfigurationSpace:
    with open(json_filename, "r") as f:
        json_string = f.read()
    return ConfigSpace.read_and_write.json.read(json_string)


def get_default_config(config_space):
    defaulthps = {}
    for (
        hyperparameter_name,
        hyperparameter,
    ) in config_space.get_hyperparameters_dict().items():
        default = hyperparameter.default_value
        defaulthps[hyperparameter_name] = default
    default_config = ConfigSpace.configuration_space.Configuration(
        config_space, values=defaulthps, allow_inactive_with_values=True
    )
    # default_config = config_space.deactivate_inactive_hyperparameters(default_config, config_space) # only available in later version...
    return default_config


def get_all_experiments(
    config_file,
    num_configs: int,
    seed: int,
    max_num_anchors_per_row: int,
    LHS: bool,
) -> List[Dict]:
    """Create a sample of experimental configurations for a given workflow.

    Args:
        workflow_class (BaseWorkflow): The workflow for which to create the experiments.
        num_configs (int): The number of experimental configurations that are being sampled.
        seed (int): The random seed used to sample the configurations.
        max_num_anchors_per_row (int): The maximum number of sample-based anchors per experimental configuration.

    Returns:
        List[Dict]: A list of dictionnary objects representing the experimental configurations.
    """

    # get the experiment grid except the hyperparameters
    config, df_experiments = get_technical_experiment_grid(
        config_file=config_file,
        max_num_anchors_per_row=max_num_anchors_per_row,
    )

    # import the workflow class
    workflow_path = config.get("PY_EXPERIMENTER", "workflow")
    workflow_class = import_attr_from_module(workflow_path)

    config_space = workflow_class.get_config_space()
    default_config = get_default_config(config_space)

    config_space.seed(seed)

    if LHS:
        print("using LHS...")
        lhs_generator = LHSGenerator(config_space, n=num_configs, seed=seed)
        hp_samples = lhs_generator.generate()
    else:
        print("using random sampling...")
        hp_samples = config_space.sample_configuration(num_configs)
        if num_configs == 1:
            hp_samples = [hp_samples]
    hp_samples.insert(0, default_config)

    # create all rows for the experiments
    experiments = [
        {
            "workflow": workflow_path,
            "openmlid": openmlid,
            "valid_prop": v_p,
            "test_prop": t_p,
            "seed_outer": s_o,
            "seed_inner": s_i,
            "train_sizes": train_sizes,
            "maxruntime": maxruntime,
            "hyperparameters": dict(hp),
            "monotonic": mon,
            "measure_memory": measure_memory,
        }
        for (
            openmlid,
            v_p,
            t_p,
            s_o,
            s_i,
            train_sizes,
            mon,
            maxruntime,
            measure_memory,
        ), hp in it.product(df_experiments.values, hp_samples)
    ]
    return workflow_class, experiments


def run(
    openmlid: int,
    workflow_class: BaseWorkflow,
    hyperparameters: dict,
    outer_seed: int,
    inner_seed: int,
    anchors: list,
    monotonic: bool,
    maxruntime: int,
    valid_prop: float,
    test_prop: float,
    measure_memory: bool,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger("lcdb.exp")

    if type(anchors) != list:
        anchor = [anchors]
    logger.info(
        f"Starting experiment on {openmlid} for workflow '{workflow_class.__name__}'. Seeds: {outer_seed}/{inner_seed}. Anchors: {anchors}, HPs: {hyperparameters}. Monotonic: {monotonic}"
    )

    # CPU
    logger.info("CPU Settings:")
    for v in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        logger.info(f"\t{v}: {os.environ[v] if v in os.environ else 'n/a'}")

    # X, y = get_openml_dataset_and_check(openmlid)
    postprocess = False
    results = {}
    for anchor in anchors:
        results_tmp2 = {}
        time_outs = 0
        for inner_seed in range(0, 5):
            results_tmp = {}
            for outer_seed in range(0, 5):
                start = time()
                (
                    X_train,
                    X_valid,
                    X_test,
                    y_train,
                    y_valid,
                    y_test,
                    binarize_sparse,
                    drop_first,
                ) = get_splits_for_anchor2(
                    openmlid,
                    outer_seed,
                    inner_seed,
                    monotonic,
                    valid_prop=valid_prop,
                    test_prop=test_prop,
                )

                # check that anchor is not bigger than allowed
                if anchor > X_train.shape[0]:
                    raise ValueError(
                        f"Invalid anchor {anchor} when available training instances are only {X_train.shape[0]}."
                    )

                if monotonic:
                    # if monotonic, we shuffle the training set deterministically,
                    # the same way for each innerseed
                    random_seed_train_shuffle = inner_seed
                else:
                    # if not monotonic, the training set should be shuffled differently for each anchor
                    # so that the training sets of different anchors do not contain eachother
                    random_seed_train_shuffle = anchor

                rs = np.random.RandomState(random_seed_train_shuffle)
                indices = rs.choice(range(X_train.shape[0]), X_train.shape[0])
                X_train_shuffled = X_train[indices]
                y_train_shuffled = y_train[indices]

                X_train_anchor = X_train_shuffled[:anchor]
                y_train_anchor = y_train_shuffled[:anchor]

                # X_train, X_valid, X_test, y_train, y_valid, y_test = get_splits_for_anchor(
                #     X, y, anchor, outer_seed, inner_seed, monotonic, valid_prop=valid_prop, test_prop=test_prop
                # )
                # create the configured workflow
                # TODO: alternatively, one could be lazy and not pass the training data here.
                #       Then the workflow might have to do some setup routine at the beginning of `fit`
                #       Or as a middle-ground solution: We pass the dimensionalities of the task but not the data itself

                time_load_data = time() - start

                start = time()

                logger.info(
                    f"Working on anchor {anchor}, trainset size is {y_train.shape}."
                )
                workflow = workflow_class(X_train, y_train, hyperparameters)

                time_workflow = time() - start

                print("Starting time limited experiment...")
                # memory=(memory_limit, "MB")

                if os.name == "nt":
                    # do not timelimit on windows with pynisher, it doesnt work
                    my_limited_experiment = run_on_data
                else:
                    my_limited_experiment = limit(
                        run_on_data,
                        wall_time=(maxruntime, "s"),
                        terminate_child_processes=False,
                    )

                try:
                    results_tmp[outer_seed] = my_limited_experiment(
                        X_train_anchor,
                        X_valid,
                        X_test,
                        y_train_anchor,
                        y_valid,
                        y_test,
                        binarize_sparse,
                        drop_first,
                        valid_prop,
                        test_prop,
                        workflow,
                        logger,
                        time_load_data,
                        time_workflow,
                        measure_memory,
                    )
                except KeyboardInterrupt:
                    print("Interrupted by keyboard")
                    results_tmp[outer_seed] = "Interrupted by keyboard"
                except WallTimeoutException:
                    print("Timed out (took more than %d seconds)" % maxruntime)
                    results_tmp[outer_seed] = (
                        "Timed out (took more than %d seconds)" % maxruntime
                    )
                    time_outs = time_outs + 1
                # except MemoryLimitException:
                #     print('Used more memory than %d' % memory_limit)
                #     results[anchor] = 'Used more memory than %d' % memory_limit
                except Exception as err:
                    print("Exception: %s" % err)
                    results_tmp[outer_seed] = str(err)

            results_tmp2[inner_seed] = results_tmp
        results[anchor] = results_tmp2

        if time_outs == 25:
            print("Skipping the remaining anchors because we had 25 timeouts.")

            postprocess = True

            anchors_done = set(results.keys())
            anchors_all = set(anchors)
            anchors_todo = anchors_all - anchors_done
            for anchor_skip in anchors_todo:
                results[
                    anchor_skip
                ] = "Anchor skipped because previous anchor timed out"
            break

    return results, postprocess


# thanks to
# https://www.geeksforgeeks.org/monitoring-memory-usage-of-a-running-python-program/
# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print(
            "{}:consumed memory: {:,}".format(
                func.__name__, mem_before, mem_after, mem_after - mem_before
            )
        )
        return result

    return wrapper


@contextlib.contextmanager
def capture():
    import sys

    oldout, olderr = sys.stdout, sys.stderr
    try:
        out = [io.StringIO(), io.StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


def compress_numpy_array(array):
    array_bytes = array.tobytes()
    compressed_bytes = gzip.compress(array_bytes)
    return compressed_bytes


def compress_numpy_array_to_base64_string(array):
    compressed_bytes = compress_numpy_array(array)
    base64_string = base64.b64encode(compressed_bytes).decode("utf-8")
    return base64_string


def decompress_base64_string_to_numpy_array(base64_string, shape, dtype):
    compressed_bytes = base64.b64decode(base64_string)
    decompressed_bytes = gzip.decompress(compressed_bytes)
    array = np.frombuffer(decompressed_bytes, dtype=dtype)
    array = array.reshape(shape)
    return array


def batch_predict(workflow, X, batch_size=10000):
    # Predict in batches
    num_samples = X.shape[0]
    predictions = []

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch_predictions = workflow.predict(X[start:end])
        predictions.extend(batch_predictions)

    # Convert predictions list to numpy array
    predictions = np.array(predictions)
    return predictions


def run_on_data(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
    binarize_sparse,
    drop_first,
    valid_prop,
    test_prop,
    workflow,
    logger,
    time_load_data,
    time_workflow,
    measure_memory,
):
    mem_before = process_memory()
    if measure_memory:
        import tracemalloc

        tracemalloc.start()

    ts_innerloop_start = time()

    start = time()
    # get the data for this experiment
    labels = sorted(
        set(np.unique(y_train)) | set(np.unique(y_valid)) | set(np.unique(y_test))
    )
    preprocessing_steps = workflow.get_preprocessing_pipeline(
        X_train, y_train, binarize_sparse=binarize_sparse, drop_first=drop_first
    )
    if preprocessing_steps:
        # with capture() as out:
        pl = Pipeline(preprocessing_steps).fit(X_train, y_train)
        X_train, X_valid, X_test = (
            pl.transform(X_train),
            pl.transform(X_valid),
            pl.transform(X_test),
        )
    X_train_std = np.std(X_train, axis=0)
    X_train_std_max = np.max(X_train_std)
    X_train_std_min = np.min(X_train_std)
    X_train_min = np.min(X_train[:])
    X_train_max = np.max(X_train[:])
    logger.debug(
        f'f"Largest variance of feature and smallest: {X_train_std_max} {X_train_std_min}'
    )
    logger.debug(
        f'f"Largest value of feature and smallest: {X_train_min} {X_train_max}'
    )
    preprocessing_time = time() - start

    # train the workflow
    logger.debug(f"Start fitting the workflow...")
    ts_fit_start = time()
    workflow.fit((X_train, y_train), (X_valid, y_valid), (X_test, y_test))
    ts_fit_end = time()
    fit_time = ts_fit_end - ts_fit_start

    logger.debug(
        f"Workflow fitted after {np.round(fit_time, 2)}s. Now obtaining predictions."
    )

    # compute confusion matrices
    start = time()
    y_hat_train = batch_predict(workflow, X_train)
    predict_time_train = time() - start

    start = time()
    y_hat_valid = batch_predict(workflow, X_valid)
    # y_hat_valid_score = workflow.decision_function(X_valid)
    predict_time_valid = time() - start

    start = time()
    y_hat_test = batch_predict(workflow, X_test)
    # y_hat_test_score = workflow.decision_function(X_test)
    predict_time_test = time() - start

    start = time()
    # compute and compress confusion matrices
    cm_train = compress_numpy_array_to_base64_string(
        sklearn.metrics.confusion_matrix(y_train, y_hat_train, labels=labels).astype(
            "uint16"
        )
    )
    cm_valid = compress_numpy_array_to_base64_string(
        sklearn.metrics.confusion_matrix(y_valid, y_hat_valid, labels=labels).astype(
            "uint16"
        )
    )
    cm_test = compress_numpy_array_to_base64_string(
        sklearn.metrics.confusion_matrix(y_test, y_hat_test, labels=labels).astype(
            "uint16"
        )
    )
    # compute the zero-one loss and store in a string
    y_valid_error = np.packbits(y_valid != y_hat_valid)
    y_test_error = np.packbits(y_test != y_hat_test)
    y_valid_error = base64.b64encode(y_valid_error.astype(np.uint8)).decode("utf-8")
    y_test_error = base64.b64encode(y_test_error.astype(np.uint8)).decode("utf-8")
    compress_time = time() - start

    start = time()
    n_train = X_train.shape[0]
    n_valid = X_valid.shape[0]
    n_test = X_test.shape[0]
    train_prop = 1 - valid_prop - test_prop
    n_valid_sub = int(np.ceil(n_train / train_prop * valid_prop))
    n_test_sub = int(np.ceil(n_train / train_prop * test_prop))
    if n_valid_sub > n_valid:
        n_valid_sub = n_valid
    if n_test_sub > n_test:
        n_test_sub = n_test
    cm_valid_sub = compress_numpy_array_to_base64_string(
        sklearn.metrics.confusion_matrix(
            y_valid[:n_valid_sub], y_hat_valid[:n_valid_sub], labels=labels
        ).astype("uint16")
    )
    cm_test_sub = compress_numpy_array_to_base64_string(
        sklearn.metrics.confusion_matrix(
            y_test[:n_test_sub], y_hat_test[:n_test_sub], labels=labels
        ).astype("uint16")
    )
    subsample_time = time() - start

    # ask workflow to update its summary information (post-processing hook)
    logger.debug("Confusion matrices computed. Computing post-hoc data.")
    workflow.update_summary()

    mem_after = process_memory()
    time_innerloop = time() - ts_innerloop_start

    if measure_memory:
        _, memory_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        memory_peak = np.nan

    logger.info("Computation ready, returning results.")
    return {
        "labels": labels,
        "cm_train": cm_train,
        "cm_valid": cm_valid,
        "cm_test": cm_test,
        "cm_valid_sub": cm_valid_sub,
        "cm_test_sub": cm_test_sub,
        "fit_time": fit_time,
        "predict_time_train": predict_time_train,
        "predict_time_valid": predict_time_valid,
        "predict_time_test": predict_time_test,
        "subsample_time": subsample_time,
        "compress_time": compress_time,
        "time_load_data": time_load_data,
        "time_workflow": time_workflow,
        "time_innerloop": time_innerloop,
        "workflow_summary": workflow.summary,
        "mem_before": mem_before,
        "mem_after": mem_after,
        "y_valid_error": y_valid_error,
        "y_test_error": y_test_error,
        "memory_peak": memory_peak,
        # "fit_log": out,
    }


def import_attr_from_module(path: str):
    """Import an attribute from a module given its path.

    Example:

    >>> from lcdb.workflow import SVMWorkflow
    >>> import_attr_from_module("lcdb.workflow.SVMWorkflow") == SVMWorkflow
    """
    path = path.split(".")
    module_name, attr_name = (
        ".".join(path[:-1]),
        path[-1],
    )
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    return attr

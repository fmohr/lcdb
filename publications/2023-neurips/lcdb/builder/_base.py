import functools
import numpy as np
import copy
import logging

try:
    # Avoid some errors on some MPI implementations
    import mpi4py

    mpi4py.rc.initialize = False
    mpi4py.rc.threads = True
    mpi4py.rc.thread_level = "multiple"
    mpi4py.rc.recv_mprobe = False
    MPI4PY_IMPORTED = True
except ModuleNotFoundError:
    MPI4PY_IMPORTED = False

from deephyper.evaluator import Evaluator, RunningJob
from ..data import load_task
from .utils import import_attr_from_module, terminate_on_memory_exceeded


import traceback
import warnings

from tqdm import tqdm

from ..data.split import train_valid_test_split
from .timer import Timer
from .utils import (
    FunctionCallTimeoutError,
    get_schedule,
    terminate_on_timeout,
)
from .scorer import ClassificationScorer, RegressionScorer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def run(
    job: RunningJob,
    openml_id: int = 3,
    task_type: str = "classification",
    workflow_class: str = "lcdb.workflow.sklearn.LibLinearWorkflow",
    monotonic: bool = True,
    valid_seed: int = 42,
    test_seed: int = 42,
    workflow_seed: int = 42,
    valid_prop: float = 0.1,
    test_prop: float = 0.1,
    timeout_on_fit=-1,
    known_categories: bool = True,
    raise_errors: bool = False,
    anchor_schedule: str = "power",
    epoch_schedule: str = "full",
    logger=None,
):
    """This function trains the workflow on a dataset and returns performance metrics.

    Args:
        job (RunningJob): A running job passed by DeepHyper (represent an instance of the function).
        openml_id (int, optional): The identifier of the OpenML dataset. Defaults to 3.
        workflow_class (str, optional): The "path" of the workflow to train. Defaults to "lcdb.workflow.sklearn.LibLinearWorkflow".
        monotonic (bool, optional): A boolean indicating if the sample-wise learning curve should be monotonic (i.e., sample set at smaller anchors are always included in sample sets at larger anchors) or not. Defaults to True.
        valid_seed (int, optional): Random state seed of train/validation split. Defaults to 42.
        test_seed (int, optional): Random state seed of train+validation/test split. Defaults to 42.
        workflow_seed (int, optional): Random state seed of the workflow. Defaults to 42.
        valid_prop (float, optional): Ratio of validation/(train+validation). Defaults to 0.1.
        test_prop (float, optional): Ratio of test/data . Defaults to 0.1.
        timeout_on_fit (int, optional): Timeout in seconds for the fit method. Defaults to -1 for infinite time.
        known_categories (bool, optional): If all the possible categories are assumed to be known in advance. Defaults to True.
        raise_errors (bool, optional): If `True`, then errors are risen to the outside. Otherwise, just a log message is generated. Defaults to False.
        anchor_schedule (str, optional): A type of schedule for anchors (over samples of the dataset). Defaults to "power".
        epoch_schedule (str, optional): A type of schedule for epochs (over epochs of the dataset). Defaults to "power".

    Returns:
        dict: a dictionary with 2 keys (objective, metadata) where objective is the objective maximized by deephyper (if used) and metadata is a JSON serializable sub-dictionnary which are complementary information about the workflow.
    """
    logger.info(f"Running job {job.id} with parameters: {job.parameters}")

    timer = Timer(precision=4)
    run_timer_id = timer.start("run")

    # Load the raw dataset
    with timer.time("load_task"):
        logger.info("Loading the dataset...")
        (X, y), dataset_metadata = load_task(f"openml.{openml_id}")

    # Create and fit the workflow
    logger.info("Importing the workflow...")
    WorkflowClass = import_attr_from_module(workflow_class)
    workflow_kwargs = copy.deepcopy(job.parameters)
    workflow_kwargs["epoch_schedule"] = epoch_schedule
    workflow_kwargs["random_state"] = workflow_seed
    workflow_factory = lambda: WorkflowClass(timer=timer, **workflow_kwargs)

    # Initialize information to be returned
    infos = {
        "openmlid": openml_id,
        "workflow_seed": workflow_seed,
        "workflow": workflow_class,
    }

    # create controller
    if task_type not in ["classification", "regression"]:
        raise ValueError(
            f"Task type must be 'classification' or 'regression' but is {task_type}."
        )
    is_classification = task_type == "classification"
    stratify = is_classification

    controller = LearningCurveBuilder(
        timer=timer,
        workflow_factory=workflow_factory,
        is_classification=is_classification,
        X=X,
        y=y,
        dataset_metadata=dataset_metadata,
        test_seed=test_seed,
        valid_seed=valid_seed,
        monotonic=monotonic,
        test_prop=test_prop,
        valid_prop=valid_prop,
        timeout_on_fit=timeout_on_fit,
        known_categories=known_categories,
        stratify=stratify,
        raise_errors=raise_errors,
        anchor_schedule=anchor_schedule,
    )

    # build the curves
    controller.build_curves()

    assert (
        timer.active_node.id == run_timer_id
    ), f"Timer is not at the right place: {timer.active_node}"
    timer.stop()

    # update infos based on report
    infos.update(controller.report)

    infos["json"] = timer.as_json()

    results = {"objective": controller.objective, "metadata": infos}

    return results


class LearningCurveBuilder:

    def __init__(
        self,
        timer: Timer,
        workflow_factory,
        is_classification: bool,
        X,
        y,
        dataset_metadata,
        test_seed,
        valid_seed,
        valid_prop: float = 0.1,
        test_prop: float = 0.1,
        stratify=True,
        monotonic=False,
        timeout_on_fit=-1,
        known_categories: bool = True,
        raise_errors: bool = False,
        anchor_schedule: str = "power",
        logger=None,
    ):

        self.logger = logger if logger is not None else logging.getLogger("LCDB")

        self.timer = timer
        self.workflow_factory = workflow_factory
        self.workflow = None
        self.is_classification = is_classification

        self.num_instances = X.shape[0]
        self.dataset_metadata = dataset_metadata

        self.X = X
        self.y = y
        self.labels = list(np.unique(y))
        self.is_binary = len(self.labels) == 2
        (
            self.X_train,
            self.X_valid,
            self.X_test,
            self.y_train,
            self.y_valid,
            self.y_test,
        ) = train_valid_test_split(
            X, y, test_seed, valid_seed, test_prop, valid_prop, stratify=stratify
        )
        self.valid_seed = valid_seed
        self.test_seed = test_seed
        self.monotonic = monotonic
        self.timeout_on_fit = timeout_on_fit
        self.raise_errors = raise_errors
        self.anchors = get_schedule(
            name=anchor_schedule, n=len(self.X_train), base=2, power=0.5, delay=7
        )

        # state variables
        self.cur_anchor = None
        self.X_train_at_anchor = None
        self.y_train_at_anchor = None
        self.labels_as_used_by_workflow = (
            None  # list of labels, this order is defined by the workflow
        )
        self.curves = None
        self.additional_data_per_anchor = None

        # Transform categorical features
        columns_categories = np.asarray(dataset_metadata["categories"], dtype=bool)
        values_categories = None

        dataset_metadata["categories"] = {"columns": columns_categories}
        if not (np.any(columns_categories)):
            one_hot_encoder = FunctionTransformer(func=lambda x: x, validate=False)
        else:
            dataset_metadata["categories"]["values"] = None
            one_hot_encoder = OneHotEncoder(
                drop="first", sparse_output=False
            )  # TODO: drop "first" could be an hyperparameter
            one_hot_encoder.fit(X[:, columns_categories])
            if known_categories:
                values_categories = one_hot_encoder.categories_
                values_categories = [v.tolist() for v in values_categories]
                dataset_metadata["categories"]["values"] = values_categories

        # create report
        self.report = {
            "valid_prop": valid_prop,
            "test_prop": valid_prop,
            "monotonic": monotonic,
            "valid_seed": valid_seed,
            "test_seed": test_seed,
            "traceback": None,
        }

        self.objective = None

    def set_anchor(self, anchor):
        train_idx = np.arange(self.X_train.shape[0])
        # If not monotonic, the training set should be shuffled differently for each anchor
        # so that the training sets of different anchors do not contain eachother
        i = self.anchors.index(anchor)
        if not self.monotonic:
            random_seed_train_shuffle = np.random.RandomState(self.valid_seed).randint(
                0, 2**32 - 1, size=len(self.anchors)
            )[i]
            rs = np.random.RandomState(random_seed_train_shuffle)
            rs.shuffle(train_idx)

            X_train = self.X_train[train_idx]
            y_train = self.y_train[train_idx]
        else:
            X_train, y_train = self.X_train, self.y_train

        self.cur_anchor = anchor
        self.X_train_at_anchor = X_train[:anchor]
        self.y_train_at_anchor = y_train[:anchor]

    def build_curves(self):
        # Build sample-wise learning curve

        with self.timer.time("build_curves"):
            for anchor in tqdm(self.anchors, disable=True):
                self.set_anchor(anchor)

                with self.timer.time("anchor", {"value": anchor}) as anchor_timer:
                    self.logger.info(
                        f"Fitting workflow {self.workflow.__class__.__name__} on sample anchor {anchor} which is {anchor / self.X_train.shape[0] * 100:.2f}% of the dataset."
                    )

                    error_code = self.fit_workflow_on_current_anchor()

                    if error_code != 0:
                        # Cancel timers that were started in fit_workflow_on_current_anchor
                        self.timer.cancel(anchor_timer.id, only_children=True)

                    assert (
                        self.timer.active_node.id == anchor_timer.id
                    ), f"The active timer is not correct, it is {self.timer.active_node} when it should be {anchor_timer} "

                    # If an error was detected then skip scoring...
                    if error_code != 0:
                        break

                    # Predict and Score
                    self.logger.info("Predicting and scoring...")
                    try:
                        self.compute_metrics_for_workflow()
                    except Exception as exception:
                        # Cancel timers that were started in 'try' block
                        self.timer.cancel(anchor_timer.id, only_children=True)

                        # Collect traceback
                        self.report["traceback"] = traceback.format_exc()

                        self.logger.error(
                            f"Error while fitting the workflow: \n{self.report['traceback']}"
                        )

                        self.report["traceback"] = r'"{}"'.format(
                            self.report["traceback"]
                        )

                        # The evaluation is considered a total failure only if
                        # None of the anchors returned scored.
                        if self.objective is None:
                            self.objective = "F"

                            if isinstance(exception, ValueError):
                                self.objective += "_value_error"

                        error_code = 1
                        break

    def fit_workflow_on_current_anchor(self) -> int:
        """Fit the workflow on the current anchor.

        Returns 0 if the workflow was fitted successfully, 1 otherwise.
        """

        # Represent success (0) or failure (1) while fitting the workflow
        error_code = 0

        with self.timer.time("create_workflow"):
            self.workflow = self.workflow_factory()

        if self.timeout_on_fit > 0:
            self.workflow.fit = functools.partial(
                terminate_on_timeout, self.timeout_on_fit, self.workflow.fit
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.workflow.fit(
                    self.X_train_at_anchor,
                    self.y_train_at_anchor,
                    X_valid=self.X_valid,
                    y_valid=self.y_valid,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    metadata=self.dataset_metadata,
                )

        except Exception as exception:
            self.report["traceback"] = traceback.format_exc()

            self.logger.error(
                f"Error while fitting the workflow: \n{self.report['traceback']}"
            )

            self.report["traceback"] = r'"{}"'.format(self.report["traceback"])

            # The evaluation is considered a total failure only if
            # None of the anchors returned scored.
            if self.objective is None:
                self.objective = "F"

                if isinstance(exception, FunctionCallTimeoutError):
                    self.objective += "_function_call_timeout_error"
                elif isinstance(exception, MemoryError):
                    self.objective += "_memory_error"

            error_code = 1

        return error_code

    def compute_metrics_for_workflow(self):
        predictions, labels = self.get_predictions()
        self.labels_as_used_by_workflow = labels
        return self.score_predictions(**predictions)

    def get_predictions(self):
        keys = {}
        labels = (
            self.workflow.infos["classes_overall"] if self.is_classification else None
        )

        with self.timer.time("get_predictions"):
            for X_split, label_split in [
                (self.X_train_at_anchor, "train"),
                (self.X_valid, "val"),
                (self.X_test, "test"),
            ]:
                with warnings.catch_warnings(), self.timer.time(label_split):
                    warnings.simplefilter("ignore")

                    keys[f"y_pred_proba_{label_split}"] = (
                        self.workflow.predict_proba(X_split)
                        if self.is_classification
                        else None
                    )
                    keys[f"y_pred_{label_split}"] = (
                        self.workflow.get_predictions_from_probas(
                            keys[f"y_pred_proba_{label_split}"]
                        )
                    )

        return keys, labels

    def score_predictions(
        self,
        y_pred_train,
        y_pred_proba_train,
        y_pred_val,
        y_pred_proba_val,
        y_pred_test,
        y_pred_proba_test,
    ):
        if self.is_classification:
            scorer = ClassificationScorer(
                classes_learner=self.workflow.infos["classes_train_orig"],
                classes_overall=self.workflow.infos["classes_overall_orig"],
                timer=self.timer,
            )
        else:
            scorer = RegressionScorer(timer=self.timer)

        with self.timer.time("metrics"):
            for y_true, y_pred, y_pred_proba, label_split in [
                (self.y_train_at_anchor, y_pred_train, y_pred_proba_train, "train"),
                (self.y_valid, y_pred_val, y_pred_proba_val, "val"),
                (self.y_test, y_pred_test, y_pred_proba_test, "test"),
            ]:
                with self.timer.time(label_split) as split_timer:
                    if self.is_classification:
                        scores = scorer.score(y_true, y_pred, y_pred_proba)
                        if label_split == "val":
                            self.objective = -scores["log_loss"]
                    else:
                        scores = scorer.score(y_true, y_pred)
                        if label_split == "val":
                            self.objective = -scores["mean_squared_error"]

        return 0  # no error occurred

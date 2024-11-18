import numpy as np
import pandas as pd
from lcdb.analysis.json import (
    QueryAnchorValues,
    QueryEpochValues,
    QueryMetricValuesFromAnchors,
    QueryMetricValuesFromEpochs,
)
from lcdb.analysis.score import balanced_accuracy_from_confusion_matrix


class LearningCurve:

    def __init__(
            self,
            workflow,
            hp_config,
            openmlid,
            values,
            metrics,
            fold_names,
            test_seeds,
            val_seeds,
            workflow_seeds,
            anchors_size,
            anchors_iteration=None
    ):
        self.workflow = workflow
        self.hp_config = hp_config
        self.openmlid = openmlid

        self.values = values
        self.metrics = metrics
        self.fold_names = fold_names
        self.test_seeds = test_seeds
        self.val_seeds = val_seeds
        self.workflow_seeds = workflow_seeds
        self.anchors_size = anchors_size
        self.anchors_iteration = anchors_iteration

    @property
    def is_iteration_wise_curve(self):
        return self.anchors_iteration is not None and len(self.anchors_iteration) > 0

    def pad(self, dim, dim_labels, inplace=False):

        dim_names = ["metrics", "fold_names", "test_seeds", "val_seeds", "workflow_seeds", "anchors_size"]
        dim_index = dim_names.index(dim)

        if self.is_iteration_wise_curve:
            dim_names.append("anchors_iteration")

        cur_domain = getattr(self, dim)
        indices_to_insert = [i for i, label in enumerate(dim_labels) if label in cur_domain]

        # create new values object with nans
        old_shape = self.values.shape
        new_shape = list(old_shape)
        new_shape[dim_names.index(dim)] = len(dim_labels)
        new_shape = tuple(new_shape)
        new_values = np.zeros(new_shape)
        new_values[:] = np.nan

        # fill up new object with existing values
        selectors = tuple([
            slice(None, None, None) if (label != dim) else indices_to_insert for i, label in enumerate(dim_names)
        ])
        new_values[selectors] = self.values

        if inplace:
            self.values = new_values
            cur_domain.clear()
            cur_domain.extend(dim_labels)
            assert len(cur_domain) == self.values.shape[dim_index]

        else:
            assert len(dim_labels) == new_values.shape[dim_index]
            new_lc = {"workflow": self.workflow, "hp_config": self.hp_config, "openmlid": self.openmlid,
                      "values": new_values, "metrics": self.metrics, "fold_names": self.fold_names,
                      "test_seeds": self.test_seeds, "val_seeds": self.val_seeds, "workflow_seeds": self.workflow_seeds,
                      "anchors_size": self.anchors_size, dim: dim_labels}
            return LearningCurve(**new_lc)

    def clone(self):
        return LearningCurve(**{
            "workflow": self.workflow,
            "hp_config": self.hp_config,
            "openmlid": self.openmlid,
            "values": self.values,
            "metrics": self.metrics,
            "fold_names": self.fold_names,
            "test_seeds": self.test_seeds,
            "val_seeds": self.val_seeds,
            "workflow_seeds": self.workflow_seeds,
            "anchors_size": self.anchors_size
        })

    def pad_anchors_size(self, anchors_size, inplace=False):

        num_anchors_size = len(anchors_size)

        axis = -2 if self.is_iteration_wise_curve else -1

        if self.values.shape[axis] >= num_anchors_size:
            return self if inplace else None
        if self.is_iteration_wise_curve:
            nans_to_add = np.full(tuple(list(self.values.shape[:-2]) + [num_anchors_size - self.values.shape[-2], self.values.shape[-1]]), np.nan)
        else:
            nans_to_add = np.full(tuple(list(self.values.shape[:-1]) + [num_anchors_size - self.values.shape[-1]]), np.nan)

        if inplace:
            self.values = np.concatenate((self.values, nans_to_add), axis=axis)
            self.anchors_size = anchors_size
            assert len(self.anchors_size) == self.values.shape[axis]

        else:
            values = np.concatenate((self.values, nans_to_add), axis=axis)
            assert len(anchors_size) == values.shape[-1]
            return LearningCurve(
                workflow=self.workflow,
                hp_config=self.hp_config,
                openmlid=self.openmlid,
                values=values,
                metrics=self.metrics,
                fold_names=self.fold_names,
                test_seeds=self.test_seeds,
                val_seeds=self.val_seeds,
                workflow_seeds=self.workflow_seeds,
                anchors_size=anchors_size
            )


class LearningCurveGroup:

    def __init__(self, curves=None):

        self.workflow = None
        self.group_attribute = None
        self.moving_attribute = None
        self.openmlids = []
        self.hp_configs = []
        self.metrics = []
        self.fold_names = []
        self.test_seeds = []
        self.val_seeds = []
        self.workflow_seeds = []
        self.anchors_size = []
        self.anchors_iteration = []

        self.attribute_names_per_dimension = [
            "openmlids", "hp_configs", "metrics", "fold_names", "test_seeds", "val_seeds", "workflow_seeds",
            "anchors_size", "anchors_iteration"
        ]

        self.curves = []
        if curves is not None:
            for curve in curves:
                self.add_curve(curve)

    def add_curve(self, curve):

        # check workflow compatibility
        if self.workflow is not None and self.workflow != curve.workflow:
            raise ValueError("Cannot join learning curves from different workflows into a group.")
        elif self.workflow is None:
            self.workflow = curve.workflow

        # check axis compatibility
        if len(self.curves) > 0:

            # determine attribute for which all curves are identical (and implicitly in which they move)
            if self.group_attribute is None:
                different_dataset = curve.openmlid != self.openmlids[0]
                different_config = curve.hp_config != self.hp_configs[0]
                if different_dataset and different_config:
                    raise ValueError("Cannot group together learning curves that differ in both dataset and hp config!")
                if different_dataset:
                    self.group_attribute = "hp_config"
                    self.moving_attribute = "openmlid"
                else:
                    self.group_attribute = "openmlid"
                    self.moving_attribute = "hp_config"

        # add the value to the group axis and make sure that it is not a double entry
        if curve.hp_config not in self.hp_configs:
            self.hp_configs.append(curve.hp_config)
        elif self.moving_attribute == "hp_config":
            raise ValueError(
                "Cannot add a second curve for the same hyperparameter configuration to a group. Merge the curves before.")
        if curve.openmlid not in self.openmlids:
            self.openmlids.append(curve.openmlid)
        elif self.moving_attribute == "openmlid":
            raise ValueError("Cannot add a second curve for the same dataset to a group. Merge the curves before.")

        # update numpy array in group or of new curve if necessary
        for attribute_name in self.attribute_names_per_dimension[2:]:

            domain_group = getattr(self, attribute_name)
            domain_new_curve = getattr(curve, attribute_name)

            values_to_add_for_group = [v for v in domain_new_curve if
                                       v not in domain_group] if domain_new_curve is not None else []
            values_to_add_for_new_curve = [v for v in domain_group if v not in domain_new_curve]

            domain_group += values_to_add_for_group

            # if there are new values, define the new domain and pad the existing curves
            if len(values_to_add_for_group) > 0:
                for existing_curve in self.curves:
                    # existing_curve.pad(dim=attribute_name, dim_labels=domain_group)
                    raise NotImplementedError

            if len(values_to_add_for_new_curve) > 0:
                curve = curve.pad(dim=attribute_name, dim_labels=domain_group, inplace=False)

            if attribute_name in ["anchors_size", "anchors_iteration"] and sorted(domain_group) != domain_group:
                print(f"WARNING: {attribute_name} has unnaturally ordered values {domain_group}")

        self.curves.append(curve.clone())

    def __getitem__(self, key):
        if type(key) == int:
            return self.curves[key].values

        if type(key) != tuple:
            _lcg = LearningCurveGroup()
            _lcg.curves = self.curves[key]
            _lcg.workflow = self.workflow
            _lcg.group_attribute = self.group_attribute
            _lcg.moving_attribute = self.moving_attribute
            _lcg.openmlids = self.openmlids
            _lcg.hp_configs = self.hp_configs
            _lcg.metrics = self.metrics
            _lcg.fold_names = self.fold_names
            _lcg.test_seeds = self.test_seeds
            _lcg.val_seeds = self.val_seeds
            _lcg.workflow_seeds = self.workflow_seeds
            _lcg.anchors_size = self.anchors_size
            _lcg.anchors_iteration = self.anchors_iteration

            _lcg.attribute_names_per_dimension = self.attribute_names_per_dimension
            return _lcg

        curve_index = key[0]
        rest_index = key[1:]

        selected_curves = self.curves[curve_index]
        if isinstance(selected_curves, LearningCurve):
            selected_curves = [selected_curves]
        if len(selected_curves) == 0:
            return None
        if len(selected_curves) == 1:
            return selected_curves[0].values[rest_index]
        else:
            return np.array([sc.values[rest_index] for sc in selected_curves])

    def __len__(self):
        return len(self.curves)


class LearningCurveExtractor:

    def __init__(
        self,
        metrics=["error_rate"],
        folds=["train", "val", "test"],
        rounding_decimals=4,
        return_none_on_error=True,
    ):
        accepted_metrics = ["error_rate", "balanced_error_rate"]

        self.funs = {}
        self.srcs = {}
        self.folds = folds

        for metric in metrics:
            if not isinstance(metric, str):
                raise ValueError(f"Each metric in metrics must be str, but at least one is {type(metric)}: {metric}")

            if metric == "error_rate":
                self.funs[metric] = lambda cm: 1 - np.diag(cm).sum() / np.sum(cm)
                self.srcs[metric] = "confusion_matrix"
            elif metric == "balanced_error_rate":
                self.funs[metric] = (
                    lambda cm: 1 - balanced_accuracy_from_confusion_matrix(cm)
                )
                self.srcs[metric] = "confusion_matrix"
            else:
                raise ValueError(
                    f"metric is {metric} but must be in {accepted_metrics}."
                )

        self.metrics = metrics
        self.rounding_decimals = rounding_decimals
        self.return_none_on_error = return_none_on_error

    def __call__(self, row):
        """
        Computes the sample-wise learning curve for a specific metric for a set of configurations, possibly across workflows and datasets.
        """

        lc_dict = row["m:json"]

        # determine anchors and whether this is an iteration curve
        try:
            anchors_size = QueryAnchorValues()(lc_dict)
            anchors_iterations_per_sample_size = QueryEpochValues()(lc_dict)
            anchors_iteration = set()
            for iteration_anchors in anchors_iterations_per_sample_size:
                anchors_iteration |= set(iteration_anchors)
            anchors_iteration = sorted(anchors_iteration)
            is_iteration_curve = len(anchors_iteration) > 0

            # create array for values of curve
            shape = (
                (
                    len(self.metrics),
                    len(self.folds),
                    1,
                    1,
                    1,
                    len(anchors_size),
                    len(anchors_iteration),
                )
                if is_iteration_curve
                else (len(self.metrics), len(self.folds), 1, 1, 1, len(anchors_size))
            )
            values = np.zeros(shape)
            values[:] = np.nan

            for i1, metric in enumerate(self.metrics):
                fun = self.funs[metric]
                for i2, fold in enumerate(self.folds):

                    if is_iteration_curve:
                        sources_for_iteration_curve_values = (
                            QueryMetricValuesFromEpochs(
                                self.srcs[metric], split_name=fold
                            )(lc_dict)
                        )
                        for i3, (anchor_size, sources_for_anchor_size) in enumerate(
                            zip(anchors_size, sources_for_iteration_curve_values)
                        ):
                            for anchor_iteration, source_for_lc_point in zip(
                                anchors_iterations_per_sample_size[i3],
                                sources_for_anchor_size,
                            ):
                                i4 = anchors_iteration.index(anchor_iteration)
                                values[i1, i2, 0, 0, 0, i3, i4] = fun(source_for_lc_point)
                    else:
                        sample_wise_curve = [
                            fun(e)
                            for e in QueryMetricValuesFromAnchors(
                                self.srcs[metric], split_name=fold
                            )(lc_dict)
                        ]
                        num_missing_entries = values.shape[-1] - len(sample_wise_curve)
                        if num_missing_entries > 0:
                            sample_wise_curve.extend(num_missing_entries * [np.nan])
                        values[i1, i2, 0, 0, 0] = sample_wise_curve

            lc_params = {
                "workflow": row["m:workflow"],
                "hp_config": {k: v for k, v in row.items() if k.startswith("p:")},
                "openmlid": row["m:openmlid"],
                "values": values,
                "metrics": self.metrics,
                "fold_names": self.folds,
                "test_seeds": [row["m:test_seed"]],
                "val_seeds": [row["m:valid_seed"]],
                "workflow_seeds": [row["m:workflow_seed"]],
                "anchors_size": anchors_size,
            }
            if is_iteration_curve:
                lc_params["anchors_iteration"] = anchors_iteration
            return LearningCurve(**lc_params)

        except KeyboardInterrupt:
            raise

        except:
            raise


class IterationWiseCurveExtractor:

    def __init__(
        self,
        metrics=["error_rate"],
        folds=["train", "val", "test"],
        rounding_decimals=4,
    ):
        accepted_metrics = ["error_rate", "balanced_error_rate"]

        self.funs = {}
        self.srcs = {}
        self.folds = folds

        for metric in metrics:
            if metric == "error_rate":
                self.funs[metric] = lambda cm: 1 - np.diag(cm).sum() / np.sum(cm)
                self.srcs[metric] = "confusion_matrix"
            elif metric == "balanced_error_rate":
                self.funs[metric] = (
                    lambda cm: 1 - balanced_accuracy_from_confusion_matrix(cm)
                )
                self.srcs[metric] = "confusion_matrix"
            else:
                raise ValueError(
                    f"metric is {metric} but must be in {accepted_metrics}."
                )

        self.metrics = metrics
        self.rounding_decimals = rounding_decimals

    def __call__(self, lc_dict):
        """
        Computes the iteration-wise learning curve for each sample-wise anchor and a specific metric for a set of configurations, possibly across workflows and datasets.
        """

        anchors_sizes = QueryAnchorValues()(lc_dict)
        anchors_iterations_per_sample_size = QueryEpochValues()(lc_dict)

        data = {}

        for metric in self.metrics:
            for fold in self.folds:
                key = f"{metric}_{fold}"
                values = QueryMetricValuesFromEpochs(
                    self.srcs[metric], split_name=fold
                )(lc_dict)

                data[key] = []

                anchor_size_column = []
                anchor_iter_column = []
                for i, (
                    anchor_size,
                    anchors_iteration_for_this_sample_anchor,
                ) in enumerate(zip(anchors_sizes, anchors_iterations_per_sample_size)):
                    num_values_for_iteration_curve = len(values[i])
                    for j, anchor_iteration in enumerate(
                        anchors_iteration_for_this_sample_anchor
                    ):
                        if j >= num_values_for_iteration_curve:
                            break
                        data[key] = self.funs[metric](values[i][j])
                        if "anchor_size" not in data:
                            anchor_size_column.append(anchor_size)
                            anchor_iter_column.append(anchor_iteration)
                if "anchor_size" not in data:
                    data["anchor_size"] = anchor_size_column
                    data["anchor_iteration"] = anchor_iter_column
        return pd.DataFrame(data)


def integrate_sample_wise_curves(df_results, column_with_sample_wise_curves):
    dfs = []
    if column_with_sample_wise_curves not in df_results.columns:
        raise ValueError(
            f"Cannot extract sample-wise curves from inexistent field {column_with_sample_wise_curves}"
        )
    cols = list(df_results.columns)
    cols.remove(column_with_sample_wise_curves)
    for _, row in df_results.iterrows():
        df = row[column_with_sample_wise_curves]
        sub_cols = list(df.columns)
        for key in cols:
            df[key] = row[key]
        df = df[cols + sub_cols]
        dfs.append(df)
    return pd.concat(dfs)


def to_numpy(df_results, curve_column):

    hp_cols = [c for c in df_results.columns if c.startswith("p:")]
    num_configs = len(df_results.groupby(hp_cols))
    num_val_seeds = len(pd.unique(df_results["m:valid_seed"]))
    num_test_seeds = len(pd.unique(df_results["m:test_seed"]))
    num_wf_seeds = len(pd.unique(df_results["m:workflow_seed"]))

    tensor = np.zeros(
        (
            num_configs,
            num_val_seeds,
            num_test_seeds,
            num_wf_seeds,
            df_results[curve_column].iloc[0].shape[1] - 1,
            df_results[curve_column].iloc[0].shape[0],
        )
    )
    tensor[:] = np.nan

    anchors_sizes_overall = None

    for i1, (hp_conf, df_i1) in enumerate(df_results.groupby(hp_cols)):
        for i2, (val_seed, df_i2) in enumerate(df_i1.groupby("m:valid_seed")):
            for i3, (test_seed, df_i3) in enumerate(df_i2.groupby("m:test_seed")):
                for i4, (wf_seed, df_i4) in enumerate(df_i3.groupby("m:workflow_seed")):
                    lc = df_i4[curve_column].iloc[0]

                    is_iteration_curve = "anchor_iteration" in lc.columns

                    anchors_size = np.array(sorted(set(lc["anchor_size"])))
                    if is_iteration_curve:
                        anchors_iteration = np.array(
                            sorted(set(lc["anchor_iteration"]))
                        )
                        lc = lc.set_index(["anchor_size", "anchor_iteration"])
                    else:
                        lc = lc.set_index(["anchor_size"])

                    if anchors_sizes_overall is None:
                        anchors_sizes_overall = anchors_size
                    else:
                        if np.any(anchors_sizes_overall != anchors_size):
                            raise ValueError()

                    # if this is a sample-wise curve, check anchors
                    tensor[i1, i2, i3, i4, :, : lc.shape[0]] = lc.values.T
    return tensor


def merge_curves(curves):

    if len(curves) == 0:
        return "None"

    curves = list(curves)

    # make sure that all curves are on the same context
    workflows = set([c.workflow for c in curves])
    datasets = set([c.openmlid for c in curves])
    hpconfigs = set([str(c.hp_config) for c in curves])
    metrics = curves[0].metrics
    fold_names = curves[0].fold_names
    is_iteration_wise_curves = set([c.is_iteration_wise_curve for c in curves])

    if len(workflows) > 1:
        raise ValueError(f"The curves are defined on more than one workflow: {workflows}")
    if len(datasets) > 1:
        raise ValueError(f"The curves are defined on more than one dataset: {datasets}")
    if len(hpconfigs) > 1:
        raise ValueError(f"The curves are defined on more than one hyperparameter configuration: {hpconfigs}")
    if len(is_iteration_wise_curves) > 1:
        raise ValueError(f"Some of the curves are iteration-wise but other sample-wise curves.")
    workflow = list(workflows)[0]
    openmlid = list(datasets)[0]
    hp_config = curves[0].hp_config
    is_iteration_wise_curve = list(is_iteration_wise_curves)[0]

    # get seeds and anchors
    test_seeds = set()
    val_seeds = set()
    workflow_seeds = set()
    anchors_size = set()
    anchors_iteration = set()

    for c in curves:
        test_seeds |= set(c.test_seeds)
        val_seeds |= set(c.val_seeds)
        workflow_seeds |= set(c.workflow_seeds)
        anchors_size |= set(c.anchors_size)
        if is_iteration_wise_curve:
            anchors_iteration |= set(c.anchors_iteration)

        if c.metrics != metrics:
            raise ValueError(f"Inconsistent metrics: {c.metrics} vs {metrics}")
        if c.fold_names != fold_names:
            raise ValueError(f"Inconsistent fold names: {c.fold_names} vs {fold_names}")

    test_seeds = sorted(test_seeds)
    val_seeds = sorted(val_seeds)
    workflow_seeds = sorted(workflow_seeds)
    anchors_size = sorted(anchors_size)
    anchors_iteration = sorted(anchors_iteration) if is_iteration_wise_curve else None

    # initialize all values with nans
    if is_iteration_wise_curve:
        values = np.zeros((len(metrics), len(fold_names), len(test_seeds), len(val_seeds), len(workflow_seeds), len(anchors_size), len(anchors_iteration)))
    else:
        values = np.zeros((len(metrics), len(fold_names), len(test_seeds), len(val_seeds), len(workflow_seeds), len(anchors_size)))
    values[:] = np.nan

    # fill the values from the existing curves
    for c in curves:
        test_seed_indices = [test_seeds.index(a) for a in c.test_seeds]
        val_seed_indices = [val_seeds.index(a) for a in c.val_seeds]
        workflow_seed_indices = [workflow_seeds.index(a) for a in c.workflow_seeds]
        anchors_size_indices = [anchors_size.index(a) for a in c.anchors_size]

        ts_indexer = test_seed_indices if len(test_seed_indices) > 1 else slice(test_seed_indices[0], test_seed_indices[0] + 1)
        vs_indexer = val_seed_indices if len(val_seed_indices) > 1 else slice(val_seed_indices[0], val_seed_indices[0] + 1)
        ws_indexer = workflow_seed_indices if len(workflow_seed_indices) > 1 else slice(workflow_seed_indices[0], workflow_seed_indices[0] + 1)

        as_indexer = slice(0, len(anchors_size_indices)) if anchors_size_indices[-1] == len(anchors_size_indices) - 1 else anchors_size_indices

        if is_iteration_wise_curve:
            anchors_iteration_indices = [anchors_iteration.index(a) for a in c.anchors_iteration]
            ai_indexer = slice(0, len(anchors_iteration_indices))
            values[:, :, ts_indexer, vs_indexer, ws_indexer, as_indexer, ai_indexer] = c.values
        else:
            values[:, :, ts_indexer, vs_indexer, ws_indexer, as_indexer] = c.values

    return LearningCurve(
        workflow=workflow,
        openmlid=openmlid,
        hp_config=hp_config,
        values=values,
        metrics=metrics,
        fold_names=fold_names,
        test_seeds=test_seeds,
        val_seeds=val_seeds,
        workflow_seeds=workflow_seeds,
        anchors_size=anchors_size,
        anchors_iteration=anchors_iteration
    )

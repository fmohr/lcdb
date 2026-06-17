import numpy as np
import pandas as pd

from lcdb.analysis.json import (
    QueryAnchorValues,
    QueryAnchorTimes,
    QueryFitTimes,
    QueryTransformTimes,
    QueryPredictTimes,
    QueryPredictTimesFoldWise,
    QueryMetricTimes,
    QueryMetricTimesFoldWise
)
from lcdb.analysis.score import balanced_accuracy_from_confusion_matrix
from lcdb.db._learning_curves import LearningCurve

import time


class RuntimeExtractor:

    def __init__(
        self,
        rounding_decimals=4,
        return_none_on_error=True,
    ):
        pass

    def __call__(self, row):
        """
            Computes the sample-wise learning curve for a specific metric for a set of configurations, possibly across workflows and datasets.
        """
        if row["results"] is None:
            return {}

        lc_dict = row["results"]

        # determine anchors and whether this is an iteration curve
        try:
            t_start = time.time()
            anchors = QueryAnchorValues()(lc_dict)
            runtimes = {
                a: {}
                for a in anchors
            }

            # get fit times for anchors
            fittimes = QueryFitTimes()(lc_dict)
            for anchor, timestamps_at_anchor in zip(anchors, fittimes):
                runtimes[anchor]["fit"] = np.round(timestamps_at_anchor[2] - timestamps_at_anchor[1], 6)
            
            # get transform times
            transformtimes = QueryTransformTimes()(lc_dict)
            for anchor, timestamps_at_anchor in zip(anchors, transformtimes):
                for t in timestamps_at_anchor:
                    runtimes[anchor][t[0]] = np.round(t[2] - t[1], 6)

            # get predict times
            predicttimes = QueryPredictTimes()(lc_dict)
            for anchor, timestamps_at_anchor in zip(anchors, predicttimes):
                runtimes[anchor]["predictions"] = np.round(timestamps_at_anchor[2] - timestamps_at_anchor[1], 6)
            
            # overall metric computation time
            metrictimes = QueryMetricTimes()(lc_dict)
            for anchor, timestamps_at_anchor in zip(anchors, metrictimes):
                runtimes[anchor]["metrics"] = np.round(timestamps_at_anchor[2] - timestamps_at_anchor[1], 6)
            
            # runtimes for metrics per fold
            metrictimes_foldwise = {
                metric: QueryMetricTimesFoldWise(metric=metric)(lc_dict)
                for metric in ["confusion_matrix", "auc", "log_loss", "brier_score"]
            } 
            for metric, runtimes_for_metric in metrictimes_foldwise.items():
                for anchor, timestamps_at_anchor in zip(anchors, runtimes_for_metric):
                    for i, fold in enumerate(["train", "val", "test"]):
                        runtimes[anchor][f"{metric}_{fold}"] = np.round(timestamps_at_anchor[i][2] - timestamps_at_anchor[i][1], 6) if i in timestamps_at_anchor else 0.0
            
            # sanity checks
            for anchor, runtimes_at_anchor in runtimes.items():
                all_metrics_runtimes = {k: v for k, v in runtimes_at_anchor.items() if k not in ["fit", "epoch", "transform_train", "transform_valid", "transform_test", "predictions", "metrics"]}
                if "metrics" in runtimes_at_anchor:
                    assert np.round(sum(all_metrics_runtimes.values()), 6) <= runtimes_at_anchor["metrics"], f"Metric runtime {runtimes_at_anchor['metrics']} at anchor {anchor} is shorter than sum of all metric runtimes, which is {sum(all_metrics_runtimes.values())} based on {all_metrics_runtimes}"
            
            # anchor-wise summary
            runtimes["summary_by_anchor"] = {}
            anchor_timestamps = QueryAnchorTimes()(lc_dict)
            for anchor, (t_start, t_end) in zip(anchors, anchor_timestamps):
                runtimes["summary_by_anchor"][anchor] = np.round(t_end - t_start, 6)
            
            # tag-wise summary
            tags = list(runtimes[anchors[0]].keys())
            runtimes["summary_by_tag"] = {
                tag: np.round(sum([runtimes[anchor][tag] if tag in runtimes[anchor] else 0 for anchor in anchors]), 6)
                for tag in tags
            }

            # compute actual learn time (fit - transformation)
            runtimes["summary_by_tag"]["learn"] = np.round(runtimes["summary_by_tag"].get("fit", 0) - (runtimes["summary_by_tag"].get("transform_train", 0) + runtimes["summary_by_tag"].get("transform_valid", 0) + runtimes["summary_by_tag"].get("transform_test", 0)), 6)

            # return dictionary with runtimes
            return {
                "runtimes": runtimes
            }
            

        except KeyboardInterrupt:
            raise

        except:
            raise



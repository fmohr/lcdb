import numpy as np
import pandas as pd

from lcdb.analysis.json import (
    QueryAnchorValues,
    QueryFitTimes,
    QueryTransformTimes,
    QueryPredictTimes,
    QueryPredictTimesFoldWise,
    QueryMetricTimes,
    QueryMetricTimesFoldWise
)
from lcdb.analysis.score import balanced_accuracy_from_confusion_matrix
from lcdb.analysis._learning_curves import LearningCurve


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
        if not row["has_result"]:
            return None

        lc_dict = row["m:json"]

        # determine anchors and whether this is an iteration curve
        try:
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
                        runtimes[anchor][f"{metric}_{fold}"] = np.round(timestamps_at_anchor[i][2] - timestamps_at_anchor[i][1], 6)
            
            # sanity checks
            for anchor, runtimes_at_anchor in runtimes.items():
                all_metrics_runtimes = {k: v for k, v in runtimes_at_anchor.items() if k not in ["fit", "epoch", "transform_train", "transform_valid", "transform_test", "predictions", "metrics"]}
                if "metrics" in runtimes_at_anchor:
                    assert sum(all_metrics_runtimes.values()) <= runtimes_at_anchor["metrics"], f"Metric runtime {runtimes_at_anchor['metrics']} at anchor {anchor} is shorter than sum of all metric runtimes: {all_metrics_runtimes}"
            
            # tag-wise summary
            tags = runtimes[np.min(list(runtimes.keys()))]
            runtimes["summary"] = {
                tag: np.round(sum([runtimes[anchor][tag] if tag in runtimes[anchor] else 0 for anchor in anchors]), 6)
                for tag in tags
            }

            # compute actual learn time (fit - transformation)
            if "transform"
            runtimes["summary"]["learn"] = np.round(runtimes["summary"]["fit"] - (runtimes["summary"]["transform_train"] + runtimes["summary"]["transform_valid"] + runtimes["summary"]["transform_test"]), 6)

            # return dictionary with runtimes
            return runtimes
            

        except KeyboardInterrupt:
            raise

        except:
            raise



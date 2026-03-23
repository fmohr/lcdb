from lcdb.db import LCDB
from lcdb.analysis.util import LearningCurveExtractor
from lcdb.analysis.util import merge_curves

if __name__ == "__main__":

    datasets = [3]#, 6, 11, 12]
    workflows = [
        "lcdb.workflow.sklearn.KNNWorkflow",
        "lcdb.workflow.sklearn.LibLinearWorkflow",
        "lcdb.workflow.sklearn.LibSVMWorkflow",
        "lcdb.workflow.sklearn.TreesEnsembleWorkflow"
    ]

    lcdb = LCDB()

    df = lcdb.query(
        openmlids=[3],
        workflows=workflows[1],
        return_generator=False,
        processors={
            "learning_curve": LearningCurveExtractor(
                metrics=["error_rate"],
                folds=["train", "val"]
            )#, "balanced_error_rate"])
        },
        show_progress=True
    )

    config_cols = [c for c in df.columns if c.startswith("p:")]
    df = df.groupby(config_cols).agg({"learning_curve": merge_curves})
    print(df)

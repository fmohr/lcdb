class ExperimentStatisticsExtractor:

    def __init__(self):
        self.rows = []

    def __call__(self, row):
        """
            Computes the sample-wise learning curve for a specific metric for a set of configurations, possibly across workflows and datasets.
        """
        report = {}
        if not row["has_result"] or row["m:json"] is None:
            report["result"] = "missing"
            return report

        lc_dict = row["m:json"]
        report["result"] = "ok"
        return report


if __name__ == "__main__":

    from lcdb.db import LCDB

    lcdb = LCDB()
    lcdb.query(
        workflows=["lcdb.workflow.sklearn.LibLinearWorkflow"],
        test_seeds=[0],
        return_generator=False,
        processors={
            "statistics": ExperimentStatisticsExtractor()
        },
        show_progress=True
    ).to_csv("statistics.csv", index=False)
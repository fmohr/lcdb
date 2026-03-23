from lcdb.db import LCDB
import pandas as pd
from lcdb.analysis.json import QueryMetricValuesFromAnchors, QueryAnchorValues
from lcdb.analysis.score import balanced_accuracy_from_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from lcdb.db._util import print_tree
from lcdb.analysis.util import LearningCurveExtractor

import json
import pprint

if __name__ == "__main__":

    datasets = [3]#, 6, 11, 12]
    workflows = [
        "lcdb.workflow.sklearn.KNNWorkflow",
        "lcdb.workflow.sklearn.LibLinearWorkflow",
        "lcdb.workflow.sklearn.LibSVMWorkflow",
        "lcdb.workflow.sklearn.TreesEnsembleWorkflow"
    ]
    #workflows = workflows[0:1]

    if True:

        lcdb = LCDB()
        df_sample_wise_curves_per_workflow = {}

        df = lcdb.query(openmlids=[6], workflows=workflows[-1], return_generator=False, processors={
            "learning_curve": LearningCurveExtractor(metrics=["error_rate"], folds=["train", "val", "test", "oob"])#, "balanced_error_rate"])
            #"iteration_wise_curve": IterationWiseCurveExtractor(metrics=["error_rate"])
        }, show_progress=True)
        print(df["learning_curve"].iloc[0].values.shape)

        lc = df["learning_curve"].iloc[0].values[0]
        lc_diff = lc[2] - lc[3]  # test - oob performance

        fig, axs = plt.subplots(1, 3, figsize=(10, 4), subplot_kw={"projection": "3d"})

        # Make data.
        X = np.log10(np.array(df["learning_curve"].iloc[0].anchors_size))
        Y = np.array(df["learning_curve"].iloc[0].anchors_iteration)
        X, Y = np.meshgrid(X, Y)

        for ax in axs:
            ax.set_xlabel("log10 of sample anchor")
            ax.set_ylabel("iteration anchor")

            ax.set_box_aspect(None, zoom=0.8)

        # Plot the surfaces.
        ax = axs[0]
        surf = ax.plot_surface(X, Y, lc[2].T, cmap="Reds", vmin=0, vmax=1.0)
        ax.set_zlabel("test error")

        ax = axs[1]
        surf = ax.plot_surface(X, Y, lc[3].T, cmap="Blues", vmin=0, vmax=1.0)
        ax.set_zlabel("oob error")

        ax = axs[2]
        surf = ax.plot_surface(X, Y, lc_diff.T, cmap="coolwarm", vmin=-0.2, vmax=0.2)
        ax.set_zlabel("test error - oob error")

        # Adjust layout to avoid cutting off labels
        fig.tight_layout()

        fig.savefig("plots/oob_vs_test.pdf", bbox_inches="tight")
        plt.show()
        exit(0)

        for df in tqdm(lcdb.query(openmlids=[3], workflows=workflows[1:2])):

            df_next_curves = get_sample_wise_curves(df=df).drop(columns=["objective", "job_id", "m:json"])
            workflow = pd.unique(df_next_curves["m:workflow"])[0]
            df_sample_wise_curves_per_workflow[workflow] = df_next_curves if workflow not in df_sample_wise_curves_per_workflow else pd.concat([df_sample_wise_curves_per_workflow[workflow], df_next_curves])
            print_tree(df["m:json"].values[0])
            exit(0)

        #df_sample_wise_curves = get_sample_wise_curves(datasets=datasets, workflows=workflows).drop(columns=["objective", "job_id", "m:json"])
        for workflow_name, df_workflow_results in df_sample_wise_curves_per_workflow.items():
            df_workflow_results["sw_error_rate"] = df_workflow_results["sw_error_rate"].apply(lambda lc: [list(e) for e in lc])
            df_workflow_results.to_csv(f"error_rates_{workflow_name}.csv", index=False)
        exit(0)

    df_sample_wise_curves = pd.read_csv(f"error_rates_{workflows[1]}.csv")
    df_sample_wise_curves["sw_error_rate"] = df_sample_wise_curves["sw_error_rate"].apply(lambda lc: np.array(json.loads(lc)))
    df_sample_wise_curves["sw_error_rate_16"] = df_sample_wise_curves["sw_error_rate"].apply(lambda lc: lc[0][1])
    df_sample_wise_curves["sw_error_rate_64"] = df_sample_wise_curves["sw_error_rate"].apply(lambda lc: lc[4][1] if len(lc) >= 5 else 1.0)
    df_sample_wise_curves["error_rate"] = df_sample_wise_curves["sw_error_rate"].apply(lambda lc: lc[-1][1])

    config_cols = [c for c in df_sample_wise_curves.columns if c.startswith("p:")]
    seed_cols = ["m:valid_seed", "m:test_seed", "m:workflow_seed"]
    preprocessor_algo_cols = ["p:pp@cat_encoder", "p:pp@decomposition", "p:pp@featuregen", "p:pp@featureselector", "p:pp@scaler"]

    print(config_cols + ["m:openmlid"])
    df_agg = df_sample_wise_curves[config_cols + ["m:openmlid"] + seed_cols + ["sw_error_rate", "sw_error_rate_16", "sw_error_rate_64", "error_rate"]].groupby(config_cols + ["m:openmlid"]).mean()

    for openmlid, df_config_dataset in df_agg.groupby("m:openmlid"):

            mask_feature_selector = df_config_dataset["p:pp@featureselector"] != "none"
            mask_scaler = df_config_dataset["p:pp@scaler"] != "none"
            mask_decomposition = df_config_dataset["p:pp@decomposition"] != "none"
            mask_featuregen = df_config_dataset["p:pp@featuregen"] != "none"
            mask_any_pp = mask_feature_selector | mask_scaler | mask_decomposition | mask_featuregen

            df_with_pp = df_config_dataset[mask_any_pp]
            df_without_pp = df_config_dataset[~mask_any_pp]

            pp_better_at_16 = df_with_pp["sw_error_rate_16"].min() < df_without_pp["sw_error_rate_16"].min()
            pp_better_at_64 = df_with_pp["sw_error_rate_64"].min() < df_without_pp["sw_error_rate_64"].min()
            pp_better_at_final = df_with_pp["error_rate"].min() < df_without_pp["error_rate"].min()
            print(openmlid, len(df_with_pp), len(df_without_pp), pp_better_at_16, pp_better_at_64, pp_better_at_final)

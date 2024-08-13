import argparse
import fanova
import lcdb.db
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trees', type=int, default=16)
    parser.add_argument('--openml_ids', type=int, nargs='+', default=[3, 6])
    parser.add_argument('--workflow_name', type=str, default="lcdb.workflow.sklearn.LibLinearWorkflow")
    parser.add_argument('--openml_taskid_name', type=str, default="m:openmlid")
    return parser.parse_args()


def run(args):
    fanova_results = []
    performance_column = "objective"

    all_results = lcdb.db.LCDB().get_results(workflows=[args.workflow_name], openmlids=args.openml_ids)
    hyperparameter_names = [i for i in all_results.columns.values if i.startswith('p:')]
    relevant_columns = hyperparameter_names + [performance_column, args.openml_taskid_name]
    all_results = all_results[relevant_columns]

    task_ids = set(all_results[args.openml_taskid_name].to_list())

    for idx, task_id in enumerate(task_ids):
        logging.info("Starting with task %d (%d/%d)" % (task_id, idx + 1, len(task_ids)))
        # note that we explicitly only include tasks from the benchmark suite that was specified (as per the for-loop)
        task_results = all_results[all_results[args.openml_taskid_name] == task_id]

        # in this simplified example, we only display numerical and float hyperparameters. For categorical
        # hyperparameters, the fanova library needs to be informed by using a configspace object.
        task_results = task_results.select_dtypes(include=["int64", "float64"])
        # drop rows with unique values. These are by definition not an interesting hyperparameter, e.g., ``axis``,
        # ``verbose``.
        columns = [
            c
            for c in list(task_results)
            if len(task_results[c].unique()) > 1
        ]
        evaluator = fanova.fanova.fANOVA(
            X=task_results[columns].to_numpy(),
            Y=task_results[performance_column].to_numpy(),
            n_trees=args.n_trees,
        )
        for idx, pname in enumerate(columns):
            logging.info('-- hyperparameter %d %s' % (idx, pname))
            unique_values = task_results[pname].unique()
            logging.info('-- UNIQUE VALUES: %d (%s)' % (len(unique_values), unique_values))
            importance = evaluator.quantify_importance([idx])
            try:
                fanova_results.append(
                    {
                        "hyperparameter": pname,
                        "fanova": importance[(idx,)]["individual importance"],
                    }
                )
            except RuntimeError as e:
                print("Task %d error: %s" % (task_id, e))
                continue

    fanova_results = pd.DataFrame(fanova_results)

    fig, ax = plt.subplots()
    sns.boxplot(x="hyperparameter", y="fanova", data=fanova_results, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Variance Contribution")
    ax.set_xlabel(None)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())

import argparse
import ConfigSpace.hyperparameters
import fanova
import lcdb.db
import lcdb.builder.utils
import lcdb.analysis.json
import lcdb.analysis.score
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trees', type=int, default=16)
    parser.add_argument('--openml_ids', type=int, nargs='+', default=[3, 6])
    parser.add_argument('--workflow_name', type=str, default="lcdb.workflow.sklearn.LibLinearWorkflow")
    parser.add_argument('--openml_taskid_name', type=str, default="m:openmlid")
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/lcdb'))
    parser.add_argument('--output_filetype', type=str, choices=['pdf', 'png'], default='png')
    parser.add_argument('--max_load', type=int, default=None)
    return parser.parse_args()


def numeric_encode(df, config_space):
    # https://automl.github.io/ConfigSpace/latest/api/ConfigSpace/configuration_space/
    result = np.zeros((len(df), len(config_space.get_hyperparameters())), dtype=float)

    for hyperparameter_name, hyperparameter in config_space.items():
        index = config_space.get_idx_by_hyperparameter_name(hyperparameter_name)
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.NumericalHyperparameter):
            result[:, index] = df[hyperparameter_name].to_numpy()
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant):
            result[:, index] = [0] * len(df)
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter):
            lst = [hyperparameter.choices.index(value) for value in df[hyperparameter_name].to_numpy()]
            result[:, index] = lst
        else:
            raise ValueError('unsupported hyperparameter type: %s' % str(hyperparameter))
    return result


def fanova_on_task(task_results, performance_column_name, config_space, n_trees):
    fanova_results = []

    # in this simplified example, we only display numerical and float hyperparameters. For categorical
    # hyperparameters, the fanova library needs to be informed by using a configspace object.
    # task_results = task_results.select_dtypes(include=["int64", "float64"])

    # query_confusion_matrix_values = lcdb.analysis.json.QueryMetricValuesFromAnchors("confusion_matrix", split_name="val")
    # out = task_results[performance_column_name].apply(query_confusion_matrix_values)
    # print(out)
    # balanced_error_rate_values_for_config = np.array(
    #    out.apply(lambda x: list(map(lambda x: 1 - lcdb.analysis.score.balanced_accuracy_from_confusion_matrix(x), x))).to_list())
    # print(balanced_error_rate_values_for_config.mean(axis=0))
    # print(out)
    evaluator = fanova.fanova.fANOVA(
        X=numeric_encode(task_results, config_space),
        Y=task_results[performance_column_name].to_numpy(),
        config_space=config_space,
        n_trees=n_trees,
    )
    for idx, pname in enumerate(config_space.get_hyperparameter_names()):
        logging.info('-- hyperparameter %d %s' % (idx, pname))
        unique_values = task_results[pname].unique()
        logging.info('-- UNIQUE VALUES: %d (%s)' % (len(unique_values), unique_values))
        importance = evaluator.quantify_importance([idx])

        fanova_results.append(
            {
                "hyperparameter": pname,
                "fanova": importance[(idx,)]["individual importance"],
            }
        )

    return fanova_results


def run(args):
    fanova_all_results = []
    performance_column = "objective"

    WorkflowClass = lcdb.builder.utils.import_attr_from_module(args.workflow_name)
    config_space = WorkflowClass.config_space()
    workflow_hyperparameter_mapping = {"p:" + name: name for name in config_space.get_hyperparameter_names()}
    id_results = dict()
    print(config_space)  # TODO: properly integrate to fanova

    all_results_all_workflows = lcdb.db.LCDB().query(workflows=[args.workflow_name], openmlids=args.openml_ids)
    load_count = 0
    for frame_workflow_job_task in all_results_all_workflows:
        workflow_ids = frame_workflow_job_task['m:workflow'].unique()
        openml_task_ids = frame_workflow_job_task['m:openmlid'].unique()
        # job_ids = frame_workflow_job_task['job_id'].unique()
        if len(workflow_ids) > 1 or len(openml_task_ids) > 1:
            raise ValueError('Should not happen. %s %s' % (str(workflow_ids), str(openml_task_ids)))
        if (workflow_ids[0], openml_task_ids[0]) not in id_results:
            id_results[(workflow_ids[0], openml_task_ids[0])] = list()
        id_results[(workflow_ids[0], openml_task_ids[0])].append(frame_workflow_job_task)

        load_count += 1
        if args.max_load and load_count >= args.max_load:
            break

    task_ids = set()
    for idx, (workflow_name, task_id) in enumerate(id_results):
        task_ids.add(task_id)
        task_results = pd.concat(id_results[(workflow_name, task_id)])
        task_results = task_results.rename(workflow_hyperparameter_mapping, axis=1)
        relevant_columns = list(workflow_hyperparameter_mapping.values()) + [performance_column, args.openml_taskid_name]
        task_results = task_results[relevant_columns]

        logging.info("Starting with task %d (%d/%d)" % (task_id, idx + 1, len(id_results)))
        fanova_task_results = fanova_on_task(task_results, performance_column, config_space, args.n_trees)
        fanova_all_results.extend(fanova_task_results)

    fanova_all_results = pd.DataFrame(fanova_all_results)

    # generate plot
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.boxplot(x="hyperparameter", y="fanova", data=fanova_all_results, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Variance Contribution")
    ax.set_xlabel(None)
    plt.title('hyperparameter importance %s on task ids %s' % (args.workflow_name, str(task_ids)))
    plt.tight_layout()

    # save plot to file
    output_file = args.output_directory + '/fanova_%s.%s' % (args.workflow_name, args.output_filetype)
    os.makedirs(args.output_directory, exist_ok=True)
    plt.savefig(output_file)
    logging.info('saved to %s' % output_file)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())

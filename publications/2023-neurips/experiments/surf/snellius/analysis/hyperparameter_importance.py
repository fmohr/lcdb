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
    parser.add_argument('--openml_ids', type=int, nargs='+', default=[3])
    parser.add_argument('--workflow_name', type=str, default="lcdb.workflow.sklearn.LibLinearWorkflow")
    parser.add_argument('--openml_taskid_name', type=str, default="m:openmlid")
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/lcdb'))
    parser.add_argument('--output_filetype', type=str, choices=['pdf', 'png'], default='png')
    parser.add_argument('--max_load', type=int, default=None)
    parser.add_argument('--anchor_value', type=int, default=2048)
    return parser.parse_args()


def numeric_encode(df, config_space):
    # https://automl.github.io/ConfigSpace/latest/api/ConfigSpace/configuration_space/
    result = np.zeros((len(df), len(config_space.values())), dtype=float)

    for hyperparameter_name, hyperparameter in config_space.items():
        index = config_space.index_of[hyperparameter_name]
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

    evaluator = fanova.fanova.fANOVA(
        X=numeric_encode(task_results, config_space),
        Y=task_results[performance_column_name].to_numpy(),
        config_space=config_space,
        n_trees=n_trees,
    )
    for idx, pname in enumerate(config_space.keys()):
        logging.info('-- hyperparameter %d %s' % (idx, pname))
        unique_values = task_results[pname].unique()
        logging.info('-- UNIQUE VALUES: %d (%s)' % (len(unique_values), unique_values))
        importance = evaluator.quantify_importance([idx])

        fanova_results.append({
            "hyperparameter": pname,
            "fanova": importance[(idx,)]["individual importance"],
        })
    return fanova_results


def run(args):
    fanova_all_results = []
    performance_column = "objective"
    anchor_size_column = "anchor_sizes"
    learning_curve_column = "learning_curve_data"

    WorkflowClass = lcdb.builder.utils.import_attr_from_module(args.workflow_name)
    config_space = WorkflowClass.config_space()
    workflow_hyperparameter_mapping = {"p:" + name: name for name in config_space.keys()}
    id_results = dict()

    all_results_all_workflows = lcdb.db.LCDB().query(
        workflows=[args.workflow_name],
        openmlids=args.openml_ids,
        processors={
            anchor_size_column: lcdb.analysis.json.QueryAnchorValues(),
            learning_curve_column: lambda x: list(map(
                lambda x: 1 - lcdb.analysis.score.balanced_accuracy_from_confusion_matrix(x),
                lcdb.analysis.json.QueryMetricValuesFromAnchors("confusion_matrix", split_name="val")(x)
            ))
        }
    )
    load_count = 0
    for frame_workflow_job_task in all_results_all_workflows:
        workflow_ids = frame_workflow_job_task['m:workflow'].unique()
        openml_task_ids = frame_workflow_job_task['m:openmlid'].unique()
        # job_ids = frame_workflow_job_task['job_id'].unique()
        if len(workflow_ids) > 1 or len(openml_task_ids) > 1:
            raise ValueError('Should not happen. %s %s' % (str(workflow_ids), str(openml_task_ids)))
        if (workflow_ids[0], openml_task_ids[0]) not in id_results:
            id_results[(workflow_ids[0], openml_task_ids[0])] = list()

        performance_values_new = list()
        for index, row in frame_workflow_job_task.iterrows():
            anchor_sizes = row[anchor_size_column]
            performance_value_at_anchor = np.nan
            if args.anchor_value is not None:
                if args.anchor_value not in anchor_sizes:
                    logging.warning('Anchor %d not available in task %d workflow %s'
                                    % (args.anchor_value, openml_task_ids[0], workflow_ids[0])
                    )
                else:
                    anchor_index = anchor_sizes.index(args.anchor_value)
                    performance_value_at_anchor = row[learning_curve_column][anchor_index]
            else:
                performance_value_at_anchor = row[learning_curve_column][-1]
            performance_values_new.append(performance_value_at_anchor)
        performance_values_new = np.array(performance_values_new, dtype=float)
        frame_workflow_job_task[performance_column] = pd.Series(performance_values_new)

        id_results[(workflow_ids[0], openml_task_ids[0])].append(frame_workflow_job_task)

        load_count += 1
        if args.max_load and load_count >= args.max_load:
            break

    task_ids = set()
    for idx, (workflow_name, task_id) in enumerate(id_results):
        task_ids.add(task_id)
        task_results = pd.concat(id_results[(workflow_name, task_id)])
        task_results = task_results.rename(workflow_hyperparameter_mapping, axis=1)
        relevant_columns = list(workflow_hyperparameter_mapping.values()) + [performance_column]
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
    filename_suffix = ""
    if args.anchor_value is not None:
        filename_suffix = "_anchor_%d" % args.anchor_value
    output_file = args.output_directory + '/fanova_%s%s.%s' % (args.workflow_name, filename_suffix, args.output_filetype)
    os.makedirs(args.output_directory, exist_ok=True)
    plt.savefig(output_file)
    logging.info('saved to %s' % output_file)


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())

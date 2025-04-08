"""Command line to create summaries of data availability and experiment meta-data (memory and time consumptions) from LCDB 2.0 repositories"""
import pandas as pd
import csv
from lcdb.db import LCDB
import os
import numpy as np
from lcdb.cli._utils import get_true_mean, get_cores

workflow_mapping = {
  "libsvm": "lcdb.workflow.sklearn.LibSVMWorkflow",
  "randomforest": "lcdb.workflow.sklearn.RandomForestWorkflow",
  "knn": "lcdb.workflow.sklearn.KNNWorkflow",
  "xgboost": "lcdb.workflow.xgboost.XGBoostWorkflow",
  "treesensemble": "lcdb.workflow.sklearn.TreesEnsembleWorkflow",
  "liblinear": "lcdb.workflow.sklearn.LibLinearWorkflow"
}

def add_subparser(subparsers):
    """
    :meta private:
    """
    subparser_name = "stats"
    function_to_call = main

    subparser = subparsers.add_parser(
        subparser_name, help="Summarizes result availability contained LCDB 2.0 repositories."
    )

    subparser.add_argument(
        "-w",
        "--workflow-class",
        type=str,
        required=False,
        help="The 'path' of the workflow to train.",
    )

    subparser.add_argument(
        "-c",
        "--campaign-name",
        type=str,
        required=False,
        help="The name of the campaign to use.",
        # help="comma separated campaign names to use.",
    )

    subparser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=False,
        help="The path to the statistics folder.",
    )

    subparser.add_argument(
        "-n",
        "--num-configs",
        type=int,
        required=False,
        default=20,
        help="The number of configurations to use.",
    )

    subparser.set_defaults(func=function_to_call)


def read_numeric_csv(csv_path):
    """
    Reads a single-column CSV and converts it to a set of integers.
    """
    df = pd.read_csv(csv_path, header=None)  # Read as no header
    return set(df.iloc[:, 0].dropna().astype(int))  # Force use first column, drop NaN, convert to int

def find_missing_values_simple(csv_a_path, csv_b_path, output_name='missing_from_b.csv'):
    """
    Finds numeric values in CSV A that are not in CSV B.
    Returns a DataFrame of missing IDs and saves to CSV.
    """
    def read_numeric_csv(csv_path):
        """
        Reads a single-column CSV and converts it to a set of integers.
        """
        df = pd.read_csv(csv_path, header=None) 
        return set(df.iloc[:, 0].dropna().astype(int)) 

    set_a = read_numeric_csv(csv_a_path)
    set_b = read_numeric_csv(csv_b_path)

    # values present in workflow but not in datasets_to_test
    missing_values = sorted(list(set_a - set_b))

    missing_df = pd.DataFrame(missing_values, columns=["OpenML_ID"])

    return missing_df

def create_folder(stats_path):
    """
    Creates a folder for the statistics.
    """
    import os

    if not os.path.exists(stats_path):
        os.makedirs(stats_path)
        print("Created statistics folder. Path: ", stats_path)


def get_datasets_inside_campaign(root_dict, workflow_class, campaign_name):
    current = root_dict
    path = ['data',  workflow_class, campaign_name]
    for folder_name in path:
        contents = current.get('contents', [])
        # Look for a folder with the matching name
        next_folder = next((item for item in contents if item.get('name') == folder_name), None)
        if next_folder is None:
            return None  
        current = next_folder
    dataset_ids = [item['name'] for item in current.get('contents', []) if item.get('isfolder')] 
    return sorted(dataset_ids, key=lambda x: int(x))      

def save_to_csv(folders, filename="folders.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # writer.writerow(["Folder Name"])  # Add header row
        for folder in folders:
            writer.writerow([folder])
    print(f"Saved {len(folders)} folders to {filename}") 



def adjust_error_rate(df):
    # TODO: ensure that error_rate calculation is correct
    # we currently error_rate is the number of missing 'm:json'

    # calculate total missing results per row
    missing_results = df['num_configs'] * df['error_rate']

    # count explicit errors ('errors' is a list of errors for each row (i.e. dataset))
    explicit_errors = df['errors'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # adjust error rate to exclude explicit errors (not memory related)
    adjusted_error_rate = (missing_results - explicit_errors) / df['num_configs']


    return adjusted_error_rate


def add_memory_bins(df, memory_bins, memory_labels):
    df["memory_gb"] = df["mean_memory"] / (1024**3)

    df["memory_bin"] = pd.cut(df["memory_gb"], bins=memory_bins, labels=memory_labels, right=False)

    def move_to_next_bin(row):
        # high memory impact
        if row["adjusted_error_rate"] > 0.10:
            current_bin = row["memory_bin"]
            #  ensuring that it is not the highest bin already
            if current_bin in memory_labels[:-1]:  
                next_index = memory_labels.index(current_bin) + 1
                return memory_labels[next_index]
        return row["memory_bin"]

    df["memory_bin"] = df.apply(move_to_next_bin, axis=1)

def get_workflow_groupby(df, memory_labels):

  # group by workflow and memory_bin, listing datasets per bin
  grouped_workflow = df.groupby(["workflow", "memory_bin"])["openmlid"].unique().reset_index()

  grouped_workflow["openmlid"] = grouped_workflow["openmlid"].apply(lambda x: list(x) if isinstance(x, (np.ndarray, list)) else [])

  grouped_workflow = grouped_workflow[grouped_workflow["openmlid"].map(len) > 0]

  # order grouped_workflow by memory_bin
  grouped_workflow["memory_bin"] = pd.Categorical(grouped_workflow["memory_bin"], categories=memory_labels, ordered=True)
  grouped_workflow = grouped_workflow.sort_values(by=["workflow", "memory_bin"])

  return grouped_workflow


def get_memory_gorupby(df, memory_labels):
    # grouped by memory bin and collect (workflow, openmlid) pairs as a list
    grouped = df.groupby("memory_bin").agg(
        workflow_dataset_combinations=("workflow", lambda x: list(set(zip(x, df.loc[x.index, "openmlid"]))))
    ).reset_index()

    # remove bins with no tuples
    grouped = grouped[grouped["workflow_dataset_combinations"].map(len) > 0]

    grouped["memory_bin"] = pd.Categorical(grouped["memory_bin"], categories=memory_labels, ordered=True)
    grouped = grouped.sort_values(by=["memory_bin"])

    # add counter per bin
    def count_workflow_dataset_combinations(row):
        return len(row["workflow_dataset_combinations"])
    grouped["count"] = grouped.apply(count_workflow_dataset_combinations, axis=1)
    return grouped

def get_memory_time_estimation(df):
    results = []
    for workflow, openmlid in df[["workflow", "openmlid"]].values:
        # get 90th percentile memory and compute time (per config)
        config_time = get_true_mean(df, workflow, openmlid, column_name="config_time")
        memory = get_true_mean(df, workflow, openmlid, column_name="memory")
        
        # convert percentile memory to GB
        memory_gb = memory / (1024**3)
        cores = get_cores(memory_gb, number_of_nodes=1)

        # calculating expected runtime for 1K configs
        expected_runtime = (config_time * 1000) / cores
        expected_runtime_hours = expected_runtime / 3600  

        results.append({
            "workflow_class": workflow,
            "openmlid": openmlid,
            "memory_required_GB": memory_gb,
            "cores_needed": cores,
            "expected_runtime_hours-1k_configs": expected_runtime_hours
        })

    # convert results to a DataFrame
    df_results = pd.DataFrame(results)
    return df_results


def main(
    workflow_class,
    campaign_name,
    output_dir,
    num_configs,
):

    lcdb = LCDB()
    ## @Andreas: please add your logic here.

    # get pcloud repo
    if not lcdb.loaded:
        lcdb._load()
    repositories = list(lcdb.repositories.values())
    pcloud_repo = repositories[0]
    print(f"Using PCloud repository: {pcloud_repo}")
    print(f"Pcloud content: {pcloud_repo.content}")


    openml_ids_on_pcloud = get_datasets_inside_campaign(
        pcloud_repo.content['metadata'],
        workflow_class,
        campaign_name,
    )

    # save locally the list of datasets on pcloud
    stats_full_path = f"{output_dir}/{campaign_name}/{workflow_class}"
    create_folder(stats_full_path)

    # convert openml_ids to int
    openml_ids_on_pcloud = [int(x) for x in openml_ids_on_pcloud]

    openml_ids_on_pcloud_path = f"{stats_full_path}/datasets_on_pcloud.csv"

    if openml_ids_on_pcloud:
        save_to_csv(openml_ids_on_pcloud, filename=openml_ids_on_pcloud_path)
    else:
        print(f"No datasets found in PCloud for campaign={campaign_name}, workflow={workflow_class}")
        return  

    # find missing datasets from expected run and pcloud
    missing_datasets = find_missing_values_simple(
        f"datasets_to_test.csv",
        openml_ids_on_pcloud_path,
    )

    missing_datasets.to_csv(f"{stats_full_path}/missing_datasets_from_pcloud.csv", index=False, header=False)


    stats_info = lcdb.statistics(
            openmlids=openml_ids_on_pcloud,
            workflows=[workflow_class],
            campaigns=[campaign_name],
            validation_seeds=[0],
            test_seeds=[0],
            show_progress=True,
            num_configs=num_configs
    )
    stats_info['adjusted_error_rate'] = adjust_error_rate(stats_info)

    # save the stats
    statistics_res_path = f"{stats_full_path}/stats.csv"
    stats_info.to_csv(statistics_res_path, index=False)

    # grouped workflow by memory bin
    memory_bins = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, float("inf")]
    memory_labels = ["<1GB", "1GB", "2GB", "4GB", "8GB", "16GB", "24GB", "32GB", "48GB", "64GB+"]
    add_memory_bins(stats_info, memory_bins, memory_labels)
    grouped_workflow = get_workflow_groupby(stats_info, memory_labels)
    grouped_workflow.to_csv(f"{stats_full_path}/grouped_workflow.csv", index=False)

    # grouped by memory
    grouper_memory = get_memory_gorupby(stats_info, memory_labels)
    grouper_memory.to_csv(f"{stats_full_path}/grouped_memory.csv", index=False)

    # get memory and time estimation 
    df_results = get_memory_time_estimation(stats_info)
    df_results.to_csv(f"{stats_full_path}/compute_needs.csv", index=False) 

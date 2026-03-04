import logging
from lcdb.db import LCDB
from lcdb.db._results import ResultSet
from lcdb.db.callbacks._base import LCDBCallback
from lcdb.db.processors import LearningCurveExtractor, RuntimeExtractor
from lcdb.db.processors._traceback_extractor import TracebackExtractor
from lcdb.db.processors._anticipated_memory import extract_anticipated_memory
from lcdb.db.processors._payload import compute_payload
from lcdb.db.processors._data_memory import DataMemoryComputer
import json

from pathlib import Path

from tqdm import tqdm

import pandas as pd

logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "./data"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# retrieve learning curve objects
lcdb = LCDB()


class Callback(LCDBCallback):

    def __init__(self, results_per_dataset):
        super().__init__()
        self.results_per_dataset = results_per_dataset
    
    def on_workflow_finished(self, workflow, total_num_records):
        pass
    
    def on_workflow_dataset_combination_finished(self, workflow, openmlid, total_num_records):
        print("\n\nFINISHED\n\n")

        folder = Path(f"{OUTPUT_DIR}/{workflow}")
        
        # now store the results in a jsonl file per dataset
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        
        file = Path(f"{folder}/{openmlid}.jsonl")
        if openmlid not in result_sets_per_dataset:
            print("No data, ignoring results")
            return
        print(f"Writing {len(result_sets_per_dataset[openmlid])} results for dataset {openmlid} to {file}. {total_num_records} have been generated.")
        result_sets_per_dataset[openmlid].save(file)
    
    def on_seed_combo_finished(self, workflow, openmlid, test_seed, validation_seed, workflow_seed, total_num_records):
        pass

for workflow_class in [
    "lcdb.workflow.sklearn.KNNWorkflow",
    "lcdb.workflow.sklearn.LibLinearWorkflow",
    "lcdb.workflow.sklearn.LibSVMWorkflow",
    "lcdb.workflow.sklearn.TreesEnsembleWorkflow",
    "lcdb.workflow.xgboost.XGBoostWorkflow",
    "lcdb.workflow.keras.DenseNNWorkflow"
]:
    print(workflow_class)

    folder = Path(f"{OUTPUT_DIR}/{workflow_class}")
    available_datasets_for_workflow = [int(p.name[:-6]) for p in folder.glob("*.jsonl")] if folder.exists else []
    print(f"Ignoring results for dataset {available_datasets_for_workflow}")

    campaign_name = "probing-test"
    #if "XGBoost" in workflow_class:
        #campaign_name = "pre-config-100-new"

    result_sets_per_dataset = {}
    gen = lcdb.query(
        campaigns=[campaign_name],
        #openmlids=[3, 12, 23, 31, 54],
        #openmlids=[41167],
        workflows=[workflow_class],
        processors=[
            LearningCurveExtractor(metrics=["error_rate"], encode_as_json_str=True),
            compute_payload,
            TracebackExtractor(),
            DataMemoryComputer(),
            RuntimeExtractor(),
            extract_anticipated_memory
        ],
        inclusion_predicate=lambda workflow, openmlid, campaign, workflow_seed, test_seed, val_seed: openmlid not in available_datasets_for_workflow + [41167],
        max_workers=8,
        buffer_size=4,
        batch_size=10,
        callbacks=[Callback(results_per_dataset=result_sets_per_dataset)]
    )

    if gen is None:
        print("No results found")
        continue

    # get all dataframes

    for i, chunk_rs in enumerate(tqdm(gen)):
        assert type(chunk_rs) == ResultSet, f"Expected ResultSet but got {type(chunk_rs)}"
        chunk_rs.drop_raw_results()

        # store the results dataset wise
        for openmlid, rs_dataset in chunk_rs.group_by_dataset():
            if openmlid not in result_sets_per_dataset:
                result_sets_per_dataset[openmlid] = ResultSet()
            result_sets_per_dataset[openmlid].extend(rs_dataset)

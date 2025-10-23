import logging
from lcdb.db import LCDB
from lcdb.db._results import ResultSet
from lcdb.db.processors import LearningCurveExtractor, RuntimeExtractor
import re
import json

from pathlib import Path

from tqdm import tqdm

import pandas as pd

logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "./experiments/debugging/data"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# retrieve learning curve objects
lcdb = LCDB()

def compute_payload(row):
    return {"payload": len(str(row["results"])) if row["results"] is not None else 0}


def extract_anticipated_memory(row):
    """
    Extract anticipated memory values (predicted and max in GB) from m:json if present.
    Returns just the predicted GB if found, otherwise None.
    """
    try:
        # parse the JSON (it's stored as string in row["m:json"])
        data = row["m:json"]
        if isinstance(data, str):
            data = json.loads(data)

        # recursive search through dicts/lists for traceback
        def find_traceback(obj):
            if isinstance(obj, dict):
                if "traceback" in obj.get("metadata", {}):
                    return obj["metadata"]["traceback"]
                for child in obj.get("children", []):
                    res = find_traceback(child)
                    if res:
                        return res
            elif isinstance(obj, list):
                for item in obj:
                    res = find_traceback(item)
                    if res:
                        return res
            return None

        traceback = find_traceback(data)
        if traceback:
            # regex to capture GB values
            match = re.search(
                r"consume approximately ([0-9.]+) GB.*?maximum is ([0-9.]+) GB",
                traceback,
                re.DOTALL,
            )
            if match:
                predicted, maximum = match.groups()
                return float(predicted)
        return None
    except Exception:
        return None


for workflow_class in [
    "lcdb.workflow.sklearn.KNNWorkflow",
    "lcdb.workflow.sklearn.LibLinearWorkflow",
    "lcdb.workflow.sklearn.LibSVMWorkflow",
    "lcdb.workflow.sklearn.TreesEnsembleWorkflow",
    "lcdb.workflow.xgboost.XGBoostWorkflow"
]:

    file = Path(f"{OUTPUT_DIR}/{workflow_class}.jsonl")
    if not file.exists():

        print(workflow_class)

        campaign_name = "liblinear-new"
        #if "XGBoost" in workflow_class:
            #campaign_name = "pre-config-100-new"

        gen = lcdb.query(
            campaigns=[campaign_name],
            workflows=[workflow_class],
            processors=[
                LearningCurveExtractor(metrics=["error_rate"]),
                compute_payload,
                RuntimeExtractor(),
                #"anticipated_memory": extract_anticipated_memory,
            ]
        )

        if gen is None:
            print("No results found")
            continue

        # get all dataframes
        results = None
        for chunk_rs in tqdm(gen):
            assert type(chunk_rs) == ResultSet, f"Expected ResultSet but got {type(chunk_rs)}"
            chunk_rs.drop_raw_results()
            if results is None:
                results = chunk_rs
            else:
                results.extend(chunk_rs)
        if results is not None:
            print(results.num_rows)

            # serialize learning curves
            #df["learning_curve"] = df["learning_curve"].apply(lambda lc: lc.to_json() if lc is not None else None)
            #df["runtimes"] = df["runtimes"].apply(lambda c: json.dumps(c) if c is not None else None)

            # save to CSV
            #df.to_csv(file, index=False)
            results.save(file)
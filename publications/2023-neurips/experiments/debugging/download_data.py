from lcdb.db import LCDB
from lcdb.analysis import LearningCurveExtractor
import re
import json

from pathlib import Path

from tqdm import tqdm

import pandas as pd

OUTPUT_DIR = "./experiments/debugging/data"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# retrieve learning curve objects
lcdb = LCDB()

def compute_payload(row):
    return len(str(row["m:json"]))



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

    file = Path(f"{OUTPUT_DIR}/{workflow_class}.csv")
    if not file.exists():

        print(workflow_class)

        campaign_name = "pre-config-100"
        if "XGBoost" in workflow_class:
            campaign_name = "pre-config-100-new"

        gen = lcdb.query(
            campaigns=[campaign_name],
            workflows=[workflow_class],
            test_seeds=[0],
            return_generator=True,
            processors={
                "learning_curve": LearningCurveExtractor(metrics=["error_rate"]),
                "payload": compute_payload,
                "anticipated_memory": extract_anticipated_memory,
            },
            show_progress=True
        )

        # get all dataframes
        dfs = []
        for chunk_df in tqdm(gen):
            dfs.append(chunk_df)
        df = pd.concat(dfs)

        # save to CSV
        df.to_csv(file, index=False)
import logging
from lcdb.db import LCDB
from lcdb.db._results import ResultSet
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

OUTPUT_DIR = "./experiments/debugging/data"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# retrieve learning curve objects
lcdb = LCDB()

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

        campaign_name = "probing"
        #if "XGBoost" in workflow_class:
            #campaign_name = "pre-config-100-new"

        gen = lcdb.query(
            campaigns=[campaign_name],
            workflows=[workflow_class],
            processors=[
                LearningCurveExtractor(metrics=["error_rate"]),
                compute_payload,
                TracebackExtractor(),
                DataMemoryComputer(),
                #RuntimeExtractor(),
                extract_anticipated_memory
            ]
        )

        if gen is None:
            print("No results found")
            continue

        # get all dataframes
        results = None
        for i, chunk_rs in enumerate(tqdm(gen)):
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
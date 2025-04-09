from abc import ABC

from lcdb.db import LCDB
from tqdm import tqdm

import pandas as pd
import json


class JsonBasedLCDBView(ABC):

    def __init__(self, processors):
        self.processors = processors

    def load_data(
        self,
        csv=None,
        repositories=None,
        campaigns=None,
        workflows=None,
        openmlids=None,
        workflow_seeds=None,
        test_seeds=None,
        validation_seeds=None,
        show_progress=False
    ):
        
        if csv is None:
            """
                Retrieves rows that this view is interested in
            """
            lcdb = LCDB()
            gen = lcdb.query(
                repositories=repositories,
                campaigns=campaigns,
                workflows=workflows,
                openmlids=openmlids,
                workflow_seeds=workflow_seeds,
                test_seeds=test_seeds,
                validation_seeds=validation_seeds,
                show_progress=show_progress,
                processors=self.processors
            )

            dfs = []
            for df in tqdm(gen, disable=not show_progress):
                if df is not None:
                    df = self.filter_results(df)
                    if df is not None and len(df) > 0:
                        dfs.append(df)
            self.df = pd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0] if dfs else None
        else:
            self.df = pd.read_csv(csv)
            for key, processor in self.processors.items():
                self.df[key] = self.df[key].apply(lambda s: json.loads(s))
            self.df = self.filter_results(self.df)
    
    def save_data(self, filename):
        df_c = self.df.copy()
        for key, processor in self.processors.items():
            df_c[key] = df_c[key].apply(json.dumps)
        df_c.to_csv(filename, index=False)
    
    def filter_results(self, df):
        return df
    
    def reduce(self, filter_fun):
        self.df = self.df[self.df.apply(filter_fun, axis=1)]
    
    def get_cli_test_command(self, iter=None):
        if iter is None:
            iter = self.df
        if isinstance(iter, pd.DataFrame):
            return iter.apply(self.get_cli_test_command, axis=1)
        if isinstance(iter, list):
            return [self.get_cli_test_command(e) for e in iter]
        if isinstance(iter, (pd.Series, dict)):
            return (
                f"lcdb test"
                f" -i {int(iter['m:openmlid']) if iter['m:openmlid'] is not None else None}"
                f" -w {iter['m:workflow']}"
                f" --parameters='{json.dumps({k: v for k, v in iter.items() if k.startswith('p:')})}'"
            )
        raise ValueError(f"Unsupported data type {type(iter)}")

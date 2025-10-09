from abc import ABC

from lcdb.db import LCDB
from lcdb.analysis.views._util import get_cli_test_command
from tqdm import tqdm

import pandas as pd
import json
import jsonlines


class JsonBasedLCDBView(ABC):

    def __init__(self, processors):
        self.processors = processors

    def load_data(
        self,
        jsonl=None,
        rows=None,
        repositories=None,
        campaigns=None,
        workflows=None,
        openmlids=None,
        workflow_seeds=None,
        test_seeds=None,
        validation_seeds=None,
        show_progress=False,
        drop_results_field=True,
    ):
        
        if rows is None and jsonl is None:
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
                validation_seeds=validation_seeds
            )
            batches = gen
            apply_processors = True
        else:
            if rows is None:
                with jsonlines.open(jsonl) as reader:
                    rows = list(reader)
                apply_processors = True
            else:
                apply_processors = False
            batches = [rows]

        # apply processors and filter rows
        rows_kept = []
        for batch in tqdm(batches, disable=not show_progress):
            for row in batch:
                if apply_processors:
                    for processor in self.processors:
                        row.update(processor(row))

                if self.filter_result(row):
                    if drop_results_field:
                        if "m:json" in row:
                            del row["m:json"]
                    rows_kept.append(row)
        self.rows = rows_kept
    
    def drop_result_field(self):
        for row in self.rows:
            if "m:json" in row:
                del row["m:json"]

    def clone(self):
        new_view = self.__class__(self.processors)
        new_view.rows = self.rows.copy()
        return new_view
    
    @property
    def num_rows(self):
        return len(self.rows)

    def save_data(self, filename):
        with jsonlines.open(filename, mode='w') as writer:
            writer.write_all(self.rows)
    
    def filter_result(self, row):
        return True
    
    def reduce(self, filter_fun):
        """
            Keeps only rows for which filter_fun(row) is True

        Args:
            filter_fun (callable): the predicate function to filter rows
        """
        self.rows = [r for r in self.rows if filter_fun(r)]
    
    def get_cli_test_command(self, iter=None):
        if iter is None:
            iter = self.rows
        if type(iter) == int:
            return self.get_cli_test_command(self.rows[iter])
        if isinstance(iter, pd.DataFrame):
            return iter.apply(self.get_cli_test_command, axis=1)
        if isinstance(iter, list):
            return [self.get_cli_test_command(e) for e in iter]
        if isinstance(iter, (pd.Series, dict)):
            return get_cli_test_command(iter)
        raise ValueError(f"Unsupported data type {type(iter)}")

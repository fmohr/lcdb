from abc import ABC

import json
import jsonlines

from lcdb.db._learning_curves import LearningCurve, merge_curves
from lcdb.db.processors._learning_curve_extractors import LearningCurveExtractor

RESULT_KEY = "results"
BUILD_ISSUES_KEY = "build_issues"

def get_packed_results(rows):
    packed_rows = []
    for row in rows:
        row_copy = row.copy()
        if RESULT_KEY in row_copy and type(row_copy[RESULT_KEY]) == dict:
            row_copy[RESULT_KEY] = json.dumps(row[RESULT_KEY])
        if BUILD_ISSUES_KEY in row_copy and type(row_copy[BUILD_ISSUES_KEY]) == dict:
            row_copy[BUILD_ISSUES_KEY] = json.dumps(row[BUILD_ISSUES_KEY])
        packed_rows.append(row_copy)
    return packed_rows


class ResultSet(ABC):

    def __init__(self, rows=None, results_unpacked=False, build_issues_unpacked=False):
        super().__init__()
        self._rows = rows
        self.build_issues_unpacked = build_issues_unpacked
        self.results_unpacked = results_unpacked
    
    def _unpack_results(self):
        if self.results_unpacked:
            raise ValueError("Results are already unpacked.")
        
        for row in self._rows:
            if RESULT_KEY in row and row[RESULT_KEY] is not None:
                row[RESULT_KEY] = json.loads(row[RESULT_KEY])
        self.results_unpacked = True

    def _unpack_build_issues(self):
        if self.build_issues_unpacked:
            raise ValueError("Build Issues are already unpacked.")
        for row in self._rows:
            if BUILD_ISSUES_KEY in row and row[BUILD_ISSUES_KEY] is not None:
                row[BUILD_ISSUES_KEY] = json.loads(row[BUILD_ISSUES_KEY])
        self.build_issues_unpacked = True
    
    @classmethod
    def read_jsonl(cls, path_to_jsonl):
         rs = ResultSet()
         rs.load(path_to_jsonl)
         return rs
    
    @classmethod
    def concat(cls, result_sets):
        rs = result_sets[0].clone()
        for rs_i in result_sets[1:]:
            rs.extend(rs_i)
        return rs

    @property
    def num_rows(self):
        return len(self._rows) if self._rows is not None else 0
    
    @property
    def empty(self):
        return self._rows is None or len(self._rows) == 0
    
    @property
    def num_rows_with_results(self):
        return len([r for r in self._rows if RESULT_KEY in r and r[RESULT_KEY] is not None])
    
    @property
    def num_rows_with_build_issues(self):
        return len([r for r in self._rows if BUILD_ISSUES_KEY in r and r[BUILD_ISSUES_KEY] is not None])

    @property
    def datasets(self):
        return set(row["openmlid"] for row in self._rows) if self._rows is not None else set()
    
    @property
    def workflows(self):
        return set(row["workflow"] for row in self._rows) if self._rows is not None else set()

    def __len__(self):
        return self.num_rows

    def __getitem__(self, idx):
        if type(idx) == int:
            return self._rows[idx]
        if type(idx) == str:
            return [r[idx] if idx in r else None for r in self._rows] if self._rows is not None else []
    
    def __iter__(self):
        for i in range(self.num_rows):
            yield self._rows[i]

    def load(self, path_to_jsonl):
        if type(path_to_jsonl) != list:
            path_to_jsonl = [path_to_jsonl]
        self._rows = []
        for path in path_to_jsonl:
            with jsonlines.open(path) as reader:
                self._rows.extend(list(reader))

    def save(self, filename):
        with jsonlines.open(filename, mode='w') as writer:
            out = get_packed_results(self._rows) if self.results_unpacked else self._rows
            writer.write_all(out)
    
    def apply(self, processors):

        # make sure that we can process a list
        if type(processors) != list:
            processors = [processors]

        # apply processors
        for row in self._rows:
            for processor in processors:
                row.update(processor(row))
    
    def drop_fields(self, field):
        if type(field) != list:
            fields = [field]
        else:
            fields = field
        
        for row in self._rows:
            for field in fields:
                if field in row:
                    del row[field]

    def drop_raw_results(self):
        if self._rows is None:
            return
        for row in self._rows:
            if RESULT_KEY in row:
                del row[RESULT_KEY]

    def head(self, n):
        self._rows = self._rows[:n]
        return self

    def tail(self, n):
        self._rows = self._rows[-n:]
        return self
    
    def split_at_index(self, idx):
        head = ResultSet(rows=self._rows[:idx], results_unpacked=self.results_unpacked, build_issues_unpacked=self.build_issues_unpacked)
        tail = ResultSet(rows=self._rows[idx:], results_unpacked=self.results_unpacked, build_issues_unpacked=self.build_issues_unpacked)
        assert len(head) + len(tail) == len(self)
        return head, tail
    
    def drop_rows_with_build_issues(self):
        self._rows = [r for r in self._rows if "build_issues" not in r or r["build_issues"] is None]
        return self
    
    def drop_rows_without_build_issues(self):
        self._rows = [r for r in self._rows if "build_issues" in r and r["build_issues"] is not None]
        return self
    
    def _create_copy_with_rows(self, rows):
        new_view = self.__class__(rows,
            results_unpacked=self.results_unpacked,
            build_issues_unpacked=self.build_issues_unpacked
        )
        return new_view

    def clone(self):
        return self._create_copy_with_rows(self._rows.copy())

    def _group(self, field):

        # set boolean to see whether we are grouping on the config
        is_on_config = field == "config"

        # collect indices
        indices_by_key = {}
        for row in self._rows:
            key = frozenset(row["config"].items()) if is_on_config else row[field]
            if key not in indices_by_key:
                indices_by_key[key] = [row]
            else:
                indices_by_key[key].append(row)
        
        # return fields
        for key, rs in indices_by_key.items():
            out_key = {e[0]: e[1] for e in key} if is_on_config else key
            yield out_key, ResultSet(
                rs,
                results_unpacked=self.results_unpacked,
                build_issues_unpacked=self.build_issues_unpacked
            )

    def filter_datasets(self, openmlids, inplace=False):
        if type(openmlids) == int:
            openmlids = [openmlids]
        new_rows = [r for r in self._rows if r["openmlid"] in openmlids]
        if inplace:
            self._rows = new_rows
            return self
        else:
            return self._create_copy_with_rows(new_rows)
    
    def filter_workflows(self, workflows, inplace=False):
        if type(workflows) == str:
            workflows = [workflows]
        new_rows = [r for r in self._rows if r["workflow"] in workflows]
        if inplace:
            self._rows = new_rows
            return self
        else:
            return self._create_copy_with_rows(new_rows)

    def group_by_campaign(self):
        return self._group("campaign")

    def group_by_config(self):
        return self._group("config")
    
    def group_by_dataset(self):
        return self._group("openmlid")
    
    def group_by_workflow(self):
        return self._group("workflow")
    
    def group_by_workflow_seed(self):
        return self._group("workflow_seed")
    
    def group_by_validation_seed(self):
        return self._group("valid_seed")
    
    def group_by_test_seed(self):
        return self._group("test_seed")
    
    def split(self, filter_fun):
        rows_left = []
        rows_right = []
        if self._rows is not None:
            for row in self._rows:
                if filter_fun(row):
                    rows_left.append(row)
                else:
                    rows_right.append(row)
        return (
            ResultSet(rows=rows_left, results_unpacked=self.results_unpacked, build_issues_unpacked=self.build_issues_unpacked),
            ResultSet(rows=rows_right, results_unpacked=self.results_unpacked, build_issues_unpacked=self.build_issues_unpacked)
        )
    
    def extend(self, rs):

        if type(rs) != ResultSet:
            raise ValueError("Can only extend with another ResultSet object")
        
        if self._rows is None:
            self._rows = rs._rows
        else:
            self._rows.extend(rs._rows)
    
    def reduce(self, filter_fun):
        """
            Keeps only rows for which filter_fun(row) is True

        Args:
            filter_fun (callable): the predicate function to filter rows
        """
        self._rows = [r for r in self._rows if filter_fun(r)]
    
    def receive_result_visitor(self, visitor):

        def visit_node(n):
            visitor(n)
            if "children" in n:
                for c in n["children"]:
                    visit_node(c)
        
        
        visit_node()
    
    def filter(self, filter_fun):
        return ResultSet(
            rows=[r for r in self._rows if filter_fun(r)],
            results_unpacked=self.results_unpacked,
            build_issues_unpacked=self.build_issues_unpacked
        )
    
    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self._rows)
    
    def get_learning_curve(self, metrics=["error_rate"]):
        if not self.results_unpacked:
            self._unpack_results()
        self.apply(LearningCurveExtractor(metrics=metrics))
        return merge_curves([row["learning_curve"] for row in self._rows if row["learning_curve"] is not None])


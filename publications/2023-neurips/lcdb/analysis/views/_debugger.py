from lcdb.analysis.views._util import get_cli_test_command as _get_cli_test_command
from lcdb.db.processors._traceback_extractor import TracebackExtractor
from lcdb.db._results import ResultSet
import pandas as pd
import numpy as np


class Debugger:

    def __init__(self, rs: ResultSet):
        self._rs = rs
    
    def filter_result(self, row):

        # keep rows with a traceback
        if row.get("traceback_summary", None) is not None:
            return True
        
        # keep rows with a failure in the objective field
        if row.get("objective", "F") == "F":
            return True
        
        # keep rows without a JSON result
        mjson = row.get("m:json", None)
        if mjson is None:
            return True

        # if nothing strange was found, don't keep the row
        return False
    
    def get_error_messages(self, openmlid=None):
        msgs = set()
        for row in self._rs:
            if openmlid is not None and row.get("m:openmlid", None) != openmlid:
                continue
            s = row["traceback_summary"]
            if s is not None:
                for error in s:
                    msgs.add(error["message"])
        return msgs
    
    #def get_occurrences_of_error_message(self, msg):
        #for openmlid, s in zip(self.df["m:openmlid"], self.df["traceback_summary"]):
         #   for error in s:
                

    def get_error_dataset_matrix(self):
        datasets = self._rs.datasets
        messages = sorted(self.get_error_messages())

        matrix = np.zeros((len(datasets), len(messages)))
        for i, openmlid in enumerate(datasets):
            for message in self.get_error_messages(openmlid=openmlid):
                matrix[i, messages.index(message)] += 1
        return matrix
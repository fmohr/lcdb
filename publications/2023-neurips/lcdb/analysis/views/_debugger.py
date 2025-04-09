from lcdb.analysis.views._base import JsonBasedLCDBView
from lcdb.analysis.views._util import get_cli_test_command as _get_cli_test_command
from lcdb.analysis.processors._traceback_extractor import TracebackExtractor
import pandas as pd
import numpy as np


class Debugger(JsonBasedLCDBView):

    def __init__(self):
        super().__init__({
            "traceback_summary": TracebackExtractor()
        })
    
    def filter_results(self, df):
        return df[df["traceback_summary"].notna()]
    
    def get_error_messages(self, openmlid=None):
        msgs = set()
        df = self.df if openmlid is None else self.df[self.df["m:openmlid"] == openmlid]
        for s in df["traceback_summary"]:
            for error in s:
                msgs.add(error["message"])
        return msgs
    
    #def get_occurrences_of_error_message(self, msg):
        #for openmlid, s in zip(self.df["m:openmlid"], self.df["traceback_summary"]):
         #   for error in s:
                

    def get_error_dataset_matrix(self):
        datasets = sorted([int(i) for i in pd.unique(self.df["m:openmlid"])])
        messages = sorted(self.get_error_messages())

        matrix = np.zeros((len(datasets), len(messages)))
        for i, openmlid in enumerate(datasets):
            for message in self.get_error_messages(openmlid=openmlid):
                matrix[i, messages.index(message)] += 1
        return matrix
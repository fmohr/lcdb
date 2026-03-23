from lcdb.db.callbacks._base import LCDBCallback
import pandas as pd

class CountingCallback(LCDBCallback):

    def __init__(self):
        super().__init__()
        self.rows = []

    @property
    def df(self):
        return pd.DataFrame(self.rows, columns=["workflow", "openmlid", "count"])

    def on_workflow_finished(self, workflow, total_num_records):
        pass

    def on_workflow_dataset_combination_finished(self, workflow, openmlid, total_num_records):
        self.rows.append([workflow, openmlid, total_num_records])
    
    def on_seed_combo_finished(self, workflow, openmlid, test_seed, validation_seed, workflow_seed, total_num_records):
        self.rows.append([workflow, openmlid, test_seed, validation_seed, workflow_seed, total_num_records])
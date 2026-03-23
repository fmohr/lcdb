class LCDBCallback:

    def on_workflow_finished(self, workflow, total_num_records):
        raise NotImplementedError
    
    def on_workflow_dataset_combination_finished(self, workflow, openmlid, total_num_records):
        raise NotImplementedError
    
    def on_seed_combo_finished(self, workflow, openmlid, test_seed, validation_seed, workflow_seed, total_num_records):
        raise NotImplementedError
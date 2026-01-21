class DataMemoryComputer:

    def __call__(self, row):
        key = "memory_data"
        out = {key: None}
        if "results" in row and row["results"] is not None:
            load_task_entry = row["results"]["children"][0]
            assert load_task_entry["tag"] == "load_task"
            out[key] = round(max(load_task_entry["memory_end"] - load_task_entry["memory_start"], 0.1), 2)
        return out
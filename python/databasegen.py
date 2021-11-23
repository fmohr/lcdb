import lcdb

datasets = [1485, 1590, 1515, 1457, 1475, 1468, 1486, 1489, 23512, 23517, 4541, 4534, 4538, 4134, 4135, 40978, 40996, 41027, 40981, 40982, 40983, 40984, 40701, 40670, 40685, 40900,  1111, 42732, 42733, 42734, 40498, 41161, 41162, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41142, 41143, 41144, 41145, 41146, 41147, 41150, 41156, 41157, 41158,  41159, 41138, 54, 181, 188, 1461, 1494, 1464, 12, 23, 3, 1487, 40668, 1067, 1049, 40975, 31]

metrics = ["accuracy", "logloss"]
for metric in metrics:
    print(metric)
    df = lcdb.compile_dataframe_for_all_datasets_on_metric(datasets, metric)
    df.to_csv(f"lcdb/database-{metric}.csv", index = False)
    print(f"Saved database for {metric}")
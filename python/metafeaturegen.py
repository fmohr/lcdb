import lcdb
import pandas as pd
import logging

# setup logger for this test suite
logger = logging.getLogger('lcdb')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# set up relevant openml ids
ids = [3, 6, 12, 14, 16, 18, 21, 22, 23, 24, 26, 28, 30, 31, 32, 36, 38, 44, 46, 54, 57, 60, 179, 180, 181, 182, 183, 184, 185, 188, 273, 293, 300, 351, 354, 357, 389, 390, 391, 392, 393, 395, 396, 398, 399, 401, 485, 554, 679, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 743, 751, 752, 761, 772, 797, 799, 803, 806, 807, 813, 816, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847, 849, 866, 871, 881, 897, 901, 903, 904, 910, 912, 913, 914, 917, 923, 930, 934, 953, 958, 959, 962, 966, 971, 976, 977, 978, 979, 980, 991, 993, 995, 1000, 1002, 1018, 1019, 1020, 1021, 1036, 1037, 1039, 1040, 1041, 1042, 1049, 1050, 1053, 1059, 1067, 1068, 1069, 1111, 1116, 1119, 1120, 1128, 1130, 1134, 1138, 1139, 1142, 1146, 1161, 1166, 1216, 1242, 1457, 1461, 1464, 1468, 1475, 1485, 1486, 1487, 1489, 1494, 1501, 1515, 1569, 1590, 4134, 4135, 4136, 4137, 4534, 4538, 4541, 4552, 23380, 23512, 23517, 40497, 40498, 40668, 40670, 40685, 40691, 40701, 40900, 40926, 40971, 40975, 40978, 40981, 40982, 40983, 40984, 40996, 41026, 41027, 41064, 41065, 41066, 41138, 41142, 41143, 41144, 41145, 41146, 41147, 41150, 41156, 41157, 41158, 41159, 41161, 41162, 41163, 41164, 41165, 41166, 41167, 41168, 41169, 41946, 42732, 42733, 42734]

# 41991 currently excluded because does not fit into my RAM

# compute the meta-features
df_datasets = lcdb.compute_meta_data_of_dataset_collection(ids)

print(df_datasets)

# save the meta-features
df_datasets.to_csv("lcdb/datasets.csv", index=False)

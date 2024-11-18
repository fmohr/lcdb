import numpy as np
import pandas as pd


from tqdm.notebook import tqdm

from lcdb.data import load_task

from lcdb.data import load_task

# https://www.openml.org/search?type=data&sort=runs&status=active&tags.tag=OpenML-CC18
openml_task_ids = [
    3,
    6,
    11,
    12,
    14,
    15,
    16,
    18,
    22,
    23,
    28,
    29,
    31,
    32,
    37,
    38,
    44,
    46,
    50,
    54,
    151,
    182,
    188,
    300,
    307,
    375,
    458,
    469,
    554,
    1049,
    1050,
    1053,
    1063,
    1067,
    1068,
    1461,
    1462,
    1464,
    1468,
    1475,
    1478,
    1480,
    1485,
    1486,
    1487,
    1489,
    1494,
    1497,
    1501,
    1510,
    1590,
    4134,
    4534,
    4538,
    6332,
    23381,
    23517,
    40499,
    40668,
    40670,
    40701,
    40923,
    40927,
    40966,
    40975,
    40978,
    40979,
    40982,
    40983,
    40984,
    40994,
    40996,
    41027,
]

# number_of_features = []
# number_of_samples = []
# number_of_classes = []
# real_variables = []
# categorical_variables = []

# for task_id in tqdm(openml_task_ids):
#     (X, y), dataset_metadata = load_task(f"openml.{task_id}")

#     number_of_features.append(np.shape(X)[1])
#     number_of_samples.append(np.shape(X)[0])
#     number_of_classes.append(dataset_metadata["num_classes"])

#     categories = dataset_metadata["categories"]
#     real_variables.append(not (all(categories)))
#     categorical_variables.append(any(categories))

# table = pd.DataFrame(
#     {
#         "OpenML Id": openml_task_ids,
#         "#Features": number_of_features,
#         "#Samples": number_of_samples,
#         "#Classes": number_of_classes,
#         "Real Features": real_variables,
#         "Categorical Features": categorical_variables,
#     }
# )
# table

print(list(sorted(openml_task_ids)))
print(list(sorted(openml_task_ids_old)))

print(set(openml_task_ids) - set(openml_task_ids_old))
print(set(openml_task_ids_old) - set(openml_task_ids))

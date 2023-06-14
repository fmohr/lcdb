from typing import Union, Tuple

import numpy as np
import sklearn.model_selection


def random_split_from_array(
    *arrays: Tuple[np.ndarray],
    train_size: float = 0.9,
    stratify_with_targets: bool = False,
    shuffle: bool = True,
    random_state: Union[int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs a random split of the input arrays.

    Args:
        arrays (Tuple[np.ndarray]): arrays to be split.
        train_size (float, optional): proportion of the training out of the whole data, between [0,1]. Defaults to 0.9.
        stratify_with_targets (bool, optional): stratify the split according to label classes. Defaults to False.
        shuffle (bool, optional): randomly shuffle the array before splitting. Defaults to True.
        random_state (Union[int, np.random.RandomState], optional): random state of the shuffling process. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (X_train,  X_test, y_train, y_test) the splitted arrays.
    """
    return sklearn.model_selection.train_test_split(
        *arrays,
        train_size=train_size,
        random_state=random_state,
        stratify=arrays[1] if stratify_with_targets else None,
        shuffle=shuffle,
    )

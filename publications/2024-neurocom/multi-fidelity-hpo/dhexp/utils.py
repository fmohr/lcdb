import importlib
import time
from functools import wraps

import numpy as np

#! info [why is it important to use "wraps"]
#! http://gael-varoquaux.info/programming/decoration-in-python-done-right-decorating-and-pickling.html


def sleep(mu=60, std=20, random_state=None):
    rs = np.random.RandomState(seed=random_state)

    def decorator(run_function):
        @wraps(run_function)
        def wrapper(*args, **kwargs):
            t_sleep = rs.normal(loc=mu, scale=std)
            t_sleep = max(t_sleep, 0)
            time.sleep(t_sleep)
            objective = run_function(*args, **kwargs)
            return objective

        return wrapper

    return decorator


def import_attr_from_module(path: str):
    """Import an attribute from a module given its path.

    Example:

    >>> from lcdb.workflow import SVMWorkflow
    >>> import_attr_from_module("lcdb.workflow.SVMWorkflow") == SVMWorkflow
    """
    path = path.split(".")
    module_name, attr_name = (
        ".".join(path[:-1]),
        path[-1],
    )
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    return attr

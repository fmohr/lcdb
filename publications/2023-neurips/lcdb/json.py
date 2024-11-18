import json

import numpy as np
import logging

class LCDBJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check if instance is numpy type and convert
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)

        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError as e:
            print(f"Failed to serialize object {obj} of type {type(obj)}: {e}")
            raise


def dumps(*args, **kwargs):
    return json.dumps(*args, cls=LCDBJsonEncoder, **kwargs)

def loads(*args, **kwargs):
    return json.loads(*args, **kwargs)

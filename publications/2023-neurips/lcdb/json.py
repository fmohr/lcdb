import json

import numpy as np


class LCDBJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def dumps(*args, **kwargs):
    return json.dumps(*args, cls=LCDBJsonEncoder, **kwargs)

def loads(*args, **kwargs):
    return json.loads(*args, **kwargs)

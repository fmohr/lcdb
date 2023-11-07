import pandas as pd

class CurveDB:

    def __init__(self, train_curve, val_curve, test_curve, times, additional_data):
        """
        :param train_curve: a Curve object (performances and prediction and metric time data) built with the train data
        :param val_curve: a Curve object (performances and prediction and metric time data) built with the validation data
        :param test_curve: a Curve object (performances and prediction and metric time data) built with the test data
        :param times: an anchor-indexed dictionary with all sorts of recorded times, per anchor
        :param additional_data: an anchor-indexed dictionary. Anchors are here *always* training set sizes
            For each anchor, there must be a dictionary with the information name and the respective information (may be an object)
            A field for `fit_time` is mandatory
        """
        self.train_curve = train_curve
        self.val_curve = val_curve
        self.test_curve = test_curve
        self.times = times
        self.additional_data = additional_data

    def dump_to_dict(self):

        """
        Compiles the whole knowledge base into a single compact dictionary.
        This dictionary should not have redundant information
        (like column names repeated over different list elements etc.).

        :return: dict
        """

        def flatten(dictionary, parent_key='', separator='::', from_depth=0):
            items = []
            for key, value in dictionary.items():
                if from_depth > 0:
                    items.append((key, flatten(value, from_depth=from_depth - 1)))
                else:
                    new_key = parent_key + separator + key if parent_key else key
                    if isinstance(value, dict):
                        items.extend(flatten(value, new_key, separator=separator, from_depth=0).items())
                    else:
                        items.append((new_key, value))
            return dict(items)

        # format times as dataframe
        runtimes_per_anchor = flatten(self.times, from_depth=1)  # keep the anchor level
        anchors = sorted([int(a) for a in runtimes_per_anchor.keys()])
        time_names = sorted(runtimes_per_anchor[str(anchors[0])].keys())
        rows = []
        for anchor in anchors:
            row = [anchor]
            row.extend([runtimes_per_anchor[str(anchor)][name] for name in time_names])
            rows.append(row)
        times_as_dataframe = pd.DataFrame(rows, columns=["anchor"] + time_names)

        if len(self.train_curve) > 0:
            out = {
                "train_curve": self.train_curve.as_compact_dict(),
                "val_curve": self.val_curve.as_compact_dict(),
                "test_curve": self.test_curve.as_compact_dict(),
                "times": times_as_dataframe.to_dict(orient="list"),
                "additional_data": self.additional_data
            }
        else:
            out = {
                "train_curve": None,
                "val_curve": None,
                "test_curve": None,
                "times": times_as_dataframe.to_dict(orient="list"),
                "additional_data": self.additional_data
            }
        return out

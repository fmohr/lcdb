import pandas as pd

from lcdb.curve import Curve


class CurveDB:
    def __init__(
        self,
        train_curve: Curve,
        val_curve: Curve,
        test_curve: Curve,
        times,
        additional_data,
    ):
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

    def as_dict(self):
        """
        Compiles the whole knowledge base into a single compact dictionary.
        This dictionary should not have redundant information
        (like column names repeated over different list elements etc.).

        :return: dict
        """

        # format times as dataframe

        if len(self.train_curve) > 0:
            out = {
                "train_curve": self.train_curve.as_dict(),
                "val_curve": self.val_curve.as_dict(),
                "test_curve": self.test_curve.as_dict(),
                "times": self.times,
                "additional_data": self.additional_data,
            }
        else:
            out = {
                "train_curve": None,
                "val_curve": None,
                "test_curve": None,
                "times": self.times,
                "additional_data": self.additional_data,
            }
        return out

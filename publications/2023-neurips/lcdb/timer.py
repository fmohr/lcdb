from time import time
import numpy as np
import pandas as pd


class Timer:

    def __init__(self, aggregate: bool = True, precision=6):
        """

        :param aggregate: bool, if `True`, then times are aggregated for identical time names.
        Otherwise they are maintained in a list.
        :param precision: int. Number of digits recorded in measurement
        """
        self.ns_stack = []
        self.runtimes = {}
        self.cur_runtimes = self.runtimes  # this is a pointer to the dictionary within `runtimes` currently accessed through the namespace stack
        self.start_timestamps = {}
        self.aggregate = aggregate
        self.precision = precision

    def update_cur_runtimes_(self):
        self.cur_runtimes = self.runtimes
        for ns in self.ns_stack:
            self.cur_runtimes = self.cur_runtimes[ns]

    def enter(self, ns):
        ns = str(ns)
        self.ns_stack.append(ns)
        if ns not in self.cur_runtimes:
            self.cur_runtimes[ns] = {}
        self.update_cur_runtimes_()

    def leave(self):
        if not self.ns_stack:
            raise ValueError("The time is in no namespace ...")
        self.ns_stack.pop(-1)
        self.update_cur_runtimes_()

    def start(self, name):
        name = str(name)
        full_name = f"{self.ns_stack}_{name}" if self.ns_stack else name
        if full_name in self.start_timestamps:
            raise ValueError(f"Timer for {full_name} already started.")
        self.start_timestamps[full_name] = time()

    def stop(self, name):
        name = str(name)
        full_name = f"{self.ns_stack}_{name}" if self.ns_stack else name
        if full_name not in self.start_timestamps:
            raise ValueError(f"No timer started for {full_name}.")
        runtime = time() - self.start_timestamps[full_name]
        del self.start_timestamps[full_name]

        if self.aggregate:
            if name not in self.cur_runtimes:
                self.cur_runtimes[name] = 0
            self.cur_runtimes[name] = np.round(self.cur_runtimes[name] + runtime, self.precision)
        else:
            if name not in self.cur_runtimes:
                self.runtimes[name] = []
            self.cur_runtimes[name].append(np.round(runtime, self.precision))

    def get_runtime(self, name):

        """

        :param name: name of the time to be extracted
        :return: returns the accumulated or listed runtime of the given tag *within* the current namespace
        """
        self.cur_runtimes[name]


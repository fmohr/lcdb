from time import time


class Timer:

    def __init__(self, aggregate: bool = True):
        """

        :param aggregate: bool, if `True`, then times are aggregated for identical time names.
        Otherwise they are maintained in a list.
        """
        self.ns_stack = []
        self.runtimes = {}
        self.start_timestamps = {}
        self.aggregate = aggregate

    def enter(self, ns):
        self.ns_stack.append(ns)

    def leave(self):
        if not self.ns_stack:
            raise ValueError("The time is in no namespace ...")
        self.ns_stack.pop(-1)

    def start(self, name):
        full_name = f"{self.ns_stack[-1]}_{name}" if self.ns_stack else name
        if full_name in self.start_timestamps:
            raise ValueError(f"Timer for {full_name} already started.")
        self.start_timestamps[full_name] = time()

    def stop(self, name):
        full_name = f"{self.ns_stack[-1]}_{name}" if self.ns_stack else name
        if full_name not in self.start_timestamps:
            raise ValueError(f"No timer started for {full_name}.")
        runtime = time() - self.start_timestamps[full_name]
        del self.start_timestamps[full_name]

        if self.aggregate:
            if full_name not in self.runtimes:
                self.runtimes[full_name] = 0
            self.runtimes[full_name] += runtime
        else:
            if full_name not in self.runtimes:
                self.runtimes[full_name] = []
            self.runtimes[full_name].append(runtime)

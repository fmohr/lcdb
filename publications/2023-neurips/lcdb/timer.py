from time import time
import numpy as np
import pprint


class Timer:

    def __init__(self, precision=6):
        """

        :param aggregate: bool, if `True`, then times are aggregated for identical time names.
        Otherwise they are maintained in a list.
        :param precision: int. Number of digits recorded in measurement
        """
        self.root = None
        self.stack = []
        self.precision = precision

    def start(self, name, metadata=None):
        node = {
            "name": name,
            "start": np.round(time(), self.precision)
        }
        if isinstance(metadata, dict) and len(metadata) > 0:
            node["metadata"] = metadata
        if self.stack:
            parent = self.stack[-1]
            if "children" not in parent:
                parent["children"] = []
            parent["children"].append(node)
        else:
            self.root = node
        self.stack.append(node)

    def stop(self):
        """
        stops the current timer and steps back to the parent node

        :return: None
        """
        if not self.stack:
            raise ValueError("No timer currently active!")
        self.stack[-1]["stop"] = np.round(time(), self.precision)
        self.stack.pop(-1)

    def get_simplified_dict(self, multiple_occurrences="raise"):
        """
        Tries to use the names of the nods as keys in order to simplify the tree representation.
        This only works if the names of the children are unique

        :return: dict
        """

        def simplify_sub_tree(t):
            d = {
                    "start": t["start"],
                    "stop": t["stop"]
                }
            if "children" not in t:
                return d
            d["children"] = {}  # now a dictionary instead of a list
            for child in t["children"]:
                name = child["name"]
                if name in d["children"]:
                    if multiple_occurrences == "raise":
                        pprint.pprint(d)
                        raise ValueError(f"Duplicate entry for name {name} in node {t['name']}")
                    elif multiple_occurrences == "merge_and_drop":
                        additional_data = simplify_sub_tree(child)
                        if "children" in additional_data:
                            if "children" in d["children"][name]:
                                d["children"][name]["children"].update(additional_data["children"])
                            else:
                                d["children"][name]["children"] = additional_data["children"]

                        if "start" in d["children"][name]:
                            del d["children"][name]["start"]
                            del d["children"][name]["stop"]
                else:
                    d["children"][name] = simplify_sub_tree(child)
            return d

        return simplify_sub_tree(self.root)

    def get_simplified_stack(self):
        """
        Tries to use the names of the nods as keys in order to simplify the tree representation.
        This only works if the names of the children are unique

        :return: dict
        """

        def simplify_entry(queue):
            t = queue[0]
            d = {
                    "start": t["start"]
                }
            if "stop" in t:
                d["stop"] = t["stop"]
            if len(queue) > 1:
                queue.pop(0)
                name_of_next = queue[0]["name"]
                d[name_of_next] = simplify_entry(queue)
            return d

        return {self.stack[0]["name"]: simplify_entry(self.stack)}

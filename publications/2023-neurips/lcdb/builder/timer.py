import logging
import pprint
import time
from contextlib import contextmanager
from typing import Any, Hashable

import lcdb.json
import numpy as np


# !Private class to be used within the Timer class
class TimerNode:
    STARTED: str = "STARTED"
    STOPPED: str = "STOPPED"
    CANCELED: str = "CANCELED"

    def __init__(
        self, id_: int, tag: str, metadata: dict = None, precision: int = 6
    ) -> None:
        self.id = id_

        self.precision = precision

        self.tag = tag
        self.status = TimerNode.STARTED
        self.cancellation_source_id = None

        self.timestamp_start = self.time()
        self.timestamp_end = None

        assert metadata is None or isinstance(metadata, dict)
        self.metadata = {} if metadata is None else metadata

        self.children = []

    def time(self) -> float:
        # return np.round(time.time(), decimals=self.precision)
        return time.time()

    def stop(self, metadata=None):
        self.timestamp_end = self.time()
        if metadata is not None:
            self.metadata.update(metadata)
        self.status = TimerNode.STOPPED

    def cancel(self):
        self.timestamp_end = self.time()
        self.status = TimerNode.CANCELED

    def __getitem__(self, key):
        return self.metadata[key]

    def __setitem__(self, key, value):
        self.metadata[key] = value

    def as_dict(self, timestamp_offset: float = 0) -> dict:
        out = dict(
            # id=self.id,
            tag=self.tag,
            timestamp_start=np.round(
                self.timestamp_start - timestamp_offset, self.precision
            ),
            timestamp_stop=np.round(
                self.timestamp_end - timestamp_offset, self.precision
            ),
            # TODO: remove because redundant with timestamp_start/end
            # duration=np.round(
            #     self.timestamp_end - self.timestamp_start, self.precision
            # ),
            # status=self.status,
        )

        if len(self.metadata) > 0:
            out["metadata"] = self.metadata

        if len(self.children) > 0:
            out["children"] = [c.as_dict(timestamp_offset) for c in self.children]

        if self.cancellation_source_id:
            out["cancellation_source_id"] = self.cancellation_source_id

        return out

    def __repr__(self) -> str:
        return f"TimerNode(id={self.id}, tag={self.tag}, status={self.status})"


class Timer:
    """Class representing a timing profiler.

    Example use:

    >>> timer = Timer()
    >>> timer.start("program")
    >>> timer.start("function_call")
    >>> result = foo()
    >>> timer.stop()
    >>> timer.stop()
    >>> time.as_dict()

    Args:
        precision (int): Number of digits recorded in measurement.
    """

    def __init__(self, precision: int = 6):
        self.root = None
        self.stack = []
        self.precision = precision
        self.id_counter = 0

    def start(self, tag: Hashable, metadata: dict = None) -> int:
        """Start the timer for a new tag (i.e., creates a child node in the time tree).

        Args:
            tag (Hashable): tag of the node.
            metadata (dict, optional): optional metadata of the node. Defaults to ``None``.

        Returns:
            int: id of the created node in the timer tree.
        """
        logging.info(f"Starting timer for '{tag}'")

        node = TimerNode(self.id_counter, tag, metadata, self.precision)
        self.id_counter += 1

        if self.root is None:
            self.root = node
        else:
            parent = self.stack[-1]
            parent.children.append(node)

        self.stack.append(node)

        return node.id

    def stop(self, metadata: dict = None):
        """Stops the current timer and steps back to the parent node"""

        if len(self.stack) == 0:
            raise ValueError("No timer currently active!")

        node = self.stack.pop()
        node.stop(metadata)
        
        logging.info(f"Stopping timer for '{node.tag}'")

    def cancel(self, node_id: int, only_children: bool = False):
        """Cancels all child nodes up to the node corresponding to node_id (i.e., cancel all branches starting from `node_id`).

        Args:
            node_id (int): id of the node to cancel.
            only_children (bool, optional): if True, only the children of the node are canceled. Defaults to ``False``.
        """

        # Node must be in current set of active nodes
        if node_id not in [n.id for n in self.stack]:
            raise ValueError(
                f"The node timer with id '{node_id}' cannot be canceled because it is not currently active."
            )

        # Active node when cancel was requested
        source = self.active_node

        # Cancel nodes of the branch
        node = None
        while node is None or node.id != node_id:
            node = self.stack.pop()

            if node.id == node_id and only_children:
                self.stack.append(node)
                break

            node.cancel()

        # Record source of cancellation at root of cancelled branch
        node.cancellation_source_id = source.id

    @property
    def active_node(self):
        """The current active timer node."""
        if len(self.stack) == 0:
            raise ValueError("No timer currently active!")
        return self.stack[-1]

    def as_dict(self):
        return self.root.as_dict(timestamp_offset=self.root.timestamp_start)

    def as_json(self):
        # try:
        json_dumped = lcdb.json.dumps(self.as_dict())
        # except Exception as e:
        #     print(f'JSON dump failed')
        #     print(self.as_dict())
        # try:
        json_data = lcdb.json.loads(json_dumped)
        # except Exception as e:
        #     print(f"Json serialization failed!")
        #     print(json_dumped)
        return json_data

    @contextmanager
    def time(self, tag: Hashable, metadata: dict = None, cancel_on_error=False):
        node_id = self.start(tag, metadata)
        try:
            yield self.active_node
        except:
            if cancel_on_error:
                self.cancel(node_id)
            else:
                raise
        else:
            self.stop()


def test_timer():
    delay = lambda: time.sleep(0.5)

    # Creating the timer
    timer = Timer()

    # Timer can be used explicitly through the start/stop methods
    run_timer_id = timer.start("run")

    # Timer can be used through context manager (recommended)
    with timer.time("load_data"):
        delay()

    # Timer can catch and cancel branch if error is raised
    with timer.time("fit", cancel_on_error=True) as fit_timer:
        # Simulation of a training loop
        for epoch_i in range(10):
            with timer.time("epoch", {"i": epoch_i}) as epoch_timer:
                delay()

                if epoch_i > 2:
                    # Possible failure for whichever reason
                    raise RuntimeError

                # We can record information for the current node
                # For the time to be accurate it must be done before at the end of the block
                epoch_timer["accuracy"] = 0.5

    # Here the current active node should me the "run"
    assert timer.active_node.id == run_timer_id

    with timer.time("predict") as predict_timer:
        delay()

        # Here is how I can record my metrics through the timer
        predict_timer["train"] = {"accuracy": 1.0}

    timer.stop()

    assert len(timer.stack) == 0

    dict_tree = timer.as_dict()
    assert all(
        k in dict_tree
        for k in [
            "id",
            "tag",
            "timestamp_start",
            "timestamp_stop",
            "duration",
            "status",
            "metadata",
            "children",
        ]
    )
    assert dict_tree["tag"] == "run"
    assert dict_tree["id"] == 0
    assert len(dict_tree["children"]) == 3

    pprint.pprint(timer.as_dict(), indent=2)


if __name__ == "__main__":
    test_timer()

import abc
from typing import List, Any

import jmespath


class JsonQuery(abc.ABC):

    @abc.abstractmethod
    def apply(self, x: dict) -> Any:
        """Query data from a dictionnary."""

    def __call__(self, x) -> Any:
        return self.apply(x)


class FullQuery(JsonQuery):
    def apply(self, x: dict):
        return x
    
class NoneQuery(JsonQuery):
    def apply(self, x: dict):
        return None


class JMESExpressionQuery(JsonQuery):
    def __init__(self, expression: str) -> None:
        self.expression = jmespath.compile(expression)

    def apply(self, x: dict):
        """Extract a metric from a dict (json-like) object."""
        out = self.expression.search(x)
        return out

    def __call__(self, x) -> Any:
        return self.apply(x)


class QueryAnchorValues(JMESExpressionQuery):
    """Extract the anchor values."""

    def __init__(self):
        super().__init__(
            f"children[?tag == 'build_curves'] | [0]"
            f".children[?tag == 'anchor'] | []"
            f".metadata.value"
        )


class QueryAnchorKeys(JMESExpressionQuery):
    """Extract the key values for all anchors."""

    def __init__(self, key):
        super().__init__(
            f"children[?tag == 'build_curves'] | [0]"
            f".children[?tag == 'anchor'] | []"
            f".{key}"
        )


class QueryAnchorsChildren(JMESExpressionQuery):
    """Extract children data from all anchors."""

    def __init__(self):
        super().__init__(
            f"children[?tag == 'build_curves'] | [0]"
            f".children[?tag == 'anchor'] | [*]"
            f".children"
        )


class QueryMetricValuesFromAnchors(JMESExpressionQuery):
    """Extract a metric for all anchors."""

    def __init__(self, metric_name: str, split_name: str = "val"):
        super().__init__(
            f"children[? tag == 'build_curves'] | [0]"  # here we replace sublists by the first element
            f".children[? tag == 'anchor'] | []"  # here we keep all elements of the list
            f".children[? tag == 'metrics'] | []"
            f".children[? tag == '{split_name}'] | []"
            f".children[? tag == '{metric_name}'] | []"
            f".metadata.value"
        )


class QueryEpochValues(JMESExpressionQuery):
    """Extract the epoch values for all anchors."""

    def __init__(self, with_epoch_test: bool = True):
        if with_epoch_test:
            super().__init__(
                f"children[? tag == 'build_curves'] | [0]"
                f".children[? tag == 'anchor'] | [*]"
                f".children[? tag == 'fit'] | [*][0]"
                f".children[? tag == 'epoch'] | [*][?(children[?(tag == 'epoch_test')])]"
                f".metadata.value"
            )
        else:
            super().__init__(
                f"children[? tag == 'build_curves'] | [0]"
                f".children[? tag == 'anchor'] | [*]"
                f".children[? tag == 'fit'] | [*][0]"
                f".children[? tag == 'epoch'] | [*][*]"
                f".metadata.value"
            )


class QueryMetricValuesFromEpochs(JMESExpressionQuery):
    """Extract the anchor children."""

    def __init__(self, metric_name: str, split_name: str = "val"):
        super().__init__(
            f"children[? tag == 'build_curves'] | [0]"
            f".children[? tag == 'anchor'] | [*]"
            f".children[? tag == 'fit'] | [*][0]"
            f".children[? tag == 'epoch'] | [*][*]"
            f".children[? tag == 'epoch_test'] | [*][*][0]"
            f".children[? tag == 'metrics'] | [*][*][0]"
            f".children[? tag == '{split_name}'] | [*][*][0]"
            f".children[? tag == '{metric_name}'] | [*][*][0]"
            f".metadata.value"
        )


class QueryPreprocessorResults(JMESExpressionQuery):

    def __init__(self):
        super().__init__ (
            f"children[? tag == 'build_curves'] | [0]"  # here we replace sublists by the first element
            f".children[? tag == 'anchor'] | [*]"  # here we keep all elements of the list
            f".children[? tag == 'fit'] | [*][0]"
            f".children[? tag == 'transform_train'] | [*][0]"
            f".children | [*][*]"
        )

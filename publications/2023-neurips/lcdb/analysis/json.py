from typing import List, Any

import jmespath


class JsonQuery:
    def __init__(self, expression: str) -> None:
        self.expression = jmespath.compile(expression)

    def apply(self, x: dict):
        """Extract a metric from a dict (json-like) object."""
        out = self.expression.search(x)
        return out

    def __call__(self, x) -> Any:
        return self.apply(x)


class QueryAnchorValues(JsonQuery):
    def __init__(self):
        super().__init__(
            f"children[?tag == 'build_curves'] | [0]"
            f".children[?tag == 'anchor'] | []"
            f".metadata.value"
        )


class QueryMetricValues(JsonQuery):
    def __init__(self, metric_name: str, split_name: str = "val"):
        super().__init__(
            f"children[? tag == 'build_curves'] | [0]"  # here we replace sublists by the first element
            f".children[? tag == 'anchor'] | []"  # here we keep all elements of the list
            f".children[? tag == 'metrics'] | []"
            f".children[? tag == '{split_name}'] | []"
            f".children[? tag == '{metric_name}'] | []"
            f".metadata.value"
        )

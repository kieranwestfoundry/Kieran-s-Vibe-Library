import json
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterTypeBuiltin
from griptape_nodes_library.lists.base_create_list import BaseCreateListNode


class CreateJson(BaseCreateListNode):
    """CreateJson Node that creates a JSON array with items provided."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(
            name,
            metadata,
            input_types=[ParameterTypeBuiltin.ANY.value],
            output_type="json",
            default_value=None,
            items_tooltip="List of items to add to the JSON array",
        )

    def process(self) -> None:
        """Run the base aggregation and serialize the result to JSON."""
        # Execute the base class logic, which typically gathers inputs into a list
        # and assigns them to self.parameter_output_values["output"]
        super().process()

        # Intercept the output and convert it to a JSON formatted string
        if "output" in self.parameter_output_values:
            items = self.parameter_output_values["output"]
            try:
                self.parameter_output_values["output"] = json.dumps(items)
            except TypeError as e:
                # Setting safe default before raising exception per production best practices
                self.parameter_output_values["output"] = "[]"
                raise ValueError(f"{self.name}: Failed to serialize items to JSON: {str(e)}") from e
import random
from typing import Any

import requests

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.traits.options import Options

class RandomNumberGenerator(ControlNode):
    """Generates a random integer using pseudo-random or quantum methods."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Generation Method Selector
        method_param = Parameter(
            name="method",
            input_types=[ParameterTypeBuiltin.STR.value],
            type=ParameterTypeBuiltin.STR.value,
            default_value="Pseudo-random",
            tooltip="Method for generating the random number. Pseudo-random uses local compute, True Random calls the ANU Quantum API.",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Generation Method"}
        )
        method_param.add_trait(Options(choices=["Pseudo-random", "True Random (ANU Quantum)"]))
        self.add_parameter(method_param)

        # Minimum Value
        self.add_parameter(Parameter(
            name="min_value",
            input_types=[ParameterTypeBuiltin.INT.value],
            type=ParameterTypeBuiltin.INT.value,
            default_value=0,
            tooltip="Minimum value (inclusive)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Minimum Value"}
        ))

        # Maximum Value
        self.add_parameter(Parameter(
            name="max_value",
            input_types=[ParameterTypeBuiltin.INT.value],
            type=ParameterTypeBuiltin.INT.value,
            default_value=100,
            tooltip="Maximum value (inclusive)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Maximum Value"}
        ))

        # Output Random Number
        self.add_parameter(Parameter(
            name="random_number",
            output_type=ParameterTypeBuiltin.INT.value,
            tooltip="The generated random integer",
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "Random Number"}
        ))

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before running the node."""
        exceptions = []
        min_val = self.get_parameter_value("min_value")
        max_val = self.get_parameter_value("max_value")

        if min_val is not None and max_val is not None:
            if min_val > max_val:
                exceptions.append(
                    ValueError(f"{self.name}: Minimum value ({min_val}) cannot be greater than maximum value ({max_val}).")
                )

        return exceptions if exceptions else None

    def process(self) -> AsyncResult:
        """Process the request asynchronously."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Main processing method."""
        # Set a safe default before processing
        self.parameter_output_values["random_number"] = None

        method = self.get_parameter_value("method")
        min_val = self.get_parameter_value("min_value")
        max_val = self.get_parameter_value("max_value")

        # Fallbacks in case inputs are missing
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = 100

        try:
            if method == "True Random (ANU Quantum)":
                url = "https://qrng.anu.edu.au/API/jsonI.php"
                params = {
                    "length": 1,
                    "type": "uint16"  # Generates an integer between 0 and 65535
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get("success"):
                    q_num = data["data"][0]
                    # Map the raw uint16 value to the user's requested range
                    range_size = max_val - min_val + 1
                    result = min_val + (q_num % range_size)
                else:
                    raise RuntimeError("ANU Quantum API returned an unsuccessful status.")
            else:
                # Standard Python PRNG
                result = random.randint(min_val, max_val)

            self.parameter_output_values["random_number"] = result

        except Exception as e:
            # Safely handle the exception and pass it to the engine
            self.parameter_output_values["random_number"] = None
            raise RuntimeError(f"Failed to generate random number: {e}") from e
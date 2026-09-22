from __future__ import annotations

import random
import hashlib
import time
from datetime import datetime
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import DataNode

class MultiOutputUtilityNode(DataNode):
    """
    A utility node that generates a random number, hashes an input string, 
    and provides current time in both Unix and ISO formats.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        # Initialize the base DataNode class
        super().__init__(name, metadata)

        # 1. Input: String to be hashed (Input + Property)
        self.add_parameter(
            Parameter(
                name="input_text",
                tooltip="The text string to hash",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="Griptape",
            )
        )

        # 2. Output: Random Number
        self.add_parameter(
            Parameter(
                name="random_number",
                tooltip="A random float between 0 and 100",
                type=ParameterTypeBuiltin.FLOAT.value,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # 3. Output: Hashed String
        self.add_parameter(
            Parameter(
                name="hashed_string",
                tooltip="SHA-256 hash of the input text",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # 4. Output: Unix Timestamp
        self.add_parameter(
            Parameter(
                name="unix_timestamp",
                tooltip="Current Unix timestamp",
                type=ParameterTypeBuiltin.FLOAT.value,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # 5. Output: Formatted Datetime
        self.add_parameter(
            Parameter(
                name="formatted_datetime",
                tooltip="Current date and time in ISO format",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        """
        Logic to generate values and populate output parameters.
        """
        # Retrieve the input text
        text_to_hash = self.get_parameter_value("input_text") or ""

        # 1. Generate Random Number
        random_val = random.uniform(0, 100)

        # 2. Generate Hash (SHA-256)
        hash_object = hashlib.sha256(text_to_hash.encode())
        hex_dig = hash_object.hexdigest()

        # 3. Get Unix Time
        u_time = time.time()

        # 4. Get Current Datetime
        dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Assign values to output parameters
        self.parameter_output_values["random_number"] = round(random_val, 4)
        self.parameter_output_values["hashed_string"] = hex_dig
        self.parameter_output_values["unix_timestamp"] = round(u_time, 4)
        self.parameter_output_values["formatted_datetime"] = dt_string

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """
        Surface errors if required inputs are missing.
        """
        errors = []
        if not self.get_parameter_value("input_text"):
            errors.append(ValueError(f"Node '{self.name}' requires 'input_text' to generate a hash."))
        
        return errors if errors else None
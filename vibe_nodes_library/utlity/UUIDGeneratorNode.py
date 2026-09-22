from __future__ import annotations

from typing import Any
import uuid

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.traits.options import Options


class UUIDGeneratorNode(DataNode):
    """A Griptape node designed to handle unique ID generation dynamically.

    Supports adding custom prefixes and choosing between different standard
    UUID versions to easily adapt to downstream schema requirements.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "utils"
        self.description = "Generates universally unique identifiers (UUIDs)."

        # Optional Prefix Parameter (Input + UI Editable Property)
        self.add_parameter(
            Parameter(
                name="prefix",
                tooltip="An optional text prefix to prepend to the generated UUID (e.g., 'usr_').",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                ui_options={
                    "display_name": "ID Prefix",
                    "placeholder_text": "e.g., node_ or txn_",
                },
            )
        )

        # UUID Version Parameter Selector (Input + Dropdown UI Property)
        self.add_parameter(
            Parameter(
                name="uuid_version",
                tooltip="Select UUID v4 for random generation or v1 for time-based tracking.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["v4 (Random)", "v1 (Time-based)"])},
                default_value="v4 (Random)",
                ui_options={
                    "display_name": "UUID Version",
                },
            )
        )

        # Output Parameter (Output-only, read by downstream nodes)
        self.add_parameter(
            Parameter(
                name="unique_id",
                tooltip="The generated universally unique identifier string.",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={
                    "display_name": "Generated Unique ID",
                },
            )
        )

    def process(self) -> None:
        """Executes the unique ID generation logic during workflow execution."""
        prefix = self.get_parameter_value("prefix") or ""
        version_selection = self.get_parameter_value("uuid_version") or "v4 (Random)"

        try:
            if "v1" in version_selection:
                generated_uuid = uuid.uuid1()
            else:
                generated_uuid = uuid.uuid4()

            # Synthesize the final string identifier
            final_id = f"{prefix}{str(generated_uuid)}"

            # Map the value to the engine output state
            self.parameter_output_values["unique_id"] = final_id

        except Exception as e:
            # Clear output parameter on unexpected failures to avoid stale data
            self.parameter_output_values["unique_id"] = None
            raise RuntimeError(f"UUID Generator Node '{self.name}' failed: {str(e)}") from e

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates configuration parameters prior to workflow orchestration."""
        errors = []
        version_selection = self.get_parameter_value("uuid_version")

        if version_selection not in ["v4 (Random)", "v1 (Time-based)"]:
            errors.append(
                ValueError(
                    f"Node '{self.name}' has an invalid UUID version configuration: {version_selection}"
                )
            )

        return errors if errors else None

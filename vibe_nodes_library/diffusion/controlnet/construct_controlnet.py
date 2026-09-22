import logging
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.traits.file_system_picker import FileSystemPicker

logger = logging.getLogger(__name__)


class ConstructControlNetNode(ControlNode):
    """Configures a ControlNet model to be injected into a Pipeline."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="controlnet_model_path",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                tooltip="Absolute path to a local ControlNet folder/file, or a HuggingFace ID.",
                ui_options={"display_name": "ControlNet Path / ID"},
                traits={FileSystemPicker()},
            )
        )

        self.add_parameter(
            Parameter(
                name="controlnet_config",
                output_type="dict",
                tooltip="The ControlNet configuration dictionary.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        if not self.get_parameter_value("controlnet_model_path"):
            errors.append(
                ValueError(
                    f"{self.name}: 'controlnet_model_path' parameter is required."
                )
            )
        return errors if errors else None

    def process(self) -> AsyncResult:
        def work() -> Any:
            controlnet_path = self.get_parameter_value("controlnet_model_path")

            # Pass a lightweight dictionary downstream instead of a massive PyTorch tensor.
            # This prevents Griptape's model_cache from prematurely evicting the base pipeline.
            config = {"type": "controlnet", "path": controlnet_path}

            self.parameter_output_values["controlnet_config"] = config
            return config

        yield work

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterList,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

# Module-level imports per best practices
try:
    import torch  # type: ignore[import-untyped]
    from diffusers import (  # type: ignore[import-untyped]
        ControlNetModel,
        StableDiffusionControlNetPipeline,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


class StableDiffusionControlNetBuilderNode(ControlNode):
    """Builds a Stable Diffusion pipeline with support for one or more ControlNets."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.log_params = LogParameter(self)

        self.add_parameter(
            Parameter(
                name="base_model",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="runwayml/stable-diffusion-v1-5",
                tooltip="HuggingFace model ID or local path for the base Stable Diffusion model.",
                ui_options={"display_name": "Base SD Model ID"},
            )
        )

        # Uses ParameterList to gracefully accept multiple incoming ControlNet nodes
        self.add_parameter(
            ParameterList(
                name="controlnets",
                input_types=["ControlNetModel", "list[ControlNetModel]", "str"],
                default_value=[],
                tooltip="Connect one or more constructed ControlNet models.",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "ControlNet Models", "expander": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="pipeline",
                output_type="StableDiffusionControlNetPipeline",
                tooltip="The constructed Stable Diffusion ControlNet Pipeline.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.log_params.add_output_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        if not self.get_parameter_value("base_model"):
            errors.append(
                ValueError(f"{self.name}: 'base_model' parameter is required.")
            )
        return errors if errors else None

    def process(self) -> AsyncResult:
        self.log_params.clear_logs()
        self.log_params.append_to_logs(
            "Starting Stable Diffusion ControlNet Pipeline build...\n"
        )

        def work() -> Any:
            try:
                base_model = self.get_parameter_value("base_model")
                controlnets_input = self.get_parameter_list_value("controlnets")

                # Filter out any empty/falsey list items caused by empty connections
                valid_controlnets = [cn for cn in controlnets_input if cn]

                if not valid_controlnets:
                    self.log_params.append_to_logs(
                        "Warning: No ControlNet models provided.\n"
                    )
                    raise ValueError(
                        f"{self.name}: At least one ControlNet model is required."
                    )

                loaded_controlnets = []
                for cn in valid_controlnets:
                    if isinstance(cn, str):
                        self.log_params.append_to_logs(
                            f"Loading ControlNet from string ID: {cn}\n"
                        )
                        loaded_cn = ControlNetModel.from_pretrained(
                            cn, torch_dtype=torch.float16
                        )
                        loaded_controlnets.append(loaded_cn)
                    else:
                        # Assumes the item is a constructed ControlNetModel
                        loaded_controlnets.append(cn)

                # Diffusers accepts a single object for one ControlNet, or a list for Multi-ControlNet
                controlnet_arg = (
                    loaded_controlnets[0]
                    if len(loaded_controlnets) == 1
                    else loaded_controlnets
                )

                self.log_params.append_to_logs(
                    f"Loading base model {base_model} with ControlNet(s)...\n"
                )

                pipe = StableDiffusionControlNetPipeline.from_pretrained(
                    base_model, controlnet=controlnet_arg, torch_dtype=torch.float16
                )

                self.log_params.append_to_logs("Pipeline built successfully.\n")
                self.parameter_output_values["pipeline"] = pipe
                return pipe

            except Exception as e:
                logger.exception("%s: Pipeline build failed", self.name)
                self.log_params.append_to_logs(f"Error building pipeline: {str(e)}\n")
                # Clear output on error to avoid stale data
                self.parameter_output_values["pipeline"] = None
                raise RuntimeError(f"Pipeline build failed: {e}") from e

        yield work

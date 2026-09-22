import hashlib
import logging
import os
from typing import Any

from diffusers_nodes_library.common.utils.huggingface_utils import model_cache
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

try:
    import torch  # type: ignore[import-untyped]
except ImportError:
    pass

logger = logging.getLogger(__name__)


class InjectSDControlNetNode(ControlNode):
    """
    Middleware node that dynamically loads and injects ControlNet models
    into an existing Stable Diffusion/Flux Pipeline config before execution.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.log_params = LogParameter(self)

        self.add_parameter(
            Parameter(
                name="input_pipeline",
                type="Pipeline Config",
                input_types=["Pipeline Config", "str", "any"],
                allowed_modes={ParameterMode.INPUT},
                tooltip="Connect the 'pipeline' output from a Diffusion Pipeline Builder here.",
            )
        )

        self.add_parameter(
            ParameterList(
                name="controlnets",
                input_types=["dict", "list[dict]", "any"],
                default_value=[],
                allowed_modes={ParameterMode.INPUT},
                tooltip="Connect one or more Construct ControlNet nodes.",
                ui_options={"display_name": "ControlNet Models", "expander": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="pipeline",
                output_type="Pipeline Config",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The modified pipeline config ready for the Generate Image node.",
            )
        )

        self.log_params.add_output_parameters()

    def process(self) -> AsyncResult:
        self.log_params.clear_logs()

        def work() -> Any:
            input_hash = self.get_parameter_value("input_pipeline")
            controlnets_input = self.get_parameter_list_value("controlnets")

            if not input_hash:
                raise ValueError(f"{self.name}: Input pipeline connection is missing.")

            # Extract valid configs
            valid_cns = [
                cn for cn in controlnets_input if isinstance(cn, dict) and "path" in cn
            ]

            if not valid_cns:
                self.log_params.append_to_logs(
                    "No valid ControlNets provided. Passing original pipeline through.\n"
                )
                self.parameter_output_values["pipeline"] = input_hash
                return input_hash

            # Create a unique hash for this modified pipeline state based on the CN paths
            cn_paths = str([cn["path"] for cn in valid_cns])
            unique_suffix = hashlib.md5(cn_paths.encode()).hexdigest()[:8]
            new_hash = f"{input_hash}-injected-cn-{unique_suffix}"

            def build_injected_pipeline() -> Any:
                self.log_params.append_to_logs(
                    "Retrieving base pipeline from memory cache...\n"
                )
                base_pipe = model_cache.get_pipeline(input_hash)

                if not base_pipe:
                    raise RuntimeError(
                        "Base pipeline not found in cache. Ensure the upstream Builder node ran successfully."
                    )

                base_class_name = type(base_pipe).__name__
                self.log_params.append_to_logs(
                    f"Detected Base Pipeline Type: {base_class_name}\n"
                )

                # Dynamically resolve the correct Diffusers classes to prevent weight mismatch warnings
                if "StableDiffusion3" in base_class_name:
                    from diffusers import (
                        SD3ControlNetModel,
                        StableDiffusion3ControlNetPipeline,
                    )

                    PipelineClass = StableDiffusion3ControlNetPipeline
                    ControlNetClass = SD3ControlNetModel
                elif "Flux" in base_class_name:
                    from diffusers import FluxControlNetModel, FluxControlNetPipeline

                    PipelineClass = FluxControlNetPipeline
                    ControlNetClass = FluxControlNetModel
                elif "StableDiffusionXL" in base_class_name:
                    from diffusers import (
                        ControlNetModel,
                        StableDiffusionXLControlNetPipeline,
                    )

                    PipelineClass = StableDiffusionXLControlNetPipeline
                    ControlNetClass = ControlNetModel
                else:
                    from diffusers import (
                        ControlNetModel,
                        StableDiffusionControlNetPipeline,
                    )

                    PipelineClass = StableDiffusionControlNetPipeline
                    ControlNetClass = ControlNetModel

                self.log_params.append_to_logs(
                    f"Injecting {len(valid_cns)} ControlNet(s)...\n"
                )

                loaded_controlnets = []
                for cn_config in valid_cns:
                    path = cn_config["path"]
                    self.log_params.append_to_logs(f"Loading {path}...\n")

                    if os.path.isfile(path) and path.lower().endswith(
                        (".safetensors", ".bin", ".ckpt")
                    ):
                        cn_model = ControlNetClass.from_single_file(
                            path, torch_dtype=torch.float16
                        )
                    else:
                        cn_model = ControlNetClass.from_pretrained(
                            path, torch_dtype=torch.float16
                        )

                    loaded_controlnets.append(cn_model)

                # Diffusers takes a single instance or a list for Multi-ControlNet
                controlnet_arg = (
                    loaded_controlnets[0]
                    if len(loaded_controlnets) == 1
                    else loaded_controlnets
                )

                # Extract components and rebuild
                components = base_pipe.components
                new_pipe = PipelineClass(**components, controlnet=controlnet_arg)

                return new_pipe

            # Safely build and register the new pipeline
            with self.log_params.append_profile_to_logs("Pipeline Injection"):
                model_cache.get_or_build_pipeline(new_hash, build_injected_pipeline)

            self.log_params.append_to_logs("ControlNet injection complete.\n")
            self.parameter_output_values["pipeline"] = new_hash
            return new_hash

        yield work

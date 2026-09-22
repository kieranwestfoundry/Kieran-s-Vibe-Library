import base64
import time
import logging
from typing import Any

import requests

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import ControlNode, DataNode, AsyncResult
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from diffusers_nodes_library.common.parameters.file_path_parameter import FilePathParameter

logger = logging.getLogger("multipass_render_nodes")

class BaseModelConfig(DataNode):
    """Selects the base diffusion model for the generation pass."""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.add_parameter(
            Parameter(
                name="model_id",
                tooltip="The HuggingFace ID or local path to the base model",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value="runwayml/stable-diffusion-v1-5",
            )
        )
        
        self.add_parameter(
            Parameter(
                name="model_config",
                tooltip="Connect to the Render Node's base model input",
                type="dict",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        self.parameter_output_values["model_config"] = {
            "model_id": self.get_parameter_value("model_id"),
        }


class LoraConfig(DataNode):
    """Outputs a generic dictionary config for a LoRA."""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.lora_file_path_params = FilePathParameter(
            self,
            file_types=[".safetensors", ".sft", ".pt", ".bin", ".json", ".lora"],
            tooltip="Absolute path to a local LoRA file",
        )
        self.lora_file_path_params.add_input_parameters()
        
        self.add_parameter(
            Parameter(
                name="weight",
                tooltip="Influence of the LoRA",
                type=ParameterTypeBuiltin.FLOAT.value,
                traits={Slider(min_val=-2.0, max_val=2.0)},
                ui_options={"step": 0.05},
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value=1.0,
            )
        )
        
        self.add_parameter(
            Parameter(
                name="lora_config",
                tooltip="Connect this output to the Render Node's LoRA list",
                type="dict",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        self.lora_file_path_params.validate_parameter_values()
        lora_path = str(self.lora_file_path_params.get_file_path())
        
        self.parameter_output_values["lora_config"] = {
            "path": lora_path,
            "weight": self.get_parameter_value("weight")
        }


class ControlNetConfig(DataNode):
    """Configures a ControlNet model and links it to a specific AOV pass."""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.add_parameter(
            Parameter(
                name="aov_target",
                tooltip="The AOV pass this ControlNet applies to",
                type=ParameterTypeBuiltin.STR.value,
                traits={Options(choices=["rgb_beauty", "normal", "world_space_normal", "position", "depth", "mask", "other"])},
                allowed_modes={ParameterMode.PROPERTY},
                default_value="normal",
            )
        )
        
        self.add_parameter(
            Parameter(
                name="controlnet_id",
                tooltip="HuggingFace ID or local path for the ControlNet",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value="lllyasviel/sd-controlnet-normal",
            )
        )
        
        self.add_parameter(
            Parameter(
                name="conditioning_scale",
                tooltip="Strength of the ControlNet influence",
                type=ParameterTypeBuiltin.FLOAT.value,
                traits={Slider(min_val=0.0, max_val=2.0)},
                ui_options={"step": 0.05},
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value=1.0,
            )
        )
        
        self.add_parameter(
            Parameter(
                name="controlnet_config",
                tooltip="Connect this output to the Render Node's ControlNet list",
                type="dict",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        self.parameter_output_values["controlnet_config"] = {
            "aov_target": self.get_parameter_value("aov_target"),
            "model_id": self.get_parameter_value("controlnet_id"),
            "conditioning_scale": self.get_parameter_value("conditioning_scale")
        }


class MultiPassRenderNode(ControlNode):
    """
    Ingests multiple render passes (AOVs) and configurations, sending them to an AI
    generation endpoint, saving the output deterministically.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Base Configuration
        self.add_parameter(
            Parameter(
                name="base_model",
                tooltip="Base model configuration dictionary",
                type="dict",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type=ParameterTypeBuiltin.STR.value,
                tooltip="Describe the desired generation output",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": True},
            )
        )

        # Dynamic Configurations
        self.add_parameter(ParameterList(
            name="loras",
            input_types=["dict", "list[dict]"],
            default_value=[],
            tooltip="Connect one or multiple LoraConfig nodes here",
            allowed_modes={ParameterMode.INPUT},
        ))

        self.add_parameter(ParameterList(
            name="controlnets",
            input_types=["dict", "list[dict]"],
            default_value=[],
            tooltip="Connect one or multiple ControlNetConfig nodes here",
            allowed_modes={ParameterMode.INPUT},
        ))

        # AOV Image Inputs
        for aov in ["rgb_beauty", "normal", "world_space_normal", "position", "depth", "mask", "other"]:
            self.add_parameter(
                ParameterImage(
                    name=f"aov_{aov}",
                    tooltip=f"{aov.replace('_', ' ').title()} Pass",
                    allow_output=False,
                )
            )

        # Output Parameters
        self.add_parameter(
            ParameterImage(
                name="generated_output",
                tooltip="The final AI generated render",
                allow_input=False,
                allow_property=False,
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="generated_frame.png",
        )
        self._output_file.add_parameter()

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generated_output"] = None

    def _get_image_data(self, param_name: str) -> str | None:
        """Safely extracts and converts image artifacts to base64 data URIs."""
        image_artifact = self.get_parameter_value(param_name)
        if not image_artifact:
            return None

        try:
            if isinstance(image_artifact, ImageUrlArtifact):
                url = image_artifact.value
                if url.startswith(('http://localhost', 'http://127.0.0.1')):
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    image_bytes = response.content
                    mime_type = response.headers.get('content-type', 'image/png')
                else:
                    return url
            elif isinstance(image_artifact, ImageArtifact):
                if hasattr(image_artifact, 'base64'):
                    return f"data:{getattr(image_artifact, 'mime_type', 'image/png')};base64,{image_artifact.base64}"
                else:
                    image_bytes = image_artifact.value
                    mime_type = "image/png"
            else:
                return None

            base64_data = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:{mime_type};base64,{base64_data}"
            
        except Exception as e:
            logger.error("Failed to process %s: %s", param_name, e)
            return None

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> None:
        self._set_safe_defaults()

        payload = {
            "base_model": self.get_parameter_value("base_model") or {},
            "prompt": self.get_parameter_value("prompt") or "",
            "loras": self.get_parameter_list_value("loras") or [],
            "controlnets": self.get_parameter_list_value("controlnets") or [],
            "passes": {}
        }

        # Dynamically map provided AOVs to the payload
        for aov in ["rgb_beauty", "normal", "world_space_normal", "position", "depth", "mask", "other"]:
            param_key = f"aov_{aov}"
            img_data = self._get_image_data(param_key)
            if img_data:
                payload["passes"][aov] = img_data

        if not payload["passes"]:
            raise ValueError(f"{self.name}: At least one render pass must be provided.")

        try:
            # ---------------------------------------------------------
            # MOCK API CALL: Replace with your chosen diffusers backend
            # ---------------------------------------------------------
            logger.info("Executing Multi-Pass Render with payload structure: %s", list(payload.keys()))
            time.sleep(3) 
            mock_output_bytes = b"MOCK_IMAGE_DATA_REPLACE_ME" 
            # ---------------------------------------------------------

            dest = self._output_file.build_file()
            saved = dest.write_bytes(mock_output_bytes)
            self.parameter_output_values["generated_output"] = ImageUrlArtifact(saved.location)
            
        except Exception as e:
            logger.error("Rendering failed: %s", e)
            raise RuntimeError(f"{self.name}: {str(e)}") from e
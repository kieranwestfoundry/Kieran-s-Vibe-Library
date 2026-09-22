import base64
import io
import logging
from typing import Any

import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter

logger = logging.getLogger("multipass_render_nodes")

class MultiPassRenderNode(ControlNode):
    """
    Executes a multi-ControlNet Stable Diffusion 1.5 pipeline locally using Diffusers,
    driven by Nuke AOV passes. Incorporates advanced model loading configs and fast-fail validation.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="base_model",
                tooltip="Base model config dict (e.g., runwayml/stable-diffusion-v1-5)",
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

        self.add_parameter(ParameterList(
            name="controlnets",
            input_types=["dict", "list[dict]"],
            default_value=[],
            tooltip="Connect ControlNetConfig nodes here",
            allowed_modes={ParameterMode.INPUT},
        ))

        # AOV Inputs
        for aov in ["normal", "depth", "rgb_beauty", "position", "mask", "other"]:
            self.add_parameter(
                ParameterImage(
                    name=f"aov_{aov}",
                    tooltip=f"{aov.title()} Pass",
                    allow_output=False,
                )
            )

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

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before running the node to fail fast."""
        exceptions = []
        cnet_configs = self.get_parameter_list_value("controlnets") or []

        for i, config in enumerate(cnet_configs):
            if not isinstance(config, dict):
                continue
            
            aov_target = config.get("aov_target")
            if not aov_target:
                exceptions.append(ValueError(f"{self.name}: ControlNet at index {i} is missing an 'aov_target'."))
                continue

            param_name = f"aov_{aov_target}"
            
            # Check if the required AOV parameter actually has data
            if not self.get_parameter_value(param_name):
                exceptions.append(ValueError(
                    f"{self.name}: ControlNet requires AOV pass '{aov_target}', but no image is connected or provided."
                ))

        return exceptions if exceptions else None

    def _get_pil_image(self, param_name: str) -> Image.Image | None:
        """Extracts Griptape ImageArtifacts and converts them directly to PIL Images."""
        image_artifact = self.get_parameter_value(param_name)
        if not image_artifact:
            return None

        try:
            if isinstance(image_artifact, ImageArtifact):
                image_bytes = image_artifact.value
            elif isinstance(image_artifact, ImageUrlArtifact) and hasattr(image_artifact, 'get_bytes'):
                image_bytes = image_artifact.get_bytes() 
            else:
                return None
            
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error("Failed to load image for %s: %s", param_name, e)
            return None

    def process(self) -> AsyncResult | None:
        """Yield to a background thread to prevent UI blocking."""
        yield lambda: self._process()

    def _process(self) -> None:
        # Clear output to prevent stale data on failures
        self.parameter_output_values["generated_output"] = None

        prompt = self.get_parameter_value("prompt") or ""
        base_model_dict = self.get_parameter_value("base_model") or {}
        base_model_id = base_model_dict.get("model_id", "runwayml/stable-diffusion-v1-5")
        
        cnet_configs = self.get_parameter_list_value("controlnets") or []
        if not cnet_configs:
            raise ValueError(f"{self.name}: No ControlNets connected.")

        # 1. Setup device and precision
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Translate the string precision from the config to actual torch types
        dtype_str = base_model_dict.get("torch_dtype", "auto")
        if dtype_str == "float16":
            torch_dtype = torch.float16
        elif dtype_str == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype_str == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if device != "cpu" else torch.float32

        try:
            logger.info("Loading %s ControlNets...", len(cnet_configs))
            controlnet_models = []
            controlnet_images = []
            controlnet_scales = []

            # 2. Load ControlNet models with network configurations
            for config in cnet_configs:
                aov_target = config.get("aov_target")
                model_id = config.get("model_id")
                scale = config.get("conditioning_scale", 1.0)
                allow_download = config.get("allow_download", False)

                pil_img = self._get_pil_image(f"aov_{aov_target}")
                if pil_img is None:
                    continue # Handled by validate_before_node_run, but kept for safety

                logger.info("Loading ControlNet: %s (Download allowed: %s)", model_id, allow_download)
                cnet = ControlNetModel.from_pretrained(
                    model_id, 
                    torch_dtype=torch_dtype,
                    local_files_only=not allow_download
                )
                
                controlnet_models.append(cnet)
                controlnet_images.append(pil_img)
                controlnet_scales.append(scale)

            # 3. Build the SD 1.5 Pipeline using BaseModelConfig optimizations
            logger.info("Loading Base Model: %s", base_model_id)
            
            pipe_kwargs = {
                "controlnet": controlnet_models,
                "torch_dtype": torch_dtype,
                "use_safetensors": base_model_dict.get("use_safetensors", True)
            }
            
            variant = base_model_dict.get("variant")
            if variant:
                pipe_kwargs["variant"] = variant

            pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_id, **pipe_kwargs)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config) 
            pipe = pipe.to(device)

            # 4. Generate the Frame
            logger.info("Generating stylized frame...")
            image_result = pipe(
                prompt,
                num_inference_steps=20,
                image=controlnet_images,
                controlnet_conditioning_scale=controlnet_scales,
            ).images[0]

            # 5. Save and Export via the Project System [MERGED FIX]
            out_bytes_io = io.BytesIO()
            # Force standard RGB and reset buffer to ensure Griptape's internal UI thumbnail 
            # generator doesn't choke on latent/metadata formats during write_bytes()
            clean_image = image_result.convert("RGB")
            clean_image.save(out_bytes_io, format="PNG")
            out_bytes_io.seek(0)
            
            dest = self._output_file.build_file()
            saved = dest.write_bytes(out_bytes_io.getvalue())
            
            self.parameter_output_values["generated_output"] = ImageUrlArtifact(saved.location)
            logger.info("Render complete! Saved to %s", saved.location)

            # 6. Aggressive cleanup
            del pipe
            del controlnet_models
            del controlnet_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error("Rendering failed: %s", e)
            raise RuntimeError(f"{self.name}: {str(e)}") from e
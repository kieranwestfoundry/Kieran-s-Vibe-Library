from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin, ParameterButtonGroup
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload

class BaseModelConfig(DataNode):
    """Selects the base diffusion model and its loading parameters."""
    
    DEFAULT_MODELS = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "black-forest-labs/FLUX.1-schnell"
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        choices = self._get_model_choices()
        default_val = choices[0] if choices else self.DEFAULT_MODELS[0]

        # 1. Base Model ID
        self.add_parameter(
            Parameter(
                name="model_id",
                tooltip="Select a base model from your local HuggingFace cache",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value=default_val,
                traits={Options(choices=choices, show_search=True)},
                ui_options={"display_name": "Model ID"}
            )
        )
        
        with ParameterButtonGroup(name="refresh_models_group") as btn_group:
            ParameterButton(
                name="refresh_models_btn",
                label="Refresh Local Models",
                icon="refresh-cw",
                on_click=self._refresh_models,
            )
        self.add_node_element(btn_group)

        # 2. Precision / Torch Dtype
        self.add_parameter(
            Parameter(
                name="torch_dtype",
                tooltip="The precision to load the model in",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value="float16",
                traits={Options(choices=["auto", "float16", "bfloat16", "float32"])},
                ui_options={"display_name": "Precision (dtype)"}
            )
        )

        # 3. Model Variant
        self.add_parameter(
            Parameter(
                name="variant",
                tooltip="Specific weight variant to load (auto-detected)",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value="",
                ui_options={"display_name": "Weight Variant"}
            )
        )

        # 4. Use Safetensors
        self.add_parameter(
            ParameterBool(
                name="use_safetensors",
                tooltip="Force loading from .safetensors files (auto-detected)",
                default_value=True,
                allow_input=True,
                allow_property=True,
                ui_options={"display_name": "Use Safetensors"}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="model_config",
                tooltip="Connect to the Render Node's base model input",
                type="dict",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Model Configuration"}
            )
        )

        # Initialize UI based on default value
        if default_val:
            self._update_params_from_local_model(default_val)

    def _get_hf_cache_path(self) -> Path:
        """Returns the base path to the HuggingFace hub cache."""
        hf_home = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
        return Path(hf_home)

    def _get_local_hf_models(self) -> list[str]:
        hf_path = self._get_hf_cache_path()
        models = set()
        if hf_path.exists():
            for d in hf_path.iterdir():
                if d.is_dir() and d.name.startswith("models--"):
                    model_name = d.name[8:].replace("--", "/")
                    models.add(model_name)
        return sorted(list(models))

    def _get_model_choices(self) -> list[str]:
        local_models = self._get_local_hf_models()
        return sorted(list(set(self.DEFAULT_MODELS + local_models)))

    def _refresh_models(self, button: Button, button_payload: ButtonDetailsMessagePayload) -> None:
        choices = self._get_model_choices()
        param = self.get_parameter_by_name("model_id")
        if param:
            for trait in param.find_elements_by_type(Options):
                trait.choices = choices
                break
            current_val = self.get_parameter_value("model_id")
            if current_val not in choices and choices:
                self.set_parameter_value("model_id", choices[0])

    def _update_params_from_local_model(self, model_id: str) -> None:
        """Inspects the local model folder to auto-detect supported variants and formats."""
        if not model_id:
            return
            
        folder_name = f"models--{model_id.replace('/', '--')}"
        model_path = self._get_hf_cache_path() / folder_name / "snapshots"
        
        has_safetensors = False
        has_fp16 = False
        
        # If the model is downloaded, scan its snapshot folders
        if model_path.exists():
            for snapshot in model_path.iterdir():
                if not snapshot.is_dir():
                    continue
                    
                # Recursively check the files in the snapshot
                for file_path in snapshot.rglob("*"):
                    if file_path.is_file():
                        if file_path.suffix == ".safetensors":
                            has_safetensors = True
                        if ".fp16." in file_path.name:
                            has_fp16 = True
                            
                # Break early if we found a snapshot and scanned it
                break 

        # Auto-configure based on findings
        self.set_parameter_value("use_safetensors", has_safetensors)
        
        # Determine variant and dtype
        if has_fp16:
            self.set_parameter_value("variant", "fp16")
            self.set_parameter_value("torch_dtype", "float16")
        else:
            self.set_parameter_value("variant", "")
            # FLUX models frequently use bfloat16 as standard without an explicit .bf16 variant file
            if "FLUX" in model_id:
                self.set_parameter_value("torch_dtype", "bfloat16")
            else:
                self.set_parameter_value("torch_dtype", "auto")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Triggered when the user selects a new model in the UI."""
        if parameter.name == "model_id":
            self._update_params_from_local_model(str(value))
            
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        self.parameter_output_values["model_config"] = {
            "model_id": self.get_parameter_value("model_id"),
            "torch_dtype": self.get_parameter_value("torch_dtype"),
            "variant": self.get_parameter_value("variant"),
            "use_safetensors": self.get_parameter_value("use_safetensors"),
        }
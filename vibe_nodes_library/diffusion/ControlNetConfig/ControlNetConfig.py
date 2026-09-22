from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

class ControlNetConfig(DataNode):
    """Configures a ControlNet model and routes it to a specific AOV pass."""
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.add_parameter(
            Parameter(
                name="aov_target",
                tooltip="The incoming AOV pass this ControlNet should analyze",
                type=ParameterTypeBuiltin.STR.value,
                traits={
                    Options(choices=["normal", "depth", "rgb_beauty", "position", "mask", "other"])
                },
                allowed_modes={ParameterMode.PROPERTY},
                default_value="normal",
            )
        )
        
        self.add_parameter(
            Parameter(
                name="controlnet_id",
                tooltip="HuggingFace ID or local path for the ControlNet model",
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                traits={
                    Options(choices=[
                        "lllyasviel/control_v11p_sd15_normal",
                        "lllyasviel/control_v11f1p_sd15_depth",
                        "lllyasviel/control_v11p_sd15_canny",
                    ])
                },
                default_value="lllyasviel/control_v11p_sd15_normal",
            )
        )
        
        self.add_parameter(
            Parameter(
                name="conditioning_scale",
                tooltip="Strength of the ControlNet influence (1.0 is standard)",
                type=ParameterTypeBuiltin.FLOAT.value,
                traits={Slider(min_val=0.0, max_val=2.0)},
                ui_options={"step": 0.05},
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value=1.0,
            )
        )

        # --- NEW: Fallback Download Toggle ---
        self.add_parameter(
            Parameter(
                name="allow_download",
                tooltip="If True, downloads the model from HuggingFace if not found locally.",
                type=ParameterTypeBuiltin.BOOL.value,
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                default_value=False,
            )
        )
        
        self.add_parameter(
            Parameter(
                name="controlnet_config",
                tooltip="Connect this output to the Render Node's 'controlnets' list",
                type="dict",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        self.parameter_output_values["controlnet_config"] = {
            "aov_target": self.get_parameter_value("aov_target"),
            "model_id": self.get_parameter_value("controlnet_id"),
            "conditioning_scale": self.get_parameter_value("conditioning_scale"),
            "allow_download": self.get_parameter_value("allow_download"),
        }
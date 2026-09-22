import logging
from typing import Any

# Third-party imports (Now guaranteed by pyproject.toml)
import tiktoken

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

logger = logging.getLogger(__name__)

class TokenEstimatorNode(DataNode):
    """
    Passthrough node that provides live UI estimates of token counts 
    using the tiktoken cl100k_base encoding.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Utils"
        self.description = "Estimates token count for input text in real-time."

        # 1. Text Input
        self.add_parameter(
            ParameterString(
                name="input_text",
                tooltip="Text to estimate tokens for.",
                default_value="",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True, 
                    "is_full_width": True,
                    "placeholder_text": "Paste text here..."
                }
            )
        )

        # 2. Live UI Feedback (Read-Only)
        self.add_parameter(
            Parameter(
                name="token_estimate_display",
                type="str",
                default_value="Estimated Tokens: 0",
                tooltip="Live token count estimation.",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={
                    "readonly": True, 
                    "display_name": "Estimation"
                }
            )
        )

        # 3. Outputs
        self.add_parameter(
            Parameter(
                name="output_text",
                output_type="str",
                type="str",
                tooltip="Passthrough of the original text.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False
            )
        )

        self.add_parameter(
            Parameter(
                name="token_count",
                output_type="int",
                type="int",
                tooltip="The integer token count for downstream math.",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False
            )
        )

    def _estimate_tokens(self, text: str) -> int:
        """Calculates tokens using OpenAI's standard encoding."""
        if not text:
            return 0
            
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(str(text)))

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Lifecycle hook to update the UI immediately when the text field changes."""
        if parameter.name == "input_text":
            count = self._estimate_tokens(value)
            self.set_parameter_value("token_estimate_display", f"Estimated Tokens: {count}")
        
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        """Standard execution pass-through."""
        text = self.get_parameter_value("input_text") or ""
        count = self._estimate_tokens(text)
        
        self.parameter_output_values["output_text"] = text
        self.parameter_output_values["token_count"] = count
import logging
from typing import Any

import tiktoken

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterGroup
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

logger = logging.getLogger(__name__)

class TokenBudgetNode(DataNode):
    """
    Calculates input token counts against specific model context limits,
    providing live UI feedback and passing constraints downstream.
    """
    
    # Hardcoded context windows for standard pipeline models
    MODEL_CONTEXT_LIMITS = {
        "gpt-4o": 128000,
        "claude-3-5-sonnet": 200000,
        "llama3.1": 128000,
        "qwen3:14b": 32768,
        "mistral": 32768
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Utils"
        self.description = "Estimates token usage against model limits and passes configs downstream."

        # ---------------------------------------------------------
        # 1. Text Input
        # ---------------------------------------------------------
        self.add_parameter(
            ParameterString(
                name="input_text",
                tooltip="Text payload to estimate.",
                default_value="",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True, 
                    "is_full_width": True,
                    "placeholder_text": "Paste prompt payload here..."
                }
            )
        )

        # ---------------------------------------------------------
        # 2. Model Settings (Grouped)
        # ---------------------------------------------------------
        self._budget_group = ParameterGroup(
            name="budget_settings",
            ui_options={"display_name": "Budget & Model Settings", "collapsed": False}
        )
        self.add_node_element(self._budget_group)

        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                type="str",
                default_value="llama3.1",
                tooltip="The target model. Used to determine maximum context window.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._budget_group.name,
                traits={Options(choices=list(self.MODEL_CONTEXT_LIMITS.keys()))}
            )
        )

        self.add_parameter(
            Parameter(
                name="max_tokens",
                input_types=["int"],
                type="int",
                default_value=0,
                tooltip="Override the max tokens to generate (0 = use model's maximum limit).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                parent_element_name=self._budget_group.name
            )
        )

        # ---------------------------------------------------------
        # 3. Live UI Feedback
        # ---------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="budget_display",
                type="str",
                default_value="Tokens: 0 / 128000",
                tooltip="Live token budget estimation.",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"readonly": True, "display_name": "Status"}
            )
        )

        # ---------------------------------------------------------
        # 4. Downstream Outputs
        # ---------------------------------------------------------
        self.add_parameter(Parameter(name="out_text", output_type="str", type="str", allowed_modes={ParameterMode.OUTPUT}, settable=False))
        self.add_parameter(Parameter(name="out_model", output_type="str", type="str", allowed_modes={ParameterMode.OUTPUT}, settable=False))
        self.add_parameter(Parameter(name="out_max_tokens", output_type="int", type="int", allowed_modes={ParameterMode.OUTPUT}, settable=False))


    def _update_budget_ui(self) -> None:
        """Calculates current usage vs limits and updates the readonly UI parameter."""
        text = self.get_parameter_value("input_text") or ""
        model = self.get_parameter_value("model") or "llama3.1"
        user_max = self.get_parameter_value("max_tokens") or 0

        # Calculate base tokens
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            used_tokens = len(enc.encode(str(text)))
        except Exception:
            used_tokens = 0

        # Determine the ceiling
        model_limit = self.MODEL_CONTEXT_LIMITS.get(model, 8192)
        actual_limit = user_max if (user_max > 0 and user_max < model_limit) else model_limit

        # Format HUD
        status = f"Used: {used_tokens:,} | Limit: {actual_limit:,}"
        if used_tokens > actual_limit:
            status = f"⚠️ EXCEEDED: {used_tokens:,} / {actual_limit:,}"
            
        self.set_parameter_value("budget_display", status)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Trigger UI updates whenever text, model, or max limits change."""
        if parameter.name in ("input_text", "model", "max_tokens"):
            self._update_budget_ui()
            
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        """Pass the text and explicit configurations downstream."""
        text = self.get_parameter_value("input_text") or ""
        model = self.get_parameter_value("model") or "llama3.1"
        user_max = self.get_parameter_value("max_tokens") or 0
        
        actual_limit = user_max if user_max > 0 else self.MODEL_CONTEXT_LIMITS.get(model, 8192)

        self.parameter_output_values["out_text"] = text
        self.parameter_output_values["out_model"] = model
        self.parameter_output_values["out_max_tokens"] = actual_limit
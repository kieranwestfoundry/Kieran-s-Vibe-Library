from typing import Any

from griptape_nodes.exe_types.core_types import (
    ControlParameterInput,
    ControlParameterOutput,
    Parameter,
    ParameterGroup,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString


class StringMultiplexer(BaseNode):
    """Routes 1 of 5 data inputs to the output based on matching an incoming string (A) to a key."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Execution flow
        self.add_parameter(
            ControlParameterInput(
                tooltip="Trigger the selection",
                name="exec_in",
            )
        )
        self.add_parameter(
            ControlParameterOutput(
                tooltip="Continue execution after selection",
                name="exec_out",
            )
        )

        # The incoming string to evaluate
        self.A = ParameterString(
            name="A",
            tooltip="Incoming string to match against the branch keys.",
            default_value="",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Input String (A)"}
        )
        self.add_parameter(self.A)

        self.match_params: list[ParameterString] = []
        self.input_params: list[Parameter] = []
        self._last_condition_param: Parameter | None = None

        # Data flow parameters in a collapsible ParameterGroup
        with ParameterGroup(name="Dictionary Multiplexer") as group:
            for i in range(5):
                # The "Key" (Text input on the node)
                match_param = ParameterString(
                    name=f"match_{i}",
                    tooltip=f"String key to match for branch {i} (e.g., 'foo')",
                    default_value=f"key_{i}",
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                    ui_options={"display_name": f"Match Key {i}"}
                )
                self.match_params.append(match_param)

                # The "Value" (The pipeline data connected to this branch)
                input_param = Parameter(
                    name=f"input_{i}",
                    tooltip=f"Data to output if A matches Match Key {i}",
                    input_types=["any"],
                    type="any",
                    allowed_modes={ParameterMode.INPUT},
                    default_value=None,
                )
                self.input_params.append(input_param)

            self.output = Parameter(
                name="output",
                tooltip="Selected output based on the string match",
                output_type=ParameterTypeBuiltin.ALL.value,
                type=ParameterTypeBuiltin.ALL.value,
                allowed_modes={ParameterMode.OUTPUT},
                default_value=None,
            )

        self.add_node_element(group)

    def process(self) -> None:
        """Find which match key equals A, and route its corresponding input to output."""
        a_val = self.get_parameter_value("A")
        if a_val is None:
            a_val = ""

        matched_idx = -1
        for i, mp in enumerate(self.match_params):
            m_val = self.get_parameter_value(mp.name)
            if str(a_val) == str(m_val):
                matched_idx = i
                break

        if matched_idx >= 0:
            selected_param = self.input_params[matched_idx]
            self.parameter_output_values["output"] = self.get_parameter_value(selected_param.name)
        else:
            self.parameter_output_values["output"] = None

    def initialize_spotlight(self) -> None:
        """Initialize conditional evaluation graph.
        
        Evaluate 'A' and all 'match_X' inputs first. Then branch to the matching data input.
        """
        condition_params = []
        
        # Add 'A' if it's connected as an input
        if ParameterMode.INPUT in self.A.get_mode():
            condition_params.append(self.A)
            
        # Add any match keys that are connected as inputs
        for mp in self.match_params:
            if ParameterMode.INPUT in mp.get_mode():
                condition_params.append(mp)

        if condition_params:
            # Link all conditions sequentially so they evaluate before we branch
            for i in range(len(condition_params) - 1):
                condition_params[i].next = condition_params[i+1]
                condition_params[i+1].prev = condition_params[i]

            self.current_spotlight_parameter = condition_params[0]
            self._last_condition_param = condition_params[-1]
        else:
            # If everything is a hardcoded property, branch immediately
            self._branch_spotlight(None)

    def _branch_spotlight(self, current_param: Parameter | None) -> bool:
        """Evaluate the match and link ONLY the winning data input branch."""
        try:
            a_val = self.get_parameter_value("A")
            if a_val is None:
                a_val = ""

            matched_idx = -1
            for i, mp in enumerate(self.match_params):
                m_val = self.get_parameter_value(mp.name)
                if str(a_val) == str(m_val):
                    matched_idx = i
                    break
        except Exception:
            self.current_spotlight_parameter = None
            return False

        if matched_idx >= 0:
            next_param = self.input_params[matched_idx]
            if ParameterMode.INPUT in next_param.get_mode():
                if current_param:
                    current_param.next = next_param
                    next_param.prev = current_param
                self.current_spotlight_parameter = next_param
                return True

        self.current_spotlight_parameter = None
        return False

    def advance_parameter(self) -> bool:
        """Advance the evaluation graph."""
        if self.current_spotlight_parameter is None:
            return False

        # If we just finished evaluating the last condition parameter, time to branch!
        if self.current_spotlight_parameter is getattr(self, "_last_condition_param", None):
            return self._branch_spotlight(self.current_spotlight_parameter)

        # Otherwise, normal advancement
        if self.current_spotlight_parameter.next is not None:
            self.current_spotlight_parameter = self.current_spotlight_parameter.next
            return True

        self.current_spotlight_parameter = None
        return False
import ast
import json
import logging
import os
import subprocess
import sys
from contextlib import suppress
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.events.parameter_events import (
    RemoveParameterFromNodeRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger(__name__)


class PythonScriptRunnerNode(ControlNode):
    """Parses a Python script for argparse definitions and runs it with dynamically generated parameters."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        self._initializing = True
        super().__init__(name, metadata)

        self._dynamic_param_names: list[str] = []
        self._script_args_meta: list[dict[str, Any]] = []

        # Input parameter for the target script path
        self.add_parameter(
            Parameter(
                name="script_path",
                input_types=["str"],
                type=ParameterTypeBuiltin.STR.value,
                tooltip="Absolute path to the Python script to execute.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True},
            )
        )

        # Output parameter for the script's parsed JSON results
        self.add_parameter(
            Parameter(
                name="script_output",
                output_type="json",
                type="json",
                tooltip="JSON output returned by the script.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._initializing = False

    def _log(self, message: str) -> None:
        """Safe logging with exception suppression."""
        with suppress(Exception):
            logger.info(message)

    def remove_parameter_element_by_name(self, element_name: str) -> None:
        """Helper to cleanly remove parameters from the UI and graph."""
        if self.get_element_by_name_and_type(element_name):
            GriptapeNodes.handle_request(
                RemoveParameterFromNodeRequest(
                    parameter_name=element_name, node_name=self.name
                )
            )

    def _parse_script_args(self, script_path: str) -> list[dict[str, Any]]:
        """Parses the target Python script AST to find parser.add_argument calls."""
        if not script_path or not os.path.isfile(script_path):
            return []

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            self._log(f"Failed to parse script AST: {e}")
            return []

        args_meta = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"
                ):
                    flags = [
                        arg.value for arg in node.args if isinstance(arg, ast.Constant)
                    ]
                    if not flags:
                        continue

                    kwargs = {}
                    for kw in node.keywords:
                        if isinstance(kw.value, ast.Constant):
                            kwargs[kw.arg] = kw.value.value
                        elif isinstance(kw.value, ast.Name):
                            kwargs[kw.arg] = kw.value.id

                    # Determine parameter name and flag type
                    long_flag = next((f for f in flags if f.startswith("--")), None)
                    short_flag = next(
                        (
                            f
                            for f in flags
                            if f.startswith("-") and not f.startswith("--")
                        ),
                        None,
                    )
                    positional = next((f for f in flags if not f.startswith("-")), None)

                    if long_flag:
                        param_name = long_flag.lstrip("-").replace("-", "_")
                        flag_to_use = long_flag
                    elif short_flag:
                        param_name = short_flag.lstrip("-")
                        flag_to_use = short_flag
                    elif positional:
                        param_name = positional.replace("-", "_")
                        flag_to_use = ""  # Empty string indicates a positional argument
                    else:
                        continue

                    args_meta.append(
                        {
                            "name": param_name,
                            "flags": flags,
                            "flag_to_use": flag_to_use,
                            "type": kwargs.get("type", "str"),
                            "action": kwargs.get("action", "store"),
                            "required": kwargs.get("required", False),
                            "help": kwargs.get("help", "No description provided."),
                            "default": kwargs.get("default", None),
                        }
                    )
        return args_meta

    def _map_arg_type(self, arg_type: str, action: str) -> str:
        """Maps argparse types to Griptape Parameter types."""
        if action in ("store_true", "store_false"):
            return ParameterTypeBuiltin.BOOL.value
        if arg_type == "int":
            return ParameterTypeBuiltin.INT.value
        if arg_type == "float":
            return ParameterTypeBuiltin.FLOAT.value
        return ParameterTypeBuiltin.STR.value

    def _update_dynamic_parameters(self, script_path: str) -> None:
        """Clears old dynamic parameters and adds new ones based on the parsed script."""
        # Clean up existing dynamic parameters
        for param_name in self._dynamic_param_names:
            self.remove_parameter_element_by_name(param_name)
        self._dynamic_param_names.clear()

        # Parse new parameters
        self._script_args_meta = self._parse_script_args(script_path)

        # Create new UI parameters
        for arg in self._script_args_meta:
            param_type = self._map_arg_type(arg["type"], arg["action"])
            param_name = f"arg_{arg['name']}"

            # Format display name based on whether it's positional or flagged
            display_name = (
                f"--{arg['name'].replace('_', '-')}"
                if arg["flag_to_use"]
                else arg["name"]
            )

            new_param = Parameter(
                name=param_name,
                input_types=[param_type],
                type=param_type,
                default_value=arg["default"],
                tooltip=f"{'[REQUIRED] ' if arg['required'] else ''}{arg['help']}",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": display_name},
                user_defined=True,  # Mark for serialization
            )

            self.add_parameter(new_param)
            self._dynamic_param_names.append(param_name)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Trigger parameter recreation when the script path changes."""
        if parameter.name == "script_path" and value:
            self._update_dynamic_parameters(str(value))

        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validates that the script path is valid and all required args are provided."""
        exceptions = []
        script_path = self.get_parameter_value("script_path")

        if not script_path or not os.path.isfile(script_path):
            exceptions.append(
                ValueError(f"{self.name}: Valid script path is required.")
            )

        for arg in self._script_args_meta:
            if arg["required"]:
                param_value = self.get_parameter_value(f"arg_{arg['name']}")
                if param_value is None or param_value == "":
                    exceptions.append(
                        ValueError(
                            f"{self.name}: Missing required argument {arg['name']}"
                        )
                    )

        return exceptions if exceptions else None

    def process(self) -> AsyncResult:
        """Yields the processing task to the async engine."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Builds the subprocess command, executes it, and parses the JSON output."""
        script_path = self.get_parameter_value("script_path")
        self.parameter_output_values["script_output"] = None

        cmd = [sys.executable, script_path]

        # Build command-line arguments based on parameter states
        for arg in self._script_args_meta:
            param_value = self.get_parameter_value(f"arg_{arg['name']}")

            # Skip empty optionals
            if param_value is None or param_value == "":
                continue

            # Handle boolean flags (store_true / store_false)
            if arg["action"] == "store_true" and param_value:
                if arg["flag_to_use"]:
                    cmd.append(arg["flag_to_use"])
            elif arg["action"] == "store_false" and not param_value:
                if arg["flag_to_use"]:
                    cmd.append(arg["flag_to_use"])
            elif arg["action"] not in ("store_true", "store_false"):
                # Clean literal quotes wrapped around UI input strings (e.g. escaping spaces)
                cleaned_value = str(param_value).strip("'\"")

                # Standard argument (add flag if it exists, then the value)
                if arg["flag_to_use"]:
                    cmd.extend([arg["flag_to_use"], cleaned_value])
                else:
                    # It's a positional argument, just append the value
                    cmd.append(cleaned_value)

        self._log(f"Executing: {' '.join(cmd)}")

        try:
            # Run the script and capture stdout
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            stdout_str = result.stdout.strip()

            # Attempt to parse script output as JSON
            try:
                parsed_json = json.loads(stdout_str)
                self.parameter_output_values["script_output"] = parsed_json
            except json.JSONDecodeError:
                self._log(
                    "Script executed successfully, but output was not valid JSON. Wrapping in dict."
                )
                self.parameter_output_values["script_output"] = {
                    "raw_output": stdout_str
                }

        except subprocess.CalledProcessError as e:
            self._log(
                f"Script execution failed with return code {e.returncode}. Stderr: {e.stderr}"
            )
            raise RuntimeError(f"Script execution failed: {e.stderr}") from e

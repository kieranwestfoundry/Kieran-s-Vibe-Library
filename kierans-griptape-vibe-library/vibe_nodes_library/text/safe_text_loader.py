import os
import logging
from contextlib import suppress
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter, 
    ParameterMode, 
    ParameterTypeBuiltin
)
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult

logger = logging.getLogger(__name__)

class SafeTextLoader(ControlNode):
    """Loads text from a file securely, respecting execution flow to prevent crashes on binary files."""

    def __init__(self, name: str = "Safe Text Loader", metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # NOTE: ControlNode automatically adds exec_in and exec_out, so we DO NOT add them manually!

        self.add_parameter(
            Parameter(
                name="path",
                input_types=["str"],
                type=ParameterTypeBuiltin.STR.value,
                tooltip="Path to the text file (e.g., .mtlx)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "File Path"}
            )
        )

        self.add_parameter(
            Parameter(
                name="output",
                output_type=ParameterTypeBuiltin.STR.value,
                tooltip="The loaded text content",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True}
            )
        )

    def _log(self, message: str) -> None:
        """Safe logging with exception suppression."""
        with suppress(Exception):
            logger.info(message)

    def process(self) -> AsyncResult:
        """Yield to background thread to prevent UI freezing."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Main processing logic."""
        # Set a safe default before trying anything
        self.parameter_output_values["output"] = None

        file_path = self.get_parameter_value("path")
        
        # 1. Ensure path exists
        if not file_path or not os.path.exists(str(file_path)):
            self._log(f"SafeTextLoader: Path does not exist or is empty: {file_path}")
            return

        # 2. Try to read the file, safely catching binary decoding errors
        try:
            with open(str(file_path), "r", encoding="utf-8") as f:
                content = f.read()
                self.parameter_output_values["output"] = content
                self._log(f"SafeTextLoader: Successfully loaded {file_path}")
                
        except UnicodeDecodeError:
            # If it's a binary file (like .usd or .exr) that sneaks through, catch it!
            self._log(f"SafeTextLoader: SKIPPED binary file. Expected text, got binary at {file_path}")
        except Exception as e:
            self._log(f"SafeTextLoader: Unexpected error loading file: {e}")
            raise RuntimeError(f"{self.name}: Failed to load text - {e}") from e
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterButtonGroup,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload


class UnzipFileNode(DataNode):
    """A node that unzips a target zip file to a specified destination directory."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # 1. Zip File Path (Input/Property)
        self.add_parameter(
            Parameter(
                name="zip_file_path",
                tooltip="The full file path to the target .zip file.",
                type=ParameterTypeBuiltin.STR.value,
                input_types=["str"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Zip File Path",
                    "placeholder_text": "/path/to/archive.zip",
                },
            )
        )

        # 2. Unzip Destination Path (Input/Property)
        self.add_parameter(
            Parameter(
                name="unzip_dest_path",
                tooltip="The folder path where files will be extracted.",
                type=ParameterTypeBuiltin.STR.value,
                input_types=["str"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Extraction Destination",
                    "placeholder_text": "/path/to/extract/folder",
                },
            )
        )

        # 3. Versioning Button
        with ParameterButtonGroup(name="versioning_button_group") as versioning_buttons:
            ParameterButton(
                name="append_version_timestamp",
                label="Append Version Timestamp",
                icon="clock",
                on_click=self._append_timestamp,
            )
        self.add_node_element(versioning_buttons)

        # 4. Outputs
        self.add_parameter(
            Parameter(
                name="extracted_files",
                tooltip="A list of full file paths for all extracted contents.",
                type=ParameterTypeBuiltin.ANY.value,
                output_type="list[str]",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Extracted Files"},
            )
        )

        self.add_parameter(
            Parameter(
                name="status",
                tooltip="Status message regarding the extraction process.",
                type=ParameterTypeBuiltin.STR.value,
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def _append_timestamp(
        self,
        button: Button,
        button_payload: ButtonDetailsMessagePayload,
    ) -> None:
        """Button handler: Appends a datetime string to the destination path."""
        current_dest = self.get_parameter_value("unzip_dest_path") or ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if current_dest:
            new_dest = str(Path(current_dest) / f"v_{timestamp}")
        else:
            new_dest = f"extracted_v_{timestamp}"

        self.set_parameter_value("unzip_dest_path", new_dest)

    def process(self) -> None:
        """Executes the node logic to unzip the file."""
        zip_path_str = self.get_parameter_value("zip_file_path")
        dest_path_str = self.get_parameter_value("unzip_dest_path")

        # Validation
        if not zip_path_str or not dest_path_str:
            self.parameter_output_values["status"] = "Error: Missing zip file path or destination path."
            self.parameter_output_values["extracted_files"] = []
            return

        zip_path = Path(zip_path_str)
        dest_path = Path(dest_path_str)

        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            self.parameter_output_values["status"] = f"Error: Invalid or missing zip file at '{zip_path}'."
            self.parameter_output_values["extracted_files"] = []
            return

        # Extraction logic
        try:
            dest_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_path)
                # Create a list of the extracted absolute file paths
                extracted = [str(dest_path / name) for name in zip_ref.namelist()]

            self.parameter_output_values["status"] = f"Success: Extracted {len(extracted)} files."
            self.parameter_output_values["extracted_files"] = extracted

        except Exception as e:
            self.parameter_output_values["status"] = f"Error during extraction: {str(e)}"
            self.parameter_output_values["extracted_files"] = []

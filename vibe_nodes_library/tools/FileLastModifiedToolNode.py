import os
from datetime import datetime
from typing import Any

from schema import Literal, Schema

from griptape.artifacts import ErrorArtifact, TextArtifact
from griptape.tools import BaseTool as GtBaseTool
from griptape.utils.decorators import activity

from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode


class FileLastModifiedSdkTool(GtBaseTool):
    """Griptape SDK Tool for checking a file's last modified date."""

    @activity(
        config={
            "description": "Can be used to get the last modified date and time of a file on the disk.",
            "schema": Schema(
                {
                    Literal(
                        "file_path",
                        description="The absolute or relative path to the file.",
                    ): str
                }
            ),
        }
    )
    def get_last_modified_date(self, params: dict) -> TextArtifact | ErrorArtifact:
        """Fetch the last modified date of the specified file."""
        file_path = params["values"]["file_path"]
        try:
            mtime = os.path.getmtime(file_path)
            dt = datetime.fromtimestamp(mtime)

            # Format the datetime in a highly readable format for the LLM
            formatted_date = dt.strftime('%Y-%m-%d %H:%M:%S')
            return TextArtifact(f"The file '{file_path}' was last modified on {formatted_date}.")

        except FileNotFoundError:
            return ErrorArtifact(f"File not found: {file_path}")
        except PermissionError:
            return ErrorArtifact(f"Permission denied when accessing file: {file_path}")
        except Exception as e:
            return ErrorArtifact(f"Error accessing file: {str(e)}")


class FileLastModifiedToolNode(DataNode):
    """Griptape Node wrapper that provides the FileLastModifiedSdkTool to an agent."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # UI informational message
        self.add_node_element(
            ParameterMessage(
                name="tool_info",
                variant="info",
                value="This tool gives an agent the ability to check the last modified date of files on disk.",
            )
        )

        # Configuration property
        self.add_parameter(
            Parameter(
                name="off_prompt",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="Whether the tool should operate in off-prompt mode (preventing the LLM from seeing the raw output directly).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Off Prompt"},
            )
        )

        # Output property to connect to the Agent node
        self.add_parameter(
            Parameter(
                name="tool",
                output_type="Tool",
                tooltip="The generated File Last Modified Tool ready to be connected to an Agent.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        """Run the node to instantiate the tool."""
        # Retrieve the off_prompt configuration (handling None safely)
        off_prompt = self.get_parameter_value("off_prompt") or False

        # Initialize the SDK tool with the specified configuration
        tool = FileLastModifiedSdkTool(off_prompt=bool(off_prompt))

        # Set the tool as the output so it can be wired downstream
        self.parameter_output_values["tool"] = tool

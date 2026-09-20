import logging
import os
from contextlib import suppress

import requests

# Griptape Nodes Engine
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult

logger = logging.getLogger(__name__)

class NormalCrafterSharedMediaNode(ControlNode):
    """
    Triggers remote NormalCrafter inference on an M1 server.
    Accepts both Image and Video file paths from a shared network drive.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.category = "Image AI"
        self.description = "Generates Normal Maps from Shared Network Images or Video."

        # ---------------------------------------------------------
        # Inputs
        # ---------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="server_address",
                input_types=["str"],
                type="str",
                default_value="http://192.168.1.100:8000",
                tooltip="The IP/URL of the M1 NormalCrafter server.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )
        
        self.add_parameter(
            Parameter(
                name="input_media",
                input_types=["str"], 
                type="str",
                default_value="",
                tooltip="Absolute path to the source RGB image or video on the shared drive.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )

        self.add_parameter(
            Parameter(
                name="output_dir",
                input_types=["str"],
                type="str",
                default_value="/mnt/shared_drive/textures/normals",
                tooltip="Directory to save the generated normal map.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}
            )
        )

        self.add_parameter(
            Parameter(
                name="single_frame_test",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip="If true AND input is a video, processes only the first frame.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Single Frame Test (Video Only)"}
            )
        )

        # ---------------------------------------------------------
        # Outputs
        # ---------------------------------------------------------
        self.add_parameter(
            Parameter(
                name="out_media",
                output_type="str", 
                type="str",
                tooltip="Absolute path to the generated normal map (.mp4 or .png).",
                allowed_modes={ParameterMode.OUTPUT},
                settable=False
            )
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(f"{self.name}: {message}")

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []
        media_path = self.get_parameter_value("input_media")
        if not media_path or not os.path.exists(media_path):
            exceptions.append(FileNotFoundError(f"{self.name}: Input media not found on workstation at '{media_path}'"))
        return exceptions if exceptions else None

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        server_addr = self.get_parameter_value("server_address")
        input_path = self.get_parameter_value("input_media")
        out_dir = self.get_parameter_value("output_dir")
        single_frame = self.get_parameter_value("single_frame_test")

        endpoint = f"{server_addr.rstrip('/')}/generate_normal_shared"
        
        # Lightweight payload (Path only, no Base64 video data)
        payload = {
            "input_path": input_path,
            "output_dir": out_dir,
            "single_frame_test": single_frame
        }

        try:
            mode_log = "SINGLE FRAME TEST" if single_frame else "FULL SEQUENCE"
            self._log(f"Dispatching media job ({mode_log}) to {server_addr} for path: {input_path}")
            
            # Keep timeout high; the HTTP connection still stays open while the M1 renders
            response = requests.post(endpoint, json=payload, timeout=3600) 
            response.raise_for_status()
            
            r_json = response.json()
            returned_path = r_json.get("output_path")
            
            if not returned_path:
                raise RuntimeError("M1 server completed, but returned no output path.")

            self.parameter_output_values["out_media"] = returned_path
            self._log(f"Render complete. Output located at: {returned_path}")

        except Exception as e:
            self.parameter_output_values["out_media"] = ""
            self._log(f"Generation Error: {e}")
            raise RuntimeError(f"Generation Error: {e}") from e
import os
import tempfile
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.files.file import File
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes_library.utils.image_utils import dict_to_image_url_artifact


class EXRToPNGConverter(SuccessFailureNode):
    """Converts EXR images to PNG and rescales to a maximum of 2K resolution."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Dual image/path input parameter
        self.add_parameter(
            Parameter(
                name="input_exr",
                input_types=["str", "ImageUrlArtifact", "ImageArtifact", "any"],
                type="str",
                tooltip="Path to the EXR file, or an incoming image artifact",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Input EXR (Path/Image)",
                    "clickable_file_browser": True,
                    "expander": True,
                },
            )
        )

        # Output image parameter 
        self.add_parameter(
            Parameter(
                name="output_image",
                tooltip="Converted and 2K scaled PNG result (embedded base64)",
                output_type="ImageArtifact",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the EXR to PNG conversion",
            result_details_placeholder="Conversion details will appear here.",
        )

    def _extract_bytes(self, input_val: Any) -> bytes:
        """Duck-type extraction of file bytes from string paths or artifacts."""
        if isinstance(input_val, str):
            if os.path.exists(input_val):
                with open(input_val, "rb") as f:
                    return f.read()
            return File(input_val).read_bytes()

        if isinstance(input_val, dict):
            art = dict_to_image_url_artifact(input_val)
            return File(art.value).read_bytes()

        if hasattr(input_val, "to_bytes"):
            return input_val.to_bytes()
        
        if hasattr(input_val, "value") and isinstance(input_val.value, str):
            return File(input_val.value).read_bytes()

        raise ValueError("Unsupported input format. Please provide a valid file path or image artifact.")

    def process(self) -> None:
        """Main processing entry point."""
        self._clear_execution_status()
        self.parameter_output_values["output_image"] = None

        try:
            import cv2  # type: ignore[import-untyped]
        except ImportError as e:
            error_msg = "OpenCV (cv2) is required. Please install 'opencv-python'."
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            raise ImportError(error_msg) from e

        input_val = self.get_parameter_value("input_exr")
        if not input_val:
            error_msg = "No input EXR provided."
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            raise ValueError(error_msg)
        if isinstance(input_val, str) and not input_val.lower().endswith(".exr"):
            self.parameter_output_values["output_image"] = None
            self._set_status_results(
                was_successful=True, 
                result_details=f"SKIPPED: {input_val} is not an EXR file."
            )
            return

        try:
            input_bytes = self._extract_bytes(input_val)

            with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(input_bytes)

            try:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                img_float = cv2.imread(temp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

                if img_float is None:
                    raise ValueError("Failed to decode EXR image. Ensure the file is a valid EXR format.")

                if len(img_float.shape) == 3 and img_float.shape[2] >= 3:
                    img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)

                img_clipped = np.clip(img_float, 0.0, 1.0)
                img_uint8 = (img_clipped ** (1.0 / 2.2) * 255.0).astype(np.uint8)

                pil_img = Image.fromarray(img_uint8)
                pil_img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)

                # --- NEW MODERN V3 SAVING LOGIC ---
                # 1. Save PIL image to an in-memory byte buffer
                img_byte_arr = BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                output_bytes = img_byte_arr.getvalue()
                
                # 2. Use ProjectFileDestination to route it to the correct project folder
                dest = ProjectFileDestination.from_situation(
                    filename="converted_exr.png", 
                    situation="save_node_output"
                )
                
                # 3. Write bytes and get the properly resolved, saved file location
                saved_file = dest.write_bytes(output_bytes)

                # 4. Wrap the raw bytes in an ImageArtifact for downstream nodes
                self.parameter_output_values["output_image"] = ImageArtifact(
                    value=output_bytes,
                    format="png",
                    width=pil_img.width,
                    height=pil_img.height
                )
                
                self._set_status_results(
                    was_successful=True,
                    result_details=f"SUCCESS: EXR converted to PNG. New dimensions: {pil_img.width}x{pil_img.height}",
                )

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:
            error_details = f"Failed to process EXR: {e}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
            raise RuntimeError(error_details) from e
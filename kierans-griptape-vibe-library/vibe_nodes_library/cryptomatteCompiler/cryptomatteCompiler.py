"""Cryptomatte Compiler Node for Griptape.

Dependencies:
    OpenEXR~=3.3.0
    numpy~=1.26.0
    opencv-python>=4.11.0
    mmh3>=5.0.0
"""

import os
import json
import struct
from typing import Any

# type: ignore[import-untyped] is used for libraries lacking type stubs
import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
import mmh3  # type: ignore[import-untyped]
import OpenEXR  # type: ignore[import-untyped]

from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.core_types import (
    Parameter, 
    ParameterList, 
    ParameterMode, 
    ParameterTypeBuiltin
)
from griptape_nodes.traits.file_system_picker import FileSystemPicker


class CryptomatteCompiler(SuccessFailureNode):
    """Compiles binary mask videos into a VFX CY2025 compliant Cryptomatte EXR sequence."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # --- Metadata Parameters ---
        self.add_parameter(Parameter(
            name="total_frames",
            type=ParameterTypeBuiltin.INT.value,
            default_value=150,
            tooltip="Total frames to process"
        ))
        
        self.add_parameter(Parameter(
            name="resolution_width",
            type=ParameterTypeBuiltin.INT.value,
            default_value=1920,
            tooltip="Output EXR width"
        ))
        
        self.add_parameter(Parameter(
            name="resolution_height",
            type=ParameterTypeBuiltin.INT.value,
            default_value=1080,
            tooltip="Output EXR height"
        ))
        
        self.add_parameter(Parameter(
            name="layer_name",
            type=ParameterTypeBuiltin.STR.value,
            default_value="SAM2_Objects",
            tooltip="The root name for the Cryptomatte layer"
        ))
        
        self.add_parameter(Parameter(
            name="output_dir",
            type=ParameterTypeBuiltin.STR.value,
            default_value="./cryptomatte_renders",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Directory to save the EXR sequence",
            traits={FileSystemPicker()}
        ))
        
        self.add_parameter(Parameter(
            name="frame_padding",
            type=ParameterTypeBuiltin.INT.value,
            default_value=4,
            tooltip="Zero-padding for frame numbers (e.g., 4 = 0001)"
        ))
        
        self.add_parameter(Parameter(
            name="start_frame",
            type=ParameterTypeBuiltin.INT.value,
            default_value=1001,
            tooltip="The starting frame number for the sequence"
        ))

        # --- List Parameters for Layers ---
        self.add_parameter(ParameterList(
            name="items",
            input_types=["list", "list[str]", "str"],
            allowed_modes={ParameterMode.INPUT},
            tooltip="List of item names (e.g., ['Car', 'Pedestrian'])"
        ))
        
        self.add_parameter(ParameterList(
            name="pathtovideo",
            input_types=["list", "list[str]", "str"],
            allowed_modes={ParameterMode.INPUT},
            tooltip="List of file paths to the MP4 masks corresponding to the items"
        ))

        # --- Outputs ---
        self.add_parameter(Parameter(
            name="output_path",
            tooltip="Directory containing the finalized Cryptomatte EXR sequence",
            type=ParameterTypeBuiltin.STR.value,
            output_type=ParameterTypeBuiltin.STR.value,
            allowed_modes={ParameterMode.OUTPUT},
        ))

        self._create_status_parameters(
            result_details_tooltip="Details about the EXR compilation operation",
            result_details_placeholder="Execution details will appear here.",
        )

    def process(self) -> None:
        """Execute the compilation from MP4 masks to Cryptomatte EXRs."""
        self._clear_execution_status()
        self.parameter_output_values["output_path"] = None

        try:
            # 1. Retrieve & Validate Lists
            items = self.get_parameter_list_value("items") or []
            paths = self.get_parameter_list_value("pathtovideo") or []

            if len(items) != len(paths):
                raise ValueError(
                    f"List mismatch: Received {len(items)} items but {len(paths)} video paths. "
                    "They must correspond 1:1."
                )

            # 2. Retrieve Metadata
            width = self.get_parameter_value("resolution_width")
            height = self.get_parameter_value("resolution_height")
            total_frames = self.get_parameter_value("total_frames")
            layer_name = self.get_parameter_value("layer_name")
            output_dir = self.get_parameter_value("output_dir")
            frame_padding = self.get_parameter_value("frame_padding")
            start_frame = self.get_parameter_value("start_frame")

            if not output_dir:
                raise ValueError("output_dir must be specified.")
            os.makedirs(output_dir, exist_ok=True)

            # 3. Cryptomatte ID Generation & Manifest
            manifest = {}
            hash_floats = {}

            for item_name in items:
                item_str = str(item_name)
                hash_int = mmh3.hash(item_str, 0, signed=False)
                hash_float = struct.unpack('<f', struct.pack('<I', hash_int))[0]
                hash_hex = f"{hash_int:08x}"

                manifest[item_str] = hash_hex
                hash_floats[item_str] = hash_float

            # 4. Video Ingestion Setup
            caps = {}
            for item_name, vid_path in zip(items, paths):
                item_str = str(item_name)
                path_str = str(vid_path)
                cap = cv2.VideoCapture(path_str)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open mask video: {path_str}")
                caps[item_str] = cap

            try:
                # 5. Frame Processing Loop
                for f in range(total_frames):
                    # Initialize blank 32-bit float arrays for RGBA
                    ch_r = np.zeros((height, width), dtype=np.float32)
                    ch_g = np.zeros((height, width), dtype=np.float32)
                    ch_b = np.zeros((height, width), dtype=np.float32)
                    ch_a = np.zeros((height, width), dtype=np.float32)

                    for item_name in items:
                        item_str = str(item_name)
                        cap = caps[item_str]
                        ret, frame = cap.read()

                        if not ret:
                            continue  

                        # Threshold first channel to create binary mask
                        mask = (frame[:, :, 0] > 127).astype(np.float32)

                        # Write hash_float to R, 1.0 coverage to G where mask is active
                        active_pixels = mask > 0
                        ch_r[active_pixels] = hash_floats[item_str]
                        ch_g[active_pixels] = 1.0

                    # 6. EXR Header & Channels (New OpenEXR 3.3+ API)
                    current_frame = start_frame + f
                    filename = f"crypto_{current_frame:0{frame_padding}d}.exr"
                    filepath = os.path.join(output_dir, filename)

                    exr_header = {
                        "compression": OpenEXR.ZIP_COMPRESSION,
                        "type": OpenEXR.scanlineimage,
                        "dataWindow": ((0, 0), (int(width) - 1, int(height) - 1)),
                        "displayWindow": ((0, 0), (int(width) - 1, int(height) - 1)),
                        # The modern API accepts custom metadata effortlessly
                        "cryptomatte/0/name": str(layer_name),
                        "cryptomatte/0/hash": "MurmurHash3_32",
                        "cryptomatte/0/manifest": json.dumps(manifest)
                    }

                    exr_channels = {
                        f"{layer_name}00.R": ch_r,
                        f"{layer_name}00.G": ch_g,
                        f"{layer_name}00.B": ch_b,
                        f"{layer_name}00.A": ch_a
                    }

                    # 7. Write File via Context Manager
                    with OpenEXR.File(exr_header, exr_channels) as out_file:
                        out_file.write(filepath)

            finally:
                # Cleanup capture objects unconditionally
                for cap in caps.values():
                    cap.release()

            # Set outputs and success status
            self.parameter_output_values["output_path"] = output_dir
            self._set_status_results(
                was_successful=True, 
                result_details=f"SUCCESS: Compiled {total_frames} frames to {output_dir}"
            )

        except Exception as e:
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {str(e)}")
            self._handle_failure_exception(e)
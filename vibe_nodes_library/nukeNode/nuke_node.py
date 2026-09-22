import json
import logging
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
import requests
from contextlib import suppress
from pathlib import Path
from typing import Any
from io import BytesIO
from PIL import Image

from griptape.artifacts import ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin, ParameterGroup
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat

# Import GriptapeNodes to access the SecretsManager and Project System
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.files.project_file import ProjectFileDestination

logger = logging.getLogger(__name__)


class BaseNukeNode(ControlNode):
    """
    Abstract base class for headless Nuke processing nodes.
    Handles script generation, path resolution, and subprocess execution.
    """
    
    NUKE_SECRET_NAME = "NUKE_BIN_PATH"
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.nuke_node_class = ""
        self._nuke_image_pipes: dict[int, str] = {}
        self._temp_downloads = []

        # 1. Input Frame Range Overrides
        self.add_parameter(
            ParameterBool(
                name="override_input_frame_range",
                default_value=False,
                tooltip="If true, manually sets the Read node frame range (useful for image sequences). If false, Nuke auto-detects video length.",
                allow_input=False,
                ui_options={"display_name": "Override Input Frame Range"}
            )
        )
        self.add_parameter(
            ParameterInt(
                name="input_first_frame",
                default_value=1,
                tooltip="First frame of the input media (used if Input Override is enabled).",
                allow_input=False,
                ui_options={"display_name": "Input First Frame"}
            )
        )
        self.add_parameter(
            ParameterInt(
                name="input_last_frame",
                default_value=100,
                tooltip="Last frame of the input media (used if Input Override is enabled).",
                allow_input=False,
                ui_options={"display_name": "Input Last Frame"}
            )
        )

        # 2. Output Frame Range Overrides
        self.add_parameter(
            ParameterBool(
                name="override_frame_range",
                default_value=False,
                tooltip="If true, manually set the output render range. If false, it auto-detects from the primary input.",
                allow_input=False,
                ui_options={"display_name": "Override Output Range"}
            )
        )
        self.add_parameter(
            ParameterInt(
                name="output_first_frame",
                default_value=1,
                tooltip="First frame to render (used if Output Override is enabled).",
                allow_input=False,
                ui_options={"display_name": "Output First Frame"}
            )
        )
        self.add_parameter(
            ParameterInt(
                name="output_last_frame",
                default_value=100,
                tooltip="Last frame to render (used if Output Override is enabled).",
                allow_input=False,
                ui_options={"display_name": "Output Last Frame"}
            )
        )

        # 3. Debug Toggle
        self.add_parameter(
            ParameterBool(
                name="save_debug_script",
                default_value=False,
                tooltip="If true, saves the generated Nuke Python script alongside the output for debugging.",
                allow_input=False,
                ui_options={"display_name": "Save Debug Script"}
            )
        )

        # 4. Video Output
        self.add_parameter(
            ParameterVideo(
                name="output_video",
                tooltip="Processed video/sequence saved by Griptape's Project System",
                allow_input=False,
                allow_property=False,
            )
        )

        # 5. Add Collapsed Write Node Settings Group
        write_group_name = "write_node_settings"
        self.add_node_element(
            ParameterGroup(
                name=write_group_name, 
                ui_options={"display_name": "Write Node Settings", "collapsed": True}
            )
        )

        # Define the exact Write node knobs provided
        write_knobs = {
            "channels": ("rgb", ParameterString),
            "proxy": ("", ParameterString),
            "frame_mode": ("expression", ParameterString), 
            "frame": ("", ParameterString),
            "views": ("main", ParameterString),
            "file_type": ("mov", ParameterString),
            "premultiplied": (False, ParameterBool),
            "raw": (False, ParameterBool),
            "transformType": ("colorspace", ParameterString),
            "colorspace": ("default", ParameterString),
            "ocioDisplay": ("default", ParameterString),
            "ocioView": ("sRGB", ParameterString),
            "create_directories": (False, ParameterBool),
            "render_order": (1, ParameterInt),
            "first": (1, ParameterInt),
            "last": (100, ParameterInt),
            "use_limit": (True, ParameterBool),
            "reading": (False, ParameterBool),
            "checkHashOnRead": (True, ParameterBool),
            "on_error": ("error", ParameterString),
            "version": (0, ParameterInt),
            "read_all_lines": (False, ParameterBool),
            "key1": ("", ParameterString),
            "value1": ("", ParameterString),
            "key2": ("", ParameterString),
            "value2": ("", ParameterString),
            "key3": ("", ParameterString),
            "value3": ("", ParameterString),
            "key4": ("", ParameterString),
            "value4": ("", ParameterString),
            "ocioColorspace": ("scene_linear", ParameterString),
            "display": ("default", ParameterString),
            "view": ("sRGB", ParameterString),
            "invert": (False, ParameterBool),
            "saturation": (1.0, ParameterFloat),
            "gain": (1.0, ParameterFloat),
            "gamma": (1.0, ParameterFloat),
            "ocioLayer": ("all", ParameterString),
            "allowGPUAcceleration": (False, ParameterBool),
            "key1_Display": ("", ParameterString),
            "value1_Display": ("", ParameterString),
            "key2_Display": ("", ParameterString),
            "value2_Display": ("", ParameterString),
            "key3_Display": ("", ParameterString),
            "value3_Display": ("", ParameterString),
            "key4_Display": ("", ParameterString),
            "value4_Display": ("", ParameterString),
            "beforeRender": ("", ParameterString),
            "beforeFrameRender": ("", ParameterString),
            "afterFrameRender": ("", ParameterString),
            "afterRender": ("", ParameterString),
            "renderProgress": ("", ParameterString)
        }

        # Dynamically add to group.
        for knob, (default_val, param_class) in write_knobs.items():
            param = param_class(
                name=f"write_{knob}",
                default_value=default_val,
                allow_input=False,
                ui_options={"display_name": knob}
            )
            param.parent_element_name = write_group_name
            self.add_parameter(param)

    def register_nuke_inputs(self, inputs_map: dict[int, str]) -> None:
        self._nuke_image_pipes = inputs_map
        for pipe_idx, name in inputs_map.items():
            param_name = f"nuke_input_{pipe_idx}"
            self.add_parameter(
                Parameter(
                    name=param_name,
                    type="any",
                    input_types=["ImageUrlArtifact", "VideoUrlArtifact", "str", "any"],
                    tooltip=f"Connect media for {name} (Pipe {pipe_idx})\nAccepts Images, Videos, or Sequence Strings (e.g. seq.%04d.exr)",
                    allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                    ui_options={"display_name": f"Input: {name}", "clickable_file_browser": True}
                )
            )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["output_video"] = None

    def _extract_local_path(self, image_input: Any) -> Path:
        input_str = ""
        if isinstance(image_input, str):
            input_str = image_input
        elif isinstance(image_input, dict):
            input_str = image_input.get("value", "")
            if not input_str and "location" in image_input:
                input_str = image_input["location"]
        elif hasattr(image_input, "value"):
            input_str = str(image_input.value)

        if not input_str:
            raise ValueError("Empty input path.")
            
        if "{" in input_str and "}" in input_str:
            raise ValueError(
                f"Received an unresolved macro template instead of a real file path:\n'{input_str}'\n\n"
                "Check your node connections! Make sure you connected the artifact output "
                "from the previous node, NOT its file configuration parameter."
            )

        if input_str.startswith("file://"):
            parsed = urllib.parse.urlparse(input_str)
            return Path(urllib.request.url2pathname(parsed.path)).absolute()
            
        elif input_str.startswith("http"):
            self._log(f"Downloading remote/localhost media: {input_str}")
            try:
                response = requests.get(input_str, timeout=30)
                response.raise_for_status()
                
                parsed_url = urllib.parse.urlparse(input_str)
                ext = Path(parsed_url.path).suffix.lower()
                video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
                
                if ext in video_extensions:
                    self._log(f"Detected video file ({ext}), bypassing PIL re-encoding.")
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                    temp_file.write(response.content)
                    temp_file.close()
                else:
                    img = Image.open(BytesIO(response.content))
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGBA')
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img.save(temp_file, format="PNG")
                    temp_file.close()
                
                self._temp_downloads.append(temp_file.name)
                return Path(temp_file.name).absolute()
                
            except Exception as e:
                 raise ValueError(f"Failed to download or parse media from {input_str}: {e}")

        return Path(input_str).absolute()

    def _generate_nuke_script(self, debug_nk_path: str = None) -> str:
        # Ignore our execution parameters
        ignore_params = {
            "output_video", 
            "save_debug_script", 
            "override_frame_range", 
            "output_first_frame", 
            "output_last_frame",
            "override_input_frame_range",
            "input_first_frame",
            "input_last_frame",
            "write_file_type"
        }

        # Extract variables
        override_out_range = self.get_parameter_value("override_frame_range")
        out_first_val = self.get_parameter_value("output_first_frame") or 1
        out_last_val = self.get_parameter_value("output_last_frame") or 100
        
        override_in_range = self.get_parameter_value("override_input_frame_range")
        in_first_val = self.get_parameter_value("input_first_frame") or 1
        in_last_val = self.get_parameter_value("input_last_frame") or 100

        script_lines = [
            "import nuke",
            "import sys",
            "import json",
            "",
            "out_file = sys.argv[1]",
            "inputs_json = sys.argv[2]",
            "",
            "input_map = {int(k): v for k, v in json.loads(inputs_json).items() if v}",
            "",
            f"target_node = nuke.nodes.{self.nuke_node_class}()",
            "write_node = nuke.nodes.Write(inputs=[target_node], file=out_file)",
            "",
            "first_frame = 1",
            "last_frame = 1",
            "",
            "for pipe_idx, filepath in input_map.items():",
            "    read_node = nuke.nodes.Read(file=filepath)",
            "    try:",
            "        # Safeguard: Instruct the Read node to hold the nearest frame instead of crashing if it overshoots",
            "        read_node['on_error'].setValue('nearest frame')",
            "    except:",
            "        pass",
            "",
            f"    if {str(override_in_range)}:",
            f"        read_node['first'].setValue({in_first_val})",
            f"        read_node['last'].setValue({in_last_val})",
            "",
            "    target_node.setInput(pipe_idx, read_node)",
            "    if pipe_idx == min(input_map.keys()):",
            "        first_frame = read_node.firstFrame()",
            "        last_frame = read_node.lastFrame()",
            ""
        ]

        for param in self.parameters:
            if param.name in ignore_params or param.name.startswith("nuke_input_"):
                continue

            val = self.get_parameter_value(param.name)
            if val is None or val == "":
                continue

            is_write_knob = param.name.startswith("write_")
            target_obj = "write_node" if is_write_knob else "target_node"
            knob_name = param.name.replace("write_", "", 1) if is_write_knob else param.name

            if isinstance(val, str):
                script_lines.append(f"{target_obj}['{knob_name}'].fromScript({repr(val)})")
            else:
                script_lines.append(f"try:")
                script_lines.append(f"    {target_obj}['{knob_name}'].setValue({val})")
                script_lines.append(f"except TypeError:")
                val_str = str(val).lower() if isinstance(val, bool) else str(val)
                script_lines.append(f"    {target_obj}['{knob_name}'].fromScript({repr(val_str)})")

        script_lines.extend([
            "",
            "# --- HARDCODE FILE TYPE AND CODEC ---",
            "write_node['file_type'].setValue('mov')",
            "try:",
            "    # Index 4 corresponds to H.264 based on the Nuke UI dropdown",
            "    write_node['mov64_codec'].setValue(4)",
            "except Exception as e:",
            "    print(f'Failed to set codec: {e}')",
            "    pass",
            "",
            f"if {str(override_out_range)}:",
            f"    first_frame = {out_first_val}",
            f"    last_frame = {out_last_val}",
            "elif write_node['use_limit'].getValue():",
            "    first_frame = int(write_node['first'].getValue())",
            "    last_frame = int(write_node['last'].getValue())",
            "",
            "nuke.Root()['first_frame'].setValue(first_frame)",
            "nuke.Root()['last_frame'].setValue(last_frame)",
            "nuke.Root()['fps'].setValue(24)"
        ])

        if debug_nk_path:
            nk_path_fwd = Path(debug_nk_path).as_posix()
            script_lines.extend([
                "",
                "# Save the fully assembled Nuke script to disk for debugging",
                f"nuke.scriptSaveAs('{nk_path_fwd}')"
            ])

        script_lines.extend([
            "",
            "print(f'Rendering frames {first_frame} to {last_frame}...')",
            "nuke.execute(write_node, first_frame, last_frame)"
        ])

        return "\n".join(script_lines)

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        self._set_safe_defaults()
        self._temp_downloads = []
        debug_nk_temp_path = None

        if not self.nuke_node_class:
            raise ValueError(f"{self.name}: 'nuke_node_class' was not defined.")

        nuke_exe = GriptapeNodes.SecretsManager().get_secret(self.NUKE_SECRET_NAME)
        if not nuke_exe:
            self._log(f"Secret '{self.NUKE_SECRET_NAME}' not found. Falling back to system PATH 'Nuke'.")
            nuke_exe = "Nuke"
        
        input_map = {}
        for pipe_idx in self._nuke_image_pipes.keys():
            val = self.get_parameter_value(f"nuke_input_{pipe_idx}")
            if val:
                img_path = self._extract_local_path(val)
                
                if not img_path.exists() and "%" not in img_path.name and "#" not in img_path.name:
                    raise FileNotFoundError(f"{self.name}: Input file does not exist at {img_path}")
                
                input_map[pipe_idx] = img_path.as_posix()

        if not input_map:
            raise ValueError(f"{self.name}: At least one input media must be provided.")

        try:
            with tempfile.NamedTemporaryFile(suffix='.mov', delete=False) as temp_out_file:
                temp_output_path = Path(temp_out_file.name)
            
            if temp_output_path.exists():
                os.remove(temp_output_path)

            save_debug = self.get_parameter_value("save_debug_script")
            if save_debug:
                with tempfile.NamedTemporaryFile(suffix='.nk', delete=False) as nk_tmp:
                    debug_nk_temp_path = nk_tmp.name
                
                if os.path.exists(debug_nk_temp_path):
                    os.remove(debug_nk_temp_path)

            inputs_json_str = json.dumps(input_map)
            nuke_script_content = self._generate_nuke_script(debug_nk_path=debug_nk_temp_path)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_file.write(nuke_script_content)
                script_path = script_file.name

            nuke_cmd = [
                nuke_exe,
                "-t", 
                script_path,
                temp_output_path.as_posix(),
                inputs_json_str
            ]

            self._log(f"Launching Nuke: {' '.join(nuke_cmd)}")

            result = subprocess.run(
                nuke_cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                error_msg = f"Nuke failed (Code {result.returncode}).\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                raise RuntimeError(error_msg)

            if not temp_output_path.exists() or temp_output_path.stat().st_size == 0:
                raise FileNotFoundError(f"{self.name}: Nuke finished successfully, but output file is missing at {temp_output_path}")

            output_bytes = temp_output_path.read_bytes()
            
            # Save it officially as an MOV
            dest = ProjectFileDestination.from_situation(
                filename=f"{self.nuke_node_class.lower()}_result.mov",
                situation="save_node_output"
            )
            saved = dest.write_bytes(output_bytes)
            
            self.parameter_output_values["output_video"] = VideoUrlArtifact(saved.location)

            # --- Save Debug Files ---
            if save_debug:
                try:
                    py_dest = ProjectFileDestination.from_situation(
                        filename=f"{self.nuke_node_class.lower()}_debug.py",
                        situation="save_node_output"
                    )
                    py_dest.write_bytes(nuke_script_content.encode('utf-8'))
                    
                    if debug_nk_temp_path and Path(debug_nk_temp_path).exists():
                        nk_dest = ProjectFileDestination.from_situation(
                            filename=f"{self.nuke_node_class.lower()}_debug.nk",
                            situation="save_node_output"
                        )
                        nk_bytes = Path(debug_nk_temp_path).read_bytes()
                        nk_saved = nk_dest.write_bytes(nk_bytes)
                        self._log(f"Saved debug .nk script to: {nk_saved.location}")
                except Exception as e:
                    self._log(f"Failed to save debug scripts: {e}")

        except Exception as e:
            self._set_safe_defaults()
            self._log(f"Nuke processing failed: {e}")
            raise RuntimeError(f"{self.name}: {str(e)}") from e
            
        finally:
            if 'script_path' in locals() and os.path.exists(script_path):
                os.remove(script_path)
            if 'temp_output_path' in locals() and temp_output_path.exists():
                os.remove(temp_output_path)
            if debug_nk_temp_path and os.path.exists(debug_nk_temp_path):
                os.remove(debug_nk_temp_path)
            for tmp_file in self._temp_downloads:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)


# =====================================================================
# SPECIFIC NUKE NODES
# =====================================================================

class NukeKronos(BaseNukeNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nuke_node_class = "Kronos"

        self.register_nuke_inputs({
            0: "Source",
            1: "Matte",
            2: "Motion",
            3: "Foreground"
        })

        self.add_parameter(ParameterBool(name="useGPUIfAvailable", default_value=True, allow_input=False))
        self.add_parameter(ParameterInt(name="input.first", default_value=1, allow_input=False))
        self.add_parameter(ParameterInt(name="input.last", default_value=100, allow_input=False))
        self.add_parameter(ParameterString(name="retimedChannels", default_value="all", allow_input=False))
        self.add_parameter(ParameterString(name="interpolation", default_value="Motion", allow_input=False))
        self.add_parameter(ParameterString(name="timing2", default_value="Output Speed", allow_input=False))
        self.add_parameter(ParameterFloat(name="timingOutputSpeed", default_value=0.5, allow_input=False))
        self.add_parameter(ParameterFloat(name="timingInputSpeed", default_value=0.5, allow_input=False))
        self.add_parameter(ParameterInt(name="timingFrame2", default_value=1, allow_input=False))
        self.add_parameter(ParameterString(name="motionEstimation", default_value="Regularized", allow_input=False))
        self.add_parameter(ParameterFloat(name="vectorDetailLocal", default_value=0.2, allow_input=False))
        self.add_parameter(ParameterFloat(name="smoothnessLocal", default_value=0.5, allow_input=False))
        self.add_parameter(ParameterInt(name="vectorSourceFlag", default_value=1, allow_input=False))
        self.add_parameter(ParameterInt(name="warpSourceFlag", default_value=1, allow_input=False))
        self.add_parameter(ParameterInt(name="computedVectorFlag", default_value=1, allow_input=False))
        self.add_parameter(ParameterFloat(name="vectorDetailReg", default_value=0.3, allow_input=False))
        self.add_parameter(ParameterFloat(name="strengthReg", default_value=1.5, allow_input=False))
        self.add_parameter(ParameterString(name="resampleType", default_value="Bilinear", allow_input=False))
        self.add_parameter(ParameterFloat(name="Shutter", default_value=0.0, allow_input=False))
        self.add_parameter(ParameterInt(name="shutterSamples", default_value=1, allow_input=False))
        self.add_parameter(ParameterFloat(name="shutterTime", default_value=0.0, allow_input=False))
        self.add_parameter(ParameterBool(name="autoShutterTime", default_value=False, allow_input=False))
        self.add_parameter(ParameterString(name="output", default_value="Result", allow_input=False))
        self.add_parameter(ParameterString(name="matteChannel", default_value="None", allow_input=False))
        self.add_parameter(ParameterInt(name="maskFlag", default_value=1, allow_input=False))
        self.add_parameter(ParameterInt(name="Advanced", default_value=1, allow_input=False))
        self.add_parameter(ParameterBool(name="flickerCompensation", default_value=False, allow_input=False))
        self.add_parameter(ParameterBool(name="showLegacyMode", default_value=False, allow_input=False))
        self.add_parameter(ParameterBool(name="legacyModeNuke9", default_value=False, allow_input=False))
        self.add_parameter(ParameterInt(name="Tolerances", default_value=0, allow_input=False))
        self.add_parameter(ParameterFloat(name="weightRed", default_value=0.3, allow_input=False))
        self.add_parameter(ParameterFloat(name="weightGreen", default_value=0.6, allow_input=False))
        self.add_parameter(ParameterFloat(name="weightBlue", default_value=0.1, allow_input=False))
        self.add_parameter(ParameterInt(name="vectorSpacing", default_value=20, allow_input=False))
        self.add_parameter(ParameterBool(name="showVectors", default_value=False, allow_input=False))


class NukeKeyer(BaseNukeNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nuke_node_class = "Keyer"

        self.register_nuke_inputs({
            0: "Source"
        })

        self.add_parameter(ParameterString(name="input", default_value="rgb", allow_input=False))
        self.add_parameter(ParameterString(name="output", default_value="rgba.alpha", allow_input=False))
        self.add_parameter(ParameterString(name="combine", default_value="replace", allow_input=False))
        self.add_parameter(ParameterBool(name="invert", default_value=False, allow_input=False))
        self.add_parameter(ParameterString(name="operation", default_value="luminance key", allow_input=False))
        self.add_parameter(ParameterString(name="range0", default_value="1 1 1", allow_input=False))


class NukeBokeh(BaseNukeNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nuke_node_class = "bokeh"

        self.register_nuke_inputs({
            0: "Image",
            1: "Depth",
            2: "Filter",
            3: "Mask"
        })

        self.add_parameter(ParameterString(name="bokehChannels", default_value="rgba", allow_input=False))
        self.add_parameter(ParameterString(name="depthChannel", default_value="depth.Z", allow_input=False))
        self.add_parameter(ParameterString(name="outputType", default_value="Defocused Image", allow_input=False))
        self.add_parameter(ParameterString(name="focusOutput", default_value="Both", allow_input=False))
        self.add_parameter(ParameterString(name="multiplier", default_value="%bokehmult", allow_input=False))
        self.add_parameter(ParameterFloat(name="frontmultiplier", default_value=1.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="backmultiplier", default_value=1.0, allow_input=False))
        self.add_parameter(ParameterString(name="depthStyle", default_value="Real", allow_input=False))
        self.add_parameter(ParameterFloat(name="focalPlane", default_value=100.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="focusRegionSize", default_value=1.0, allow_input=False))
        self.add_parameter(ParameterString(name="focusRegionFalloff", default_value="Linear", allow_input=False))
        self.add_parameter(ParameterString(name="defocusMatteChannel", default_value="none", allow_input=False))
        self.add_parameter(ParameterBool(name="normalizeVisualization", default_value=True, allow_input=False))
        self.add_parameter(ParameterBool(name="integrateFrontAndBackSeperately", default_value=False, allow_input=False))
        self.add_parameter(ParameterBool(name="realWorldLens", default_value=False, allow_input=False))
        self.add_parameter(ParameterFloat(name="focalLength", default_value=35.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="fStop", default_value=6.0, allow_input=False))
        self.add_parameter(ParameterString(name="worldScale", default_value="cm", allow_input=False))
        self.add_parameter(ParameterFloat(name="worldScaleMultiplier", default_value=1.0, allow_input=False))
        self.add_parameter(ParameterString(name="filmFormat", default_value="35mm", allow_input=False))
        self.add_parameter(ParameterFloat(name="apertureWidth", default_value=22.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="apertureHeight", default_value=16.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="bloom", default_value=0.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="bloomCurvature", default_value=2.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="bloomThreshold", default_value=0.7, allow_input=False))
        self.add_parameter(ParameterString(name="bloomMatteChannel", default_value="none", allow_input=False))
        self.add_parameter(ParameterFloat(name="kSphAbb", default_value=0.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="kChrAbb", default_value=0.0, allow_input=False))
        self.add_parameter(ParameterString(name="kChrAbbOff0", default_value="0.6000000238 1 1", allow_input=False))
        self.add_parameter(ParameterBool(name="kChrAbbOff_panelDropped", default_value=False, allow_input=False))
        self.add_parameter(ParameterString(name="chromicAberrationMatteChannel", default_value="none", allow_input=False))
        self.add_parameter(ParameterString(name="kernelType", default_value="Circular", allow_input=False))
        self.add_parameter(ParameterInt(name="max_kernelsize", default_value=256, allow_input=False))
        self.add_parameter(ParameterBool(name="expandBBoxToMaxKernelSize", default_value=False, allow_input=False))
        self.add_parameter(ParameterString(name="missingKernelChannel", default_value="none", allow_input=False))
        self.add_parameter(ParameterFloat(name="kAspectRatio", default_value=1.0, allow_input=False))
        self.add_parameter(ParameterFloat(name="kSoftness", default_value=0.1, allow_input=False))
        self.add_parameter(ParameterInt(name="kNumSides", default_value=6, allow_input=False))
        self.add_parameter(ParameterFloat(name="kCurvature", default_value=0.1, allow_input=False))
        self.add_parameter(ParameterFloat(name="kRotation", default_value=0.0, allow_input=False))
        self.add_parameter(ParameterInt(name="correctiveSlices", default_value=10, allow_input=False))
        self.add_parameter(ParameterBool(name="overrideNearAndFarForSlices", default_value=False, allow_input=False))
        self.add_parameter(ParameterString(name="depthSlicesNearAndFar0", default_value="0 0", allow_input=False))
import os
import subprocess
import tempfile
import shutil
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

# Import PIL at the module level per Griptape best practices
from PIL import Image

from griptape.artifacts import ImageArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import ControlNode, AsyncResult
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger(__name__)


class UsdNukeInspector(ControlNode):
    """Generates a 2x2 turntable grid screenshot from a USD file using Nuke's headless USD environment (Flat Profile)."""
    
    # Must match the secret defined in griptape_nodes_library.json
    NUKE_SECRET_NAME = "NUKE_BIN_PATH"

    def __init__(self, name: str = "USD Nuke Inspector", metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # 1. Input: USD File Path
        self.add_parameter(
            Parameter(
                name="usd_file",
                tooltip="Path to the input USD file",
                input_types=["str"],
                type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "clickable_file_browser": True,
                    "display_name": "USD File Path"
                }
            )
        )

        # 2. Output: Rendered Image for the Vision Agent
        # Replaced ParameterImage with a generic Parameter to explicitly set output_type to ImageArtifact
        self.add_parameter(
            Parameter(
                name="output_image",
                tooltip="2x2 Rendered screenshot grid of the USD model (embedded base64)",
                output_type="ImageArtifact",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # 3. Output: Folder Path
        self.add_parameter(
            Parameter(
                name="output_folder",
                tooltip="Directory path where the screenshot was saved within the Griptape workspace",
                output_type=ParameterTypeBuiltin.STR.value,
                allowed_modes={ParameterMode.OUTPUT},
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
        self.parameter_output_values["output_image"] = None
        self.parameter_output_values["output_folder"] = None

        # 1. Fetch Input USD Path
        usd_path = self.get_parameter_value("usd_file")
        
        # --- EAGER EXECUTION GUARD ---
        # If the incoming string is not a USD file, silently skip it.
        # This prevents Nuke from crashing during eager data-flow evaluation.
        if isinstance(usd_path, str) and not usd_path.lower().endswith((".usd", ".usda", ".usdc")):
            self._log(f"Skipped: '{usd_path}' is not a USD file.")
            return
        # -----------------------------

        # 2. Validate Input USD Exists
        if not usd_path or not os.path.exists(str(usd_path)):
            raise ValueError(f"{self.name}: USD file not found at path: {usd_path}")

        # 3. Validate Secret (Nuke Executable Path)
        nuke_exec = GriptapeNodes.SecretsManager().get_secret(self.NUKE_SECRET_NAME)
        if not nuke_exec or not os.path.exists(nuke_exec):
            raise ValueError(
                f"{self.name}: Nuke executable not found. Please configure the "
                f"'{self.NUKE_SECRET_NAME}' secret in Griptape settings. "
                f"Current value: {nuke_exec}"
            )

        safe_usd_path = str(usd_path).replace("\\", "/")

        # 4. Setup a temporary directory to hold the 4 rendered frames
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Generate paths for the 4 intermediate renders
            out_paths = [
                os.path.join(temp_dir, f"frame_{i}.png").replace("\\", "/")
                for i in range(4)
            ]
            
            # Format the Python list as a string so we can inject it into the Nuke script
            out_paths_str = "[" + ", ".join([f'r"{p}"' for p in out_paths]) + "]"

            # The script we will inject into Nuke's headless Python environment
            nuke_python_script = f"""
import math
from pxr import Usd, UsdGeom, UsdAppUtils, Gf

usd_path = r"{safe_usd_path}"
out_paths = {out_paths_str}

try:
    # Open USD Stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError("Failed to open USD stage.")

    # Compute Bounding Box
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.EarliestTime(), ['default', 'proxy', 'render'])
    bbox = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot())
    bounds = bbox.ComputeAlignedRange()

    if bounds.IsEmpty():
        min_pt, max_pt = Gf.Vec3d(0, 0, 0), Gf.Vec3d(0, 0, 0)
    else:
        min_pt = bounds.GetMin()
        max_pt = bounds.GetMax()

    center = (min_pt + max_pt) / 2.0
    size = max((max_pt[0] - min_pt[0]), (max_pt[1] - min_pt[1]), (max_pt[2] - min_pt[2]))
    radius = size / 2.0

    # Create a Temporary Camera
    stage.SetEditTarget(stage.GetSessionLayer())
    camera = UsdGeom.Camera.Define(stage, "/_TempRenderCamera")
    camera.CreateFocalLengthAttr().Set(50.0)
    xform_op = camera.AddTransformOp()

    fov_radians = math.radians(40.0)
    distance = (radius / math.tan(fov_radians / 2.0)) * 1.5

    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis == UsdGeom.Tokens.z:
        up_dir = Gf.Vec3d(0, 0, 1)
        base_dir = Gf.Vec3d(0, -distance, 0)
    else:
        up_dir = Gf.Vec3d(0, 1, 0)
        base_dir = Gf.Vec3d(0, 0, distance)

    recorder = UsdAppUtils.FrameRecorder()
    recorder.SetImageWidth(512) # 512x512 per frame = 1024x1024 total grid

    # Angles for the 4 views (flat horizontal offset, no elevation)
    angles = [45, 135, 225, 315]

    for i, angle in enumerate(angles):
        rot = Gf.Rotation(up_dir, angle)
        
        # Calculate new position: perfectly flat rotation around center
        cam_pos = center + rot.TransformDir(base_dir)
        
        # Orient camera to look at the center perfectly level.
        view_mat = Gf.Matrix4d().SetLookAt(cam_pos, center, up_dir)
        xform_op.Set(view_mat.GetInverse())

        # Render current angle
        recorder.Record(stage, camera, Usd.TimeCode.EarliestTime(), out_paths[i])
    
except Exception as e:
    import traceback
    print("NUKE_USD_ERROR:")
    traceback.print_exc()
    import sys
    sys.exit(1)
"""

            script_path = os.path.join(temp_dir, "render_script.py")
            with open(script_path, 'w') as f:
                f.write(nuke_python_script)

            # 5. Execute Headless Nuke
            process_result = subprocess.run(
                [nuke_exec, "-t", script_path],
                capture_output=True,
                text=True
            )

            # Check for our custom error string or a bad exit code
            if process_result.returncode != 0 or "NUKE_USD_ERROR:" in process_result.stdout:
                self._log(f"Nuke Output: {process_result.stdout}")
                self._log(f"Nuke Error: {process_result.stderr}")
                raise RuntimeError(f"{self.name}: Nuke failed to render USD grid. Check node logs for trace.")

            # Validate all 4 frames rendered successfully
            for p in out_paths:
                if not os.path.exists(p) or os.path.getsize(p) == 0:
                    raise RuntimeError(f"{self.name}: Nuke ran successfully but failed to create frame {p}.")

            # 6. Stitch the 4 frames into a 2x2 grid using PIL
            images = [Image.open(p) for p in out_paths]
            w, h = images[0].size
            
            grid_img = Image.new('RGB', (w * 2, h * 2))
            grid_img.paste(images[0], (0, 0))      # Top-Left (45 deg)
            grid_img.paste(images[1], (w, 0))      # Top-Right (135 deg)
            grid_img.paste(images[2], (0, h))      # Bottom-Left (225 deg)
            grid_img.paste(images[3], (w, h))      # Bottom-Right (315 deg)

            # Save the stitched grid to the temp folder
            grid_temp_path = os.path.join(temp_dir, "final_grid.png")
            grid_img.save(grid_temp_path)

            # 7. Import the final grid into Griptape Project System
            with open(grid_temp_path, 'rb') as f:
                output_bytes = f.read()
                
            base_name = os.path.splitext(os.path.basename(str(usd_path)))[0]
            dest = ProjectFileDestination.from_situation(
                filename=f"{base_name}_turntable.png",
                situation="save_node_output"
            )
            saved = dest.write_bytes(output_bytes)

            # 8. Set Final Outputs - Output an ImageArtifact to embed raw base64 data
            self.parameter_output_values["output_image"] = ImageArtifact(
                value=output_bytes,
                format="png",
                width=w * 2,
                height=h * 2
            )
            self.parameter_output_values["output_folder"] = os.path.dirname(saved.location)

        finally:
            # 9. Clean up the entire temporary directory (all frames + script)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
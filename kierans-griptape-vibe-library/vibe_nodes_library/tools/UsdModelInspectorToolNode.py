import math
import os
import shutil
import subprocess
import tempfile
from typing import Any

from PIL import Image
from schema import Literal, Schema

from griptape.artifacts import ErrorArtifact, ImageArtifact
from griptape.tools import BaseTool as GtBaseTool
from griptape.utils.decorators import activity
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

# ---------------------------------------------------------------------------
# SDK Tool  ––  the Griptape agent-facing tool
# ---------------------------------------------------------------------------

class UsdModelInspectorSdkTool(GtBaseTool):
    """Griptape SDK Tool that renders a 2x2 turntable grid from a USD file
    using Nuke's headless USD environment and returns the result as an
    ImageArtifact so a vision-capable agent can inspect it."""

    # Must match the secret defined in griptape_nodes_library.json
    NUKE_SECRET_NAME = "NUKE_BIN_PATH"

    @activity(
        config={
            "description": (
                "Renders a 2x2 turntable screenshot grid from a USD (.usd / .usda / .usdc) "
                "file using Nuke's headless USD renderer and returns the composite image. "
                "Use this to visually inspect the geometry, scale, and general appearance "
                "of a 3-D USD model."
            ),
            "schema": Schema(
                {
                    Literal(
                        "usd_file_path",
                        description=(
                            "The absolute path to the USD file (.usd, .usda, or .usdc) "
                            "that should be inspected."
                        ),
                    ): str
                }
            ),
        }
    )
    def inspect_usd_model(self, params: dict) -> ImageArtifact | ErrorArtifact:
        """Render four turntable views of the USD model and return a stitched 2x2 grid."""
        usd_path: str = params["values"]["usd_file_path"]

        # ── Validate file extension ──────────────────────────────────────────
        if not usd_path.lower().endswith((".usd", ".usda", ".usdc")):
            return ErrorArtifact(
                f"The path '{usd_path}' does not point to a USD file. "
                "Please provide a .usd, .usda, or .usdc file."
            )

        # ── Validate file exists ─────────────────────────────────────────────
        if not os.path.exists(usd_path):
            return ErrorArtifact(f"USD file not found at path: {usd_path}")

        # ── Validate Nuke executable ─────────────────────────────────────────
        nuke_exec = GriptapeNodes.SecretsManager().get_secret(self.NUKE_SECRET_NAME)
        if not nuke_exec or not os.path.exists(nuke_exec):
            return ErrorArtifact(
                f"Nuke executable not found. Please configure the "
                f"'{self.NUKE_SECRET_NAME}' secret in Griptape settings. "
                f"Current value: {nuke_exec}"
            )

        safe_usd_path = usd_path.replace("\\", "/")
        temp_dir = tempfile.mkdtemp()

        try:
            # ── Build per-frame output paths ─────────────────────────────────
            out_paths = [
                os.path.join(temp_dir, f"frame_{i}.png").replace("\\", "/")
                for i in range(4)
            ]
            out_paths_str = "[" + ", ".join([f'r"{p}"' for p in out_paths]) + "]"

            # ── Nuke headless Python script ───────────────────────────────────
            nuke_python_script = f"""
import math
from pxr import Usd, UsdGeom, UsdAppUtils, Gf

usd_path = r"{safe_usd_path}"
out_paths = {out_paths_str}

try:
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise RuntimeError("Failed to open USD stage.")

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
    recorder.SetImageWidth(512)

    angles = [45, 135, 225, 315]
    for i, angle in enumerate(angles):
        rot = Gf.Rotation(up_dir, angle)
        cam_pos = center + rot.TransformDir(base_dir)
        view_mat = Gf.Matrix4d().SetLookAt(cam_pos, center, up_dir)
        xform_op.Set(view_mat.GetInverse())
        recorder.Record(stage, camera, Usd.TimeCode.EarliestTime(), out_paths[i])

except Exception as e:
    import traceback
    print("NUKE_USD_ERROR:")
    traceback.print_exc()
    import sys
    sys.exit(1)
"""

            script_path = os.path.join(temp_dir, "render_script.py")
            with open(script_path, "w") as f:
                f.write(nuke_python_script)

            # ── Execute headless Nuke ─────────────────────────────────────────
            result = subprocess.run(
                [nuke_exec, "-t", script_path],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0 or "NUKE_USD_ERROR:" in result.stdout:
                return ErrorArtifact(
                    f"Nuke failed to render USD grid.\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )

            # ── Validate all 4 frames ────────────────────────────────────────
            for p in out_paths:
                if not os.path.exists(p) or os.path.getsize(p) == 0:
                    return ErrorArtifact(
                        f"Nuke ran successfully but failed to create frame: {p}"
                    )

            # ── Stitch 2x2 grid ──────────────────────────────────────────────
            images = [Image.open(p) for p in out_paths]
            w, h = images[0].size

            grid_img = Image.new("RGB", (w * 2, h * 2))
            grid_img.paste(images[0], (0, 0))   # 45°  – top-left
            grid_img.paste(images[1], (w, 0))   # 135° – top-right
            grid_img.paste(images[2], (0, h))   # 225° – bottom-left
            grid_img.paste(images[3], (w, h))   # 315° – bottom-right

            grid_temp_path = os.path.join(temp_dir, "final_grid.png")
            grid_img.save(grid_temp_path)

            with open(grid_temp_path, "rb") as f:
                output_bytes = f.read()

            return ImageArtifact(
                value=output_bytes,
                format="png",
                width=w * 2,
                height=h * 2,
            )

        except Exception as exc:
            return ErrorArtifact(f"Unexpected error during USD inspection: {exc}")

        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# DataNode wrapper  ––  the Griptape Nodes UI node
# ---------------------------------------------------------------------------

class UsdModelInspectorToolNode(DataNode):
    """Griptape Node that exposes the UsdModelInspectorSdkTool to an Agent node."""

    def __init__(self, name: str = "USD Model Inspector Tool", metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Informational banner
        self.add_node_element(
            ParameterMessage(
                name="tool_info",
                variant="info",
                value=(
                    "Gives an agent the ability to render and visually inspect USD models "
                    "via Nuke's headless USD renderer. Requires the NUKE_BIN_PATH secret."
                ),
            )
        )

        # Off-prompt toggle
        self.add_parameter(
            Parameter(
                name="off_prompt",
                input_types=["bool"],
                type="bool",
                default_value=False,
                tooltip=(
                    "When enabled the rendered image is stored as a memory artifact "
                    "rather than shown inline in the agent's response."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Off Prompt"},
            )
        )

        # Tool output — connect this to an Agent node's "tools" input
        self.add_parameter(
            Parameter(
                name="tool",
                output_type="Tool",
                tooltip="The USD Model Inspector Tool ready to be connected to an Agent.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def process(self) -> None:
        """Instantiate the SDK tool and expose it as an output."""
        off_prompt = self.get_parameter_value("off_prompt") or False
        tool = UsdModelInspectorSdkTool(off_prompt=bool(off_prompt))
        self.parameter_output_values["tool"] = tool

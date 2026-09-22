import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin, ParameterMessage
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.clamp import Clamp
from griptape_nodes.traits.color_picker import ColorPicker
from griptape_nodes.traits.options import Options
from PIL import Image, UnidentifiedImageError

from griptape_nodes_library.utils.file_utils import generate_filename
from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor
from griptape_nodes_library.utils.image_utils import (
    DEFAULT_PLACEHOLDER_HEIGHT,
    DEFAULT_PLACEHOLDER_WIDTH,
    cleanup_temp_files,
    create_background_image,
    create_grid_layout,
    create_masonry_layout,
    create_placeholder_image,
    image_to_bytes,
)

class ExtractFramesToGrid(BaseVideoProcessor):
    """Extract every Nth frame from a video and display them in an image grid."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Hide base video parameters we don't need for this specific operation
        self.hide_parameter_by_name("output_frame_rate")
        self.hide_parameter_by_name("processing_speed")
        self.hide_parameter_by_name("output")

        # Nth frame parameter
        self.nth_frame = ParameterInt(
            name="nth_frame",
            default_value=30,
            tooltip="Extract every Nth frame from the video",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Extract Every Nth Frame", "step": 1},
        )
        self.nth_frame.add_trait(Clamp(min_val=1, max_val=10000))
        self.add_parameter(self.nth_frame)

        # UI Info Message
        self.frame_range_message = ParameterMessage(
            name="frame_range_message",
            variant="info",
            value="Connect a video to calculate frames.",
        )
        self.add_node_element(self.frame_range_message)

        # --- Grid Parameters (Adapted from DisplayImageGrid) ---
        self.layout_style = ParameterString(
            name="layout_style",
            default_value="grid",
            tooltip="Layout style: 'grid' for uniform tiles, 'masonry' for variable heights",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.layout_style.add_trait(Options(choices=["grid", "masonry"]))
        self.add_parameter(self.layout_style)

        self.grid_justification = ParameterString(
            name="grid_justification",
            default_value="left",
            tooltip="How to justify images in grid layout (grid layout only)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.grid_justification.add_trait(Options(choices=["left", "center", "right"]))
        self.add_parameter(self.grid_justification)

        self.columns = ParameterInt(
            name="columns",
            default_value=4,
            tooltip="Number of columns in the grid",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"slider": {"min_val": 1, "max_val": 10, "step": 1}},
        )
        self.columns.add_trait(Clamp(min_val=1, max_val=10))
        self.add_parameter(self.columns)

        self.add_parameter(
            ParameterInt(
                name="spacing",
                default_value=10,
                tooltip="Spacing between images in pixels",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"slider": {"min_val": 0, "max_val": 100, "step": 1}},
            )
        )

        self.border_radius = ParameterInt(
            name="border_radius",
            default_value=8,
            tooltip="Border radius for rounded corners (0 for square)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"slider": {"min_val": 0, "max_val": 500, "step": 1}},
        )
        self.add_parameter(self.border_radius)

        self.crop_to_fit = ParameterBool(
            name="crop_to_fit",
            default_value=True,
            tooltip="Crop images to fit perfectly within the grid/masonry for clean borders",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(self.crop_to_fit)

        self.transparent_bg = ParameterBool(
            name="transparent_bg",
            default_value=False,
            tooltip="Use transparent background instead of solid color",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(self.transparent_bg)

        self.background_color = ParameterString(
            name="background_color",
            default_value="#000000",
            tooltip="Background color of the grid (hex color)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={ColorPicker(format="hex")},
        )
        self.add_parameter(self.background_color)

        self.output_image_size = ParameterString(
            name="output_image_size",
            default_value="custom",
            tooltip="Use custom width or preset sizes",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.output_image_size.add_trait(Options(choices=["custom", "preset"]))
        self.add_parameter(self.output_image_size)

        self.output_preset = ParameterString(
            name="output_preset",
            default_value="1080p (1920x1080)",
            tooltip="Preset output size",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"hide": True},
        )
        self.output_preset.add_trait(
            Options(choices=["4K (3840x2160)", "1440p (2560x1440)", "1080p (1920x1080)", "720p (1280x720)"])
        )
        self.add_parameter(self.output_preset)

        self.output_image_width = ParameterInt(
            name="output_image_width",
            default_value=1200,
            tooltip="Maximum width of the output image in pixels",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(self.output_image_width)

        self.output_format = ParameterString(
            name="output_format",
            default_value="png",
            tooltip="Output format for the generated image grid",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.output_format.add_trait(Options(choices=["png", "jpeg", "webp"]))
        self.add_parameter(self.output_format)

        self.grid_output = ParameterImage(
            name="grid_output",
            default_value=None,
            tooltip="Generated image grid containing the extracted frames",
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"pulse_on_run": True, "expander": True},
        )
        self.add_parameter(self.grid_output)

    def _setup_custom_parameters(self) -> None:
        """Setup custom parameters specific to this video processor."""
        pass

    # --- Required Abstract Methods ---
    def _get_processing_description(self) -> str:
        """Get a description of what this processor does."""
        return "extracting frames and assembling grid layout"

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get the output filename suffix."""
        return "_frames_grid"

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:  # noqa: ARG002
        """Build the FFmpeg command for extracting frames into a directory pattern."""
        ffmpeg_path, _ = self._get_ffmpeg_paths()
        nth_frame = self.get_parameter_value("nth_frame") or 30

        cmd = [
            ffmpeg_path,
            "-i", input_url,
            "-vf", f"select=not(mod(n\\,{nth_frame}))",
            "-vsync", "vfr",
            "-q:v", "2",
            output_path
        ]
        return cmd
    # ---------------------------------

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle UI visibility changes for grid settings and video updates."""
        # Grid layout UI logic
        if parameter.name == "layout_style":
            if value == "masonry":
                self.hide_parameter_by_name("grid_justification")
            else:
                self.show_parameter_by_name("grid_justification")
        if parameter.name == "output_image_size":
            if value == "custom":
                self.show_parameter_by_name("output_image_width")
                self.hide_parameter_by_name("output_preset")
            else:
                self.hide_parameter_by_name("output_image_width")
                self.show_parameter_by_name("output_preset")
        if parameter.name == "transparent_bg":
            if value:
                self.hide_parameter_by_name("background_color")
            else:
                self.show_parameter_by_name("background_color")
        if parameter.name == "output_format" and value == "jpeg":
            self.set_parameter_value("transparent_bg", False)
            self.show_parameter_by_name("background_color")

        # Dynamic Video Information Logic
        if "Video" in str(parameter.type) or parameter.name == "nth_frame":
            url = None
            if "Video" in str(parameter.type):
                if value:
                    if isinstance(value, dict):
                        url = value.get("value")
                    elif hasattr(value, "value"):
                        url = value.value
                    elif isinstance(value, str):
                        url = value
            else:
                try:
                    video_artifact, _ = self._get_video_input_data()
                    url = video_artifact
                except Exception:
                    url = None

            if url:
                self._update_frame_info(url)
            else:
                self.frame_range_message.value = "Connect a video to calculate frames."

        return super().after_value_set(parameter, value)

    def _update_frame_info(self, url: str) -> None:
        """Calculate and display estimated extracted frames based on video stats."""
        nth = self.get_parameter_value("nth_frame") or 30
        try:
            _, ffprobe_path = self._get_ffmpeg_paths()
            cmd = [
                ffprobe_path, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,duration,r_frame_rate",
                "-of", "json", url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                if streams:
                    stream = streams[0]
                    total_frames = 0
                    if stream.get("nb_frames"):
                        total_frames = int(stream.get("nb_frames"))
                    elif stream.get("duration") and stream.get("r_frame_rate"):
                        dur = float(stream.get("duration"))
                        num, den = stream.get("r_frame_rate").split("/")
                        total_frames = int(dur * (float(num) / float(den)))
                    
                    if total_frames > 0:
                        estimated_extractions = max(1, total_frames // nth)
                        self.frame_range_message.value = f"Grid will contain approx. {estimated_extractions} frames (Total: {total_frames})."
                        return
        except Exception:
            pass
        self.frame_range_message.value = "Frame calculation unknown."

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []
        if self.get_parameter_value("output_image_width") <= 0:
            exceptions.append(ValueError(f"{self.name}: Output image width must be > 0"))
        if self.get_parameter_value("columns") <= 0:
            exceptions.append(ValueError(f"{self.name}: Columns must be > 0"))
        if self.get_parameter_value("nth_frame") <= 0:
            exceptions.append(ValueError(f"{self.name}: Extract Every Nth Frame must be > 0"))
        return exceptions

    def _get_output_dimensions(self, output_image_size: str, output_preset: str, output_image_width_param: int) -> tuple[int, int | None, bool]:
        """Determine output dimensions based on size mode."""
        if output_image_size == "preset":
            preset_dimensions = {
                "4K (3840x2160)": (3840, 2160),
                "1440p (2560x1440)": (2560, 1440),
                "1080p (1920x1080)": (1920, 1080),
                "720p (1280x720)": (1280, 720),
            }
            width, height = preset_dimensions.get(output_preset, (1920, 1080))
            return width, height, True
        return output_image_width_param, None, False

    def _scale_grid_to_fit(self, grid_image: Any, target_width: int, target_height: int) -> tuple[Any, int, int]:
        """Scale grid image to fit within target dimensions."""
        grid_width, grid_height = grid_image.width, grid_image.height
        scale_factor = min(target_width / grid_width, target_height / grid_height, 1.0)
        
        if scale_factor < 1.0:
            new_width = int(grid_width * scale_factor)
            new_height = int(grid_height * scale_factor)
            return grid_image.resize((new_width, new_height), Image.Resampling.LANCZOS), new_width, new_height
        return grid_image, grid_width, grid_height

    def _apply_preset_canvas(self, grid_image: Any, target_dimensions: tuple[int, int], *, background_color: str, grid_justification: str, transparent_bg: bool) -> Any:
        """Apply preset dimensions by scaling grid and placing on exact-sized canvas."""
        output_image_width, output_image_height = target_dimensions
        grid_image, grid_width, grid_height = self._scale_grid_to_fit(grid_image, output_image_width, output_image_height)

        canvas = create_background_image(output_image_width, output_image_height, background_color, transparent_bg=transparent_bg)
        y_offset = (output_image_height - grid_height) // 2

        if grid_justification == "center":
            x_offset = (output_image_width - grid_width) // 2
        elif grid_justification == "right":
            x_offset = output_image_width - grid_width
        else:
            x_offset = 0

        canvas.paste(grid_image, (x_offset, y_offset), grid_image if grid_image.mode == "RGBA" else None)
        return canvas

    def process(self) -> AsyncResult[None]:
        """Entry point for async Griptape engine execution."""
        input_url, detected_format = self._get_video_input_data()
        self._log_format_detection(detected_format)
        
        yield lambda: self._process_frames_to_grid(input_url)

    def _process_frames_to_grid(self, input_url: str) -> None:
        """Extract frames using FFmpeg, generate the grid, and cleanup."""
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            self.append_value_to_parameter("logs", "[Started extracting frames..]\n")
            self._validate_url_safety(input_url)
            
            # Use the newly added _build_ffmpeg_command 
            output_pattern = str(temp_dir / "frame_%05d.png")
            _, ffprobe_path = self._get_ffmpeg_paths()
            input_frame_rate, _, _ = self._detect_video_properties(input_url, ffprobe_path)
            
            cmd = self._build_ffmpeg_command(input_url, output_pattern, input_frame_rate)
            self._run_ffmpeg_command(cmd, timeout=600)
            
            # Gather extracted frames
            extracted_files = sorted(list(temp_dir.glob("*.png")))
            if not extracted_files:
                raise ValueError("FFmpeg completed but no frames were extracted.")
                
            self.append_value_to_parameter("logs", f"[Extracted {len(extracted_files)} frames. Generating grid..]\n")

            # --- Grid Generation Logic ---
            layout_style = self.get_parameter_value("layout_style")
            columns = self.get_parameter_value("columns")
            output_image_size = self.get_parameter_value("output_image_size")
            output_preset = self.get_parameter_value("output_preset")
            output_image_width_param = self.get_parameter_value("output_image_width")
            spacing = self.get_parameter_value("spacing")
            background_color = self.get_parameter_value("background_color")
            border_radius = self.get_parameter_value("border_radius")
            crop_to_fit = self.get_parameter_value("crop_to_fit")
            output_format = self.get_parameter_value("output_format")
            transparent_bg = self.get_parameter_value("transparent_bg")
            grid_justification = self.get_parameter_value("grid_justification")

            output_width, output_height, use_preset = self._get_output_dimensions(
                output_image_size, output_preset, output_image_width_param
            )

            # Convert file paths to strings to be processed by grid util
            image_paths = [str(p) for p in extracted_files]

            if layout_style.lower() == "masonry":
                grid_image = create_masonry_layout(
                    image_paths, columns, output_width, spacing,
                    background_color, border_radius, transparent_bg=transparent_bg,
                )
            else:
                grid_image = create_grid_layout(
                    image_paths, columns, output_width, spacing,
                    background_color, border_radius, crop_to_fit=crop_to_fit,
                    transparent_bg=transparent_bg, justification=grid_justification,
                )

            if use_preset and output_height is not None:
                grid_image = self._apply_preset_canvas(
                    grid_image, (output_width, output_height),
                    background_color=background_color,
                    grid_justification=grid_justification,
                    transparent_bg=transparent_bg,
                )

            # Save the final grid
            filename = generate_filename(node_name=self.name, suffix=self._get_output_suffix(), extension=output_format)
            static_url = GriptapeNodes.StaticFilesManager().save_static_file(
                image_to_bytes(grid_image, output_format), filename
            )
            
            self.parameter_output_values["grid_output"] = ImageUrlArtifact(value=static_url)
            self.append_value_to_parameter("logs", "[Finished generating grid.]\n")

        except Exception as e:
            msg = f"{self.name}: Error processing frames to grid: {e}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise RuntimeError(msg) from e
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            cleanup_temp_files()
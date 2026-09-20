import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin, ParameterMessage
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from PIL import Image

from griptape_nodes_library.utils.image_utils import save_pil_image_to_static_file
from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor


class ExtractFrame(BaseVideoProcessor):
    """Extract a specific frame from a video and output it as an ImageUrlArtifact, along with frame metadata."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Hide parameters that aren't relevant for frame extraction
        self.hide_parameter_by_name("output_frame_rate")
        self.hide_parameter_by_name("processing_speed")
        self.hide_parameter_by_name("output")

        # Frame number integer parameter
        self.add_parameter(
            Parameter(
                name="frame_number",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=0,
                tooltip="The specific frame index to extract (0-indexed).",
                ui_options={"display_name": "Frame Number", "step": 1},
            )
        )

        # Dynamic parameter message to display the available frame range
        self.frame_range_message = ParameterMessage(
            name="frame_range_message",
            variant="info",
            value="Connect a video to see available frames.",
        )
        self.add_node_element(self.frame_range_message)

        # Add image output parameter
        self.add_parameter(
            ParameterImage(
                name="extracted_frame",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The extracted frame from the video as an image",
                ui_options={"pulse_on_run": True, "expander": True},
            )
        )

        # --- New Metadata Outputs ---
        self.add_parameter(
            Parameter(
                name="total_frames",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The total number of frames detected in the video.",
                ui_options={"display_name": "Total Frames"},
            )
        )

        self.add_parameter(
            Parameter(
                name="start_frame",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The index of the first frame (0).",
                ui_options={"display_name": "Start Frame"},
            )
        )

        self.add_parameter(
            Parameter(
                name="end_frame",
                type=ParameterTypeBuiltin.INT.value,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The index of the last available frame.",
                ui_options={"display_name": "End Frame"},
            )
        )

    def _setup_custom_parameters(self) -> None:
        """Setup custom parameters specific to this video processor."""
        pass

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Lifecycle callback to dynamically display available frames when a video is connected."""
        if "Video" in str(parameter.type):
            if value:
                url = None
                if isinstance(value, dict):
                    url = value.get("value")
                elif hasattr(value, "value"):
                    url = value.value
                elif isinstance(value, str):
                    url = value

                if url:
                    self._update_frame_range(url)
                else:
                    self.frame_range_message.value = "Invalid video input."
            else:
                self.frame_range_message.value = "Connect a video to see available frames."

        return super().after_value_set(parameter, value)

    def _get_frame_stats(self, url: str) -> tuple[int, int, int] | None:
        """Fetch video stats using ffprobe and return (start_frame, end_frame, total_frames)."""
        try:
            _, ffprobe_path = self._get_ffmpeg_paths()
            cmd = [
                ffprobe_path,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames,duration,r_frame_rate",
                "-of", "json",
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                if streams:
                    stream = streams[0]
                    total_frames = 0
                    
                    # First try direct frame count
                    nb_frames = stream.get("nb_frames")
                    if nb_frames:
                        total_frames = int(nb_frames)
                    else:
                        # Fallback to duration * fps estimate
                        duration = stream.get("duration")
                        fps_str = stream.get("r_frame_rate")
                        if duration and fps_str and "/" in fps_str:
                            dur = float(duration)
                            num, den = fps_str.split("/")
                            fps = float(num) / float(den)
                            total_frames = int(dur * fps)
                    
                    if total_frames > 0:
                        return (0, total_frames - 1, total_frames)

        except Exception as e:
            self._log(f"Error calculating frame stats: {e}")
            
        return None

    def _update_frame_range(self, url: str) -> None:
        """Update the UI message with the calculated frame range."""
        stats = self._get_frame_stats(url)
        if stats:
            start_frame, end_frame, total_frames = stats
            self.frame_range_message.value = f"Available frames: {start_frame} to {end_frame} (Total: {total_frames})"
        else:
            self.frame_range_message.value = "Frame range: Unknown"

    def _get_processing_description(self) -> str:
        """Get a description of what this processor does."""
        frame_number = self.get_parameter_value("frame_number") or 0
        return f"extracting frame {frame_number} from video"

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:  # noqa: ARG002
        """Build the FFmpeg command for extracting the requested frame."""
        ffmpeg_path, _ = self._get_ffmpeg_paths()
        frame_number = self.get_parameter_value("frame_number") or 0

        cmd = [
            ffmpeg_path,
            "-i",
            input_url,
            "-vf",
            f"select=eq(n\\,{frame_number})",
            "-vsync",
            "vfr",
            "-frames:v",
            "1",
            "-q:v",
            "0", 
            "-f",
            "image2",
            "-update",
            "1",
            "-y",
            output_path,
        ]

        return cmd

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get the output filename suffix."""
        frame_number = self.get_parameter_value("frame_number") or 0
        return f"_frame_{frame_number}"

    def process(self) -> AsyncResult[None]:
        """Extract the frame from the input video and save as ImageUrlArtifact."""
        input_url, detected_format = self._get_video_input_data()
        self._log_format_detection(detected_format)
        self.append_value_to_parameter("logs", "[Processing extract frame..]\n")

        try:
            self.append_value_to_parameter("logs", "[Started extracting frame..]\n")
            yield lambda: self._process_extract_frame(input_url)
            self.append_value_to_parameter("logs", "[Finished extracting frame.]\n")
        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error extracting frame: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise ValueError(msg) from e

    def _process_extract_frame(self, input_url: str) -> None:
        """Extract the frame, populate metadata, and save as ImageUrlArtifact."""
        def _validate_output_file(file_path: Path) -> None:
            if not file_path.exists() or file_path.stat().st_size == 0:
                error_msg = "FFmpeg did not create output file or file is empty"
                raise ValueError(error_msg)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_image_path = Path(temp_file.name)

        try:
            self.append_value_to_parameter("logs", f"{self._get_processing_description()}\n")
            self._validate_url_safety(input_url)
            
            # Retrieve frame statistics and populate output parameters
            stats = self._get_frame_stats(input_url)
            if stats:
                start_frame, end_frame, total_frames = stats
                self.parameter_output_values["start_frame"] = start_frame
                self.parameter_output_values["end_frame"] = end_frame
                self.parameter_output_values["total_frames"] = total_frames
            else:
                self.parameter_output_values["start_frame"] = 0
                self.parameter_output_values["end_frame"] = 0
                self.parameter_output_values["total_frames"] = 0
                self._log("Warning: Could not determine frame stats during processing.")

            # FFmpeg execution
            _ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()
            input_frame_rate, _, _ = self._detect_video_properties(input_url, ffprobe_path)
            cmd = self._build_ffmpeg_command(input_url, str(temp_image_path), input_frame_rate)

            self._run_ffmpeg_command(cmd, timeout=300)
            _validate_output_file(temp_image_path)

            last_frame_pil = Image.open(temp_image_path)
            image_artifact = save_pil_image_to_static_file(last_frame_pil, "PNG")

            self.parameter_output_values["extracted_frame"] = image_artifact
            self.append_value_to_parameter("logs", "Successfully extracted frame as image\n")

        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error extracting frame: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise ValueError(msg) from e
        finally:
            self._cleanup_temp_file(temp_image_path)
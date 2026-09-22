"""Griptape Nodes wrapper that runs NormalCrafter (Bin et al., ICCV 2025) in-process.

Unlike NormalCrafterSharedMediaNode in this same folder (which dispatches a
job to a remote M1 server over HTTP), this node loads and runs the
NormalCrafter pipeline directly in the engine's own process: feed it a
video, get back a temporally consistent normal-map video.

The pipeline code (normal_crafter_ppl.py, unet.py, utils.py) is vendored,
unmodified, into ./_normalcrafter_pkg/ from
https://github.com/Binyr/NormalCrafter, so this node has no dependency on
any other checkout on disk. Only this file's own directory is added to
sys.path, to import that vendored package.
"""

from __future__ import annotations

import gc
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from memory_estimate import estimate as estimate_memory  # noqa: E402
from video_probe import probe_video_dimensions, resolve_local_video_path, scale_to_max_res  # noqa: E402

logger = logging.getLogger("vibe_nodes_library.normalCrafter")

# One loaded pipeline at a time, keyed by (unet_repo, base_svd_repo, cpu_offload).
_PIPELINE_CACHE: dict[tuple[str, str, str], Any] = {}

CPU_OFFLOAD_CHOICES = ["model", "sequential", "none"]

HIGH_MEMORY_WARNING_GB = 24.0


class NormalCrafterLocalNode(SuccessFailureNode):
    """Generates a temporally consistent normal map video from an input video, running NormalCrafter locally."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._video_dims = None  # cached VideoDimensions of the last-probed input_video

        self.add_parameter(
            Parameter(
                name="input_video",
                input_types=["VideoArtifact", "VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="Source video to estimate normals for.",
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="max_res",
                type="int",
                default_value=1024,
                tooltip=(
                    "Longest side (px) the video is downscaled to before inference. "
                    "Lower = less VRAM and faster, at the cost of detail."
                ),
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                traits={Slider(min_val=256, max_val=1920)},
            )
        )
        self.add_parameter(
            Parameter(
                name="process_length",
                type="int",
                default_value=-1,
                tooltip="Maximum number of frames to process. -1 processes every frame.",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="target_fps",
                type="int",
                default_value=-1,
                tooltip="Frame rate to resample the input to before inference. -1 keeps the source fps.",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="window_size",
                type="int",
                default_value=14,
                tooltip="Temporal window size (frames denoised together). Matches the SVD UNet's native window.",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="time_step_size",
                type="int",
                default_value=10,
                tooltip="Stride between windows. Smaller values overlap more: smoother but slower.",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="decode_chunk_size",
                type="int",
                default_value=7,
                tooltip="Frames decoded per VAE decode step. Lower uses less VRAM at decode time.",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="seed",
                type="int",
                default_value=42,
                tooltip="Random seed.",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="cpu_offload",
                type="str",
                default_value="model",
                tooltip=(
                    "model: offload submodules to CPU between steps (default -- matches the ~6-20GB README figures). "
                    "sequential: offload more aggressively, slower, lowest VRAM. "
                    "none: keep the whole pipeline resident on GPU."
                ),
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.INPUT},
                traits={Options(choices=CPU_OFFLOAD_CHOICES)},
            )
        )
        self.add_parameter(
            Parameter(
                name="unet_repo",
                type="str",
                default_value="Yanrui95/NormalCrafter",
                tooltip="HuggingFace repo providing the NormalCrafter UNet + VAE weights.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="base_svd_repo",
                type="str",
                default_value="stabilityai/stable-video-diffusion-img2vid-xt",
                tooltip="Base Stable Video Diffusion repo the pipeline's other components load from.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        # Dynamic estimated-memory label: recomputed in after_value_set() whenever
        # input_video or max_res changes.
        self._memory_message = ParameterMessage(
            name="memory_estimate",
            variant="info",
            title="Estimated Memory",
            value="Connect a video to estimate VRAM usage.",
        )
        self.add_node_element(self._memory_message)

        self.add_parameter(
            Parameter(
                name="output_video",
                output_type="VideoUrlArtifact",
                tooltip="Generated normal map video.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="preprocessed_video",
                output_type="VideoUrlArtifact",
                tooltip="The (resized/resampled) input video actually fed to the pipeline.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the NormalCrafter run.",
            result_details_placeholder="Run details appear here after execution.",
        )

    # -- dynamic memory label --------------------------------------------------

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter.name == "input_video":
            self._video_dims = probe_video_dimensions(value) if value is not None else None
            self._update_memory_estimate()
        elif parameter.name == "max_res":
            self._update_memory_estimate()

    def _update_memory_estimate(self) -> None:
        if self._video_dims is None:
            self._memory_message.value = "Connect a video to estimate VRAM usage."
            self._memory_message.variant = "info"
            return

        max_res = self.get_parameter_value("max_res") or 1024
        proc_w, proc_h = scale_to_max_res(self._video_dims.width, self._video_dims.height, int(max_res))
        est = estimate_memory(proc_w, proc_h)

        self._memory_message.value = (
            f"{est.width}x{est.height} ({est.total_pixels:,} px -- {est.context_label}) "
            f"from {self._video_dims.width}x{self._video_dims.height} source "
            f"→ approx. {est.estimated_gb:.1f} GB VRAM\n"
            "Fitted to NormalCrafter's own README figures (fp16, window_size=14, "
            "decode_chunk_size=7, cpu_offload='model') -- not a live measurement of your GPU."
        )
        self._memory_message.variant = "warning" if est.estimated_gb >= HIGH_MEMORY_WARNING_GB else "info"

    # -- inference ---------------------------------------------------------------

    def process(self) -> AsyncResult:
        self._clear_execution_status()
        yield lambda: self._run()

    def _run(self) -> None:
        try:
            self._infer()
            self._set_status_results(was_successful=True, result_details="NormalCrafter run completed.")
        except Exception as e:  # noqa: BLE001
            logger.exception("%s: NormalCrafter inference failed", self.name)
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)

    def _infer(self) -> None:
        import torch
        from _normalcrafter_pkg.utils import read_video_frames, save_video, vis_sequence_normal
        from diffusers.training_utils import set_seed

        input_video = self.get_parameter_value("input_video")
        if input_video is None:
            raise ValueError("Missing required 'input_video' input.")

        max_res = int(self.get_parameter_value("max_res") or 1024)
        process_length = int(self.get_parameter_value("process_length") or -1)
        target_fps = int(self.get_parameter_value("target_fps") or -1)
        window_size = int(self.get_parameter_value("window_size") or 14)
        time_step_size = int(self.get_parameter_value("time_step_size") or 10)
        decode_chunk_size = int(self.get_parameter_value("decode_chunk_size") or 7)
        seed = int(self.get_parameter_value("seed") or 42)
        cpu_offload = self.get_parameter_value("cpu_offload") or "model"
        unet_repo = self.get_parameter_value("unet_repo") or "Yanrui95/NormalCrafter"
        base_svd_repo = self.get_parameter_value("base_svd_repo") or "stabilityai/stable-video-diffusion-img2vid-xt"

        set_seed(seed)

        pipe = self._get_pipeline(unet_repo, base_svd_repo, cpu_offload)

        local_video_path, is_temp_input = resolve_local_video_path(input_video)
        out_dir = Path(tempfile.mkdtemp(prefix="normalcrafter_"))
        try:
            frames, resolved_fps = read_video_frames(str(local_video_path), process_length, target_fps, max_res)

            with torch.inference_mode():
                normals = pipe(
                    frames,
                    decode_chunk_size=decode_chunk_size,
                    time_step_size=time_step_size,
                    window_size=window_size,
                ).frames[0]

            vis = vis_sequence_normal(normals)

            vis_path = out_dir / "normal_vis.mp4"
            input_path = out_dir / "preprocessed_input.mp4"
            save_video(vis, str(vis_path), fps=resolved_fps)
            save_video(frames, str(input_path), fps=resolved_fps)

            self.parameter_output_values["output_video"] = self._publish_video(vis_path)
            self.parameter_output_values["preprocessed_video"] = self._publish_video(input_path)
        finally:
            if is_temp_input:
                local_video_path.unlink(missing_ok=True)
            for f in out_dir.glob("*"):
                f.unlink(missing_ok=True)
            out_dir.rmdir()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_pipeline(self, unet_repo: str, base_svd_repo: str, cpu_offload: str) -> Any:
        import torch
        from _normalcrafter_pkg.normal_crafter_ppl import NormalCrafterPipeline
        from _normalcrafter_pkg.unet import DiffusersUNetSpatioTemporalConditionModelNormalCrafter
        from diffusers import AutoencoderKLTemporalDecoder

        cache_key = (unet_repo, base_svd_repo, cpu_offload)
        cached = _PIPELINE_CACHE.get(cache_key)
        if cached is not None:
            return cached

        logger.info("%s: loading NormalCrafter pipeline (%s / %s)", self.name, unet_repo, base_svd_repo)
        unet = DiffusersUNetSpatioTemporalConditionModelNormalCrafter.from_pretrained(
            unet_repo, subfolder="unet", low_cpu_mem_usage=True
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(unet_repo, subfolder="vae")
        weight_dtype = torch.float16
        vae.to(dtype=weight_dtype)
        unet.to(dtype=weight_dtype)

        pipe = NormalCrafterPipeline.from_pretrained(
            base_svd_repo, unet=unet, vae=vae, torch_dtype=weight_dtype, variant="fp16"
        )

        if cpu_offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif cpu_offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:  # noqa: BLE001
            logger.info("%s: xformers not enabled (%s)", self.name, e)

        _PIPELINE_CACHE.clear()  # single-slot cache: one loaded pipeline resident at a time
        _PIPELINE_CACHE[cache_key] = pipe
        return pipe

    @staticmethod
    def _publish_video(path: Path) -> VideoUrlArtifact:
        filename = f"{uuid.uuid4()}{path.suffix}"
        url = GriptapeNodes.StaticFilesManager().save_static_file(path.read_bytes(), filename)
        return VideoUrlArtifact(url)

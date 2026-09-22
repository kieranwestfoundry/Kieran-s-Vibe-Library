"""Resolve a Griptape video artifact to its dimensions, for memory estimation.

Reads just the container header via OpenCV -- no frames are decoded. Also
carries the same input-resolution downscale rule NormalCrafter's own
``normalcrafter.utils.read_video_frames`` applies, so the numbers shown on
the node match what actually gets processed.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoDimensions:
    width: int
    height: int
    frame_count: int
    fps: float


def resolve_local_video_path(video_artifact: Any) -> tuple[Path, bool]:
    """Return a local filesystem path for a video artifact's contents.

    Returns (path, is_temp) -- if is_temp is True, the caller is responsible
    for deleting the file once done with it.
    """
    url = getattr(video_artifact, "value", video_artifact)

    # Path("") normalizes to Path(".") -- the current working directory -- which
    # .exists() happily reports as True. An empty/blank url would silently resolve
    # to the cwd and get handed to decord as a "video", which fails deep inside its
    # C++ reader with an opaque "Is a directory" error instead of a clear one here.
    if isinstance(url, (str, os.PathLike)) and not str(url).strip():
        raise ValueError(
            "input_video resolved to an empty path. The connected video artifact has "
            "no url/value set -- check that a video is actually loaded upstream of "
            "this node (not just an empty/unconnected video parameter)."
        )

    # is_file(), not exists(): exists() is also True for a directory, which decord
    # will fail on just as unhelpfully as it does on an empty path.
    if isinstance(url, (str, os.PathLike)) and Path(url).is_file():
        return Path(url), False

    from griptape_nodes.files.file import File

    suffix = Path(str(url)).suffix or ".mp4"
    fd, temp_path_str = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    temp_path = Path(temp_path_str)
    temp_path.write_bytes(File(str(url)).read_bytes())
    return temp_path, True


def probe_video_dimensions(video_artifact: Any) -> VideoDimensions | None:
    """Read width/height/frame count/fps from a video artifact without decoding frames.

    Returns None if the artifact is empty or the file can't be opened.
    """
    if video_artifact is None:
        return None

    import cv2  # type: ignore[reportMissingImports]

    local_path, is_temp = resolve_local_video_path(video_artifact)
    try:
        cap = cv2.VideoCapture(str(local_path))
        if not cap.isOpened():
            return None
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
        finally:
            cap.release()
        if width <= 0 or height <= 0:
            return None
        return VideoDimensions(width=width, height=height, frame_count=frame_count, fps=fps)
    finally:
        if is_temp:
            local_path.unlink(missing_ok=True)


def scale_to_max_res(original_width: int, original_height: int, max_res: int) -> tuple[int, int]:
    """Fit the longest side to max_res, preserving aspect ratio, no upscaling.

    This mirrors normalcrafter.utils.read_video_frames' downscale rule exactly:

        if max(h, w) > max_res:
            scale = max_res / max(h, w)
            height, width = round(h * scale), round(w * scale)
    """
    longest = max(original_width, original_height)
    if longest <= max_res:
        return original_width, original_height
    scale = max_res / longest
    return round(original_width * scale), round(original_height * scale)

"""Estimated VRAM usage for a NormalCrafter inference pass.

The formula below is fitted to Kieran's own reference table (256x256 up to
1920x1080), which in turn was built from the two data points NormalCrafter's
own README states directly:

    ~6 GB VRAM  @ 512x256  (`python run.py --max-res 512`)
    ~20 GB VRAM @ 1024x576 (`python run.py`, default max-res)

Fitting a line against total pixel count reproduces every value in the table
to within its own rounding:

    estimated_gb = BASE_OVERHEAD_GB + total_pixels / PIXELS_PER_GB

with BASE_OVERHEAD_GB = 2.0 and PIXELS_PER_GB = 32768 (2**15). Read as: a
~2 GB fixed overhead for the SVD UNet + VAE in fp16, plus ~1 GB for every
32,768 pixels in a frame. This models running the pipeline the way
`run.py`/`app.py` do by default -- fp16, `window_size=14`,
`decode_chunk_size=7`, `cpu_offload="model"` -- so it's an estimate fitted to
those reference figures, not a live measurement of your GPU.

Reference table this was fitted against:

      256x256    65,536 px   Low res            4.0 GB
      512x256   131,072 px   (given baseline)    6.0 GB
      512x512   262,144 px   Standard 1:1       10.0 GB
      768x512   393,216 px   Medium 3:2         14.0 GB
     1024x576   589,824 px   (given baseline)   20.0 GB
     1280x720   921,600 px   720p HD            30.1 GB
    1024x1024 1,048,576 px   High res 1:1       34.0 GB
    1920x1080 2,073,600 px   1080p Full HD      65.3 GB
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_OVERHEAD_GB = 2.0
PIXELS_PER_GB = 32768  # 2**15

# (width, height, context label) -- used only to label the closest matching tier.
REFERENCE_TABLE: tuple[tuple[int, int, str], ...] = (
    (256, 256, "Low res"),
    (512, 256, "Given baseline"),
    (512, 512, "Standard 1:1"),
    (768, 512, "Medium 3:2"),
    (1024, 576, "Given baseline"),
    (1280, 720, "720p HD"),
    (1024, 1024, "High res 1:1"),
    (1920, 1080, "1080p Full HD"),
)

# How close (as a fraction of total pixels) a resolution needs to be to a
# reference tier before it's labelled with that tier's name.
_NEAREST_LABEL_TOLERANCE = 0.1


@dataclass(frozen=True)
class MemoryEstimate:
    width: int
    height: int
    total_pixels: int
    estimated_gb: float
    context_label: str


def estimate_memory_gb(width: int, height: int) -> float:
    """Estimate peak VRAM (GB) for a fp16 NormalCrafter pass at this resolution."""
    total_pixels = max(int(width), 0) * max(int(height), 0)
    return round(BASE_OVERHEAD_GB + total_pixels / PIXELS_PER_GB, 1)


def nearest_reference_label(width: int, height: int) -> str:
    """Label the resolution with the closest reference tier, if it's within tolerance."""
    total_pixels = width * height
    if total_pixels <= 0:
        return "Custom resolution"
    closest = min(REFERENCE_TABLE, key=lambda row: abs(row[0] * row[1] - total_pixels))
    closest_pixels = closest[0] * closest[1]
    if abs(closest_pixels - total_pixels) / closest_pixels <= _NEAREST_LABEL_TOLERANCE:
        return closest[2]
    return "Custom resolution"


def estimate(width: int, height: int) -> MemoryEstimate:
    return MemoryEstimate(
        width=width,
        height=height,
        total_pixels=width * height,
        estimated_gb=estimate_memory_gb(width, height),
        context_label=nearest_reference_label(width, height),
    )

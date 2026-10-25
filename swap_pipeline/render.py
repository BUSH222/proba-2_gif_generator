"""Turning the grey level frames of the pipeline into the files people look at: tint, contrast stretch, PNG,
GIF and MP4. Everything works on uint8 grey frames plus one shared 256 colour palette, so a tinted frame is stored as
its grey levels and a palette (P mode): lossless, and every frame of an animation has exactly the same colours."""

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

GOLD = "#edb103"  # the fill of the old ImageMagick tint (main.py on dev: -fill #edb103 -tint 100)
STRETCH = 0.0  # percent clipped at each end. Off: the scalings are calibrated, and 0.3 (the old ImageMagick
# -contrast-stretch 0.3%) puts the white point at ~200 DN/s instead of 334, clipping the active regions
LUMA = np.array([0.2126, 0.7152, 0.0722])  # Rec. 709, what ImageMagick 7 uses for the intensity of the fill


def parse_color(color: str) -> np.ndarray:
    """'#rrggbb' (or 'rrggbb') -> [r, g, b] as floats 0-255."""
    text = color.removeprefix("#")
    if len(text) != 6:
        raise ValueError(f"not a #rrggbb colour: {color}")
    return np.array([int(text[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def tint_palette(fill: str = GOLD, percent: float = 100.0) -> list[int]:
    """The palette (768 values: r, g, b for grey levels 0-255) of ImageMagick's `-fill <fill> -tint <percent>` on a
    grey image: the mid tones are pushed by (fill - its luma) with weight 1 - 4 (g - 1/2)^2, black and white stay.
    Matches `magick mogrify -tint` to 1 grey level."""
    color = parse_color(fill)
    vector = percent / 100.0 * color - float(LUMA @ color)
    grey = np.arange(256, dtype=np.float64)
    weight = 1.0 - 4.0 * (grey / 255.0 - 0.5) ** 2
    rgb = np.clip(np.round(grey[:, None] + vector[None, :] * weight[:, None]), 0, 255).astype(np.uint8)
    return rgb.ravel().tolist()


def stretch_limits(frames: Sequence[np.ndarray], percent: float = STRETCH) -> tuple[int, int]:
    """Grey levels below which and above which `percent` of all the pixels of all the frames lie. One pair for the
    whole set, so that an animation doesn't flicker as it did with a stretch per image. (0, 255), no change, for 0."""
    if percent <= 0:
        return 0, 255
    histogram = sum(np.bincount(frame.ravel(), minlength=256) for frame in frames)
    cumulative = np.cumsum(histogram) / np.sum(histogram)
    low = int(np.searchsorted(cumulative, percent / 100.0, side="right"))
    high = int(np.searchsorted(cumulative, 1.0 - percent / 100.0, side="left"))
    return (low, high) if high > low else (0, 255)


def apply_stretch(frame: np.ndarray, low: int, high: int) -> np.ndarray:
    """Map grey level low -> 0 and high -> 255, linearly, clipping outside."""
    if (low, high) == (0, 255):
        return frame
    lut = np.clip(np.round((np.arange(256) - low) * 255.0 / (high - low)), 0, 255).astype(np.uint8)
    return lut[frame]


def to_image(frame: np.ndarray, palette: list[int] | None) -> Image.Image:
    """A grey frame as a PIL image: P mode with the palette, L (grey) without one."""
    image = Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8))
    if palette is not None:
        image.putpalette(palette)  # L -> P, grey level g becomes palette index g
    return image


def to_rgb(frame: np.ndarray, palette: list[int] | None) -> np.ndarray:
    if palette is None:
        return np.repeat(frame[:, :, None], 3, axis=2)
    return np.asarray(palette, dtype=np.uint8).reshape(256, 3)[frame]


def save_png(frame: np.ndarray, path: Path, palette: list[int] | None = None) -> None:
    to_image(frame, palette).save(path)


def same_size(frames: Sequence[np.ndarray]) -> Iterable[np.ndarray]:
    """The frames, those of another size than the first resized to it."""
    shape = frames[0].shape
    for frame in frames:
        if frame.shape != shape:
            frame = np.asarray(Image.fromarray(frame).resize((shape[1], shape[0]), Image.Resampling.BICUBIC))
        yield frame


def save_gif(frames: Sequence[np.ndarray], path: Path, fps: float, palette: list[int] | None = None) -> None:
    """Animated GIF, looping, all frames on the same palette (grey without one)."""
    shared = palette if palette is not None else [level for level in range(256) for _ in range(3)]
    first, *rest = (to_image(frame, shared) for frame in same_size(frames))
    first.save(path, save_all=True, append_images=rest, duration=round(1000 / fps), loop=0, optimize=False)


def save_mp4(frames: Sequence[np.ndarray], path: Path, fps: float, palette: list[int] | None = None) -> None:
    """H.264 video."""
    import imageio.v2 as imageio  # brings its own ffmpeg, only needed for videos

    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=2) as writer:
        for frame in same_size(frames):
            writer.append_data(to_rgb(frame, palette))

"""swap_pipeline.render: the tint, the stretch, the rotation and the files."""

import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from swap_pipeline import render

if TYPE_CHECKING:
    from pathlib import Path


def palette_rgb(palette: list[int]) -> np.ndarray:
    return np.asarray(palette, dtype=np.int32).reshape(256, 3)


def test_tint_keeps_black_and_white_and_colours_the_mid_tones() -> None:
    rgb = palette_rgb(render.tint_palette())
    assert rgb[0].tolist() == [0, 0, 0]
    assert rgb[255].tolist() == [255, 255, 255]
    assert rgb[128].tolist() == [188, 128, 0]  # measured with ImageMagick 7.1: -fill #edb103 -tint 100
    assert np.all(np.diff(rgb.sum(axis=1)) >= 0)  # brightness still rises with the grey level


def test_bad_colour() -> None:
    with pytest.raises(ValueError, match="rrggbb"):
        render.parse_color("#abc")


@pytest.mark.skipif(shutil.which("magick") is None, reason="ImageMagick not installed")
def test_tint_matches_imagemagick(tmp_path: Path) -> None:
    ramp = tmp_path / "ramp.png"
    Image.fromarray(np.tile(np.arange(256, dtype=np.uint8), (2, 1))).save(ramp)
    subprocess.run(["magick", "mogrify", "-fill", render.GOLD, "-tint", "100", str(ramp)], check=True)
    theirs = np.asarray(Image.open(ramp).convert("RGB"))[0].astype(np.int32)
    assert np.abs(theirs - palette_rgb(render.tint_palette())).max() <= 1


def test_stretch_uses_all_the_frames_together() -> None:
    dark = np.full((10, 10), 20, dtype=np.uint8)
    bright = np.full((10, 10), 200, dtype=np.uint8)
    low, high = render.stretch_limits([dark, bright], percent=1.0)
    assert (low, high) == (20, 200)  # one pair for both: each frame alone would stretch to a flat image
    assert render.apply_stretch(dark, low, high).max() == 0
    assert render.apply_stretch(bright, low, high).min() == 255


def test_stretch_of_a_flat_set_does_nothing() -> None:
    flat = np.full((4, 4), 7, dtype=np.uint8)
    assert render.stretch_limits([flat]) == (0, 255)
    assert render.apply_stretch(flat, 0, 255) is flat


def test_rotation_is_clockwise() -> None:
    frame = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    assert render.rotate(frame, 1).tolist() == [[3, 1], [4, 2]]
    assert render.rotate(frame, 0) is frame


def test_png_keeps_the_grey_levels_with_the_palette(tmp_path: Path) -> None:
    frame = np.arange(256, dtype=np.uint8).reshape(16, 16)
    palette = render.tint_palette()
    render.save_png(frame, tmp_path / "a.png", palette)
    image = Image.open(tmp_path / "a.png")
    assert image.mode == "P"
    assert np.array_equal(np.asarray(image), frame)
    assert np.array_equal(np.asarray(image.convert("RGB")), palette_rgb(palette)[frame].astype(np.uint8))


def test_gif_frames_share_the_palette(tmp_path: Path) -> None:
    frames = [np.full((8, 8), level, dtype=np.uint8) for level in (0, 64, 128, 255)]
    palette = render.tint_palette()
    render.save_gif(frames, tmp_path / "out.gif", fps=10, palette=palette)
    gif = Image.open(tmp_path / "out.gif")
    assert gif.n_frames == len(frames)
    assert gif.info["duration"] == 100
    for index, frame in enumerate(frames):
        gif.seek(index)
        assert np.array_equal(np.asarray(gif.convert("RGB")), palette_rgb(palette)[frame].astype(np.uint8))


def test_no_stretch_by_default() -> None:
    frames = [np.arange(256, dtype=np.uint8).reshape(16, 16)]
    assert render.stretch_limits(frames) == (0, 255)  # the calibrated brightness is kept: nothing clipped
    assert render.stretch_limits(frames, percent=0.0) == (0, 255)

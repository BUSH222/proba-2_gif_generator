# Proba 2 SWAP Image Calibrator and GIF Generator

Turns raw dumps from ESA's PROBA-2 satellite (the `.cadu` files SatDump records) into calibrated, sun-centered,
solar-north-up and tinted images from its SWAP instrument (17.4 nm EUV), plus a GIF and an optional MP4.

Here is an example output of this application:
![result](out.gif)

### How it works
1) Read the dump: CADU frames → CCSDS packets (`swap_pipeline/cadu.py`, `rs.py`).
2) Rebuild each SWAP image (APID 20): its header and its JPEG, cut cleanly where packets were lost. Each image is timed in UTC from the on-board GPS records (APID 18), and named `SWAP_YYYYMMDDTHHMMSSZ.png` after its start of acquisition (`extract_images.py`).
3) Calibrate it with a Python port of the SWAP team's SolarSoft pipeline (`p2sw_prep.py`): dark current, flat field, bad pixels, despiking. Then, in one interpolation:
   - center the sun found by a limb fit
   - make the pixels square
   - turn solar north up, from the spacecraft roll in the telemetry
   
   The results match the PROBA2 archive's own level 1 images.
4) Render it like the PROBA2 Science Center's JPEG 2000 images on Helioviewer: their tone curve (grey = 18.03 (DN/s)^0.4485 − 0.83) and their sharpening of the 3–20 px structure. These were measured on 10 images, and the result is within 2.4 grey levels of theirs. Then tint, gold by default (`render.py`).

The packet format and the calibration are documented in [swap_pipeline/NOTES.md](swap_pipeline/NOTES.md).

## Requirements
- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (`uv sync` installs the dependencies from `pyproject.toml`)

ImageMagick and OpenCV are no longer needed.

## How to run:
1) clone and navigate to the root of the git repository
2) Run `uv sync`
3) Run `uv run main.py -i <dump.cadu or folder of dumps>`

```
uv run main.py -i cadus/proba2_dump.cadu               # -> cadus/SWAP_pro/SWAP_*.png + out.gif
uv run main.py -i cadus --make_mp4                     # every dump in cadus/ -> ./SWAP_pro, + out.mp4
uv run main.py -i cadus/proba2_dump.cadu --color none  # grey
```

### Command line arguments:

| Argument | Default | |
|---|---|---|
| `-i`, `--in_path` (or positional) | required | a `.cadu` dump, or a folder of them (all merged in time order) |
| `-o`, `--out_path` | `SWAP_pro` next to the input | output folder for the images, `out.gif` and `out.mp4` |
| `-r`, `--extra_rotation` | 0 | extra **clockwise** rotation in 90° steps (0-3), of the images and the animations |
| `-f`, `--fps` | 10 | GIF/MP4 frames per second |
| `--make_mp4` (`--mp4`) | off | also write `out.mp4` |
| `--no_gif` | off | don't write `out.gif` |
| `--color` | `tint` | `tint` or `none` (grey) |
| `--tint` | `#edb103` | tint colour |
| `--stretch` | 0 | contrast stretch: percent clipped to black and to white, over all images together. 0.3 is the old ImageMagick look, but it clips the brightest active regions |
| `--scaling` | `helioviewer` | `helioviewer` (as the P2SC's images on Helioviewer), `quicklook` (their quicklook movies: plain square root, softer) or `p2sw` (brighter faint corona) |
| `--raw` | off | skip the calibration: decoded grey levels, only oriented and centered |
| `--fits` | off | also keep each level 1 FITS file |
| `--archive-headers` | none | folder of the archive's level 0 headers or FITS: real detector temperature and LAR timing |
| `--detector-temp` | -0.73 | detector temperature (°C) assumed without an archive header |
| `--no_magick` | | deprecated, same as `--color none --stretch 0` |

The images are already aligned to solar north, so `-r` is only needed for a different framing. Images taken while
the spacecraft was turning (LAR) are rejected when `--archive-headers` provides their timing.

### Other tools
- `uv run stacker.py SWAP_pro --method median` stacks a folder of images (`avg`, `median`, `ks` kappa-sigma).
- `uv run python -m swap_pipeline.p2sw_prep swap_lv0_*.fits -o level1 --png` makes level 1 images from the archive's level 0 FITS files.
- `uv run pytest` runs the tests. The comparison with the archive needs `swap_pipeline/swap_archive/` (not in git) and is skipped without it.

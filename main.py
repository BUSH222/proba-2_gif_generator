"""PROBA-2 SWAP images from raw .cadu dumps (SatDump): calibrated, sun centered, solar north up, tinted, and stitched
into a GIF (and an MP4).

    uv run main.py -i cadus/proba2_dump.cadu             # -> cadus/SWAP_pro/SWAP_<time>.png + out.gif
    uv run main.py -i cadus --make_mp4                   # every dump in the folder -> ./SWAP_pro, + out.mp4

Every image goes through the archive's level 1 pipeline (swap_pipeline/p2sw_prep.py, a port of SolarSoft's
p2sw_prep.pro): dark current, flat field, bad pixels, despiking, sun centered, square pixels, solar north up. The
orientation comes from the spacecraft roll in the telemetry and a fit of the solar limb, so no alignment between frames
is needed. --raw skips the calibration and keeps the decoded grey levels, aligned the same way.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from swap_pipeline import render
from swap_pipeline.extract_images import (
    DETECTOR_TEMP,
    LEVEL1_WORKERS,
    Level1Job,
    assemble,
    collect,
    decode,
    find_archive_header,
    find_cadu_files,
    image_name,
    load_archive_headers,
    make_aligned,
    make_level1,
)

OUT_DIR = "SWAP_pro"
GIF_NAME = "out.gif"
MP4_NAME = "out.mp4"


def colour(text: str) -> str:
    try:
        render.parse_color(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrated, north up and tinted PROBA-2 SWAP images and a GIF from .cadu dumps."
    )
    parser.add_argument("cadu", nargs="?", type=Path, help="a .cadu dump, or a folder of them (same as -i)")
    parser.add_argument("-i", "--in_path", type=Path, help="a .cadu dump, or a folder of them")
    parser.add_argument(
        "-o", "--out_path", type=Path, help=f"output folder (default: {OUT_DIR} next to the dump or the folder)"
    )
    parser.add_argument("-f", "--fps", type=float, default=10, help="GIF and MP4 frames per second (default: 10)")
    parser.add_argument("--make_mp4", "--mp4", action="store_true", help=f"also render the images as {MP4_NAME}")
    parser.add_argument("--no_gif", action="store_true", help=f"don't stitch the images into {GIF_NAME}")
    parser.add_argument(
        "--decode-lyra",
        action="store_true",
        help="also run lyra.py on the same input, with its default settings (LYRA irradiance FITS files and a plot, "
        "in LYRA/ next to the dump or the folder)",
    )

    look = parser.add_argument_group("look")
    look.add_argument(
        "--color",
        choices=["tint", "none"],
        default="tint",
        help="tint the images with --tint (default), or keep them grey",
    )
    look.add_argument(
        "--tint", type=colour, default=render.GOLD, help=f"tint colour, #rrggbb (default: {render.GOLD})"
    )
    look.add_argument(
        "--stretch",
        type=float,
        default=render.STRETCH,
        help="contrast stretch: percent of the pixels clipped to black and to white, over all the images together "
        "(default: 0, the P2SC's brightness; 0.3 is the old ImageMagick look, which clips the brightest regions)",
    )
    look.add_argument(
        "--scaling",
        choices=("helioviewer", "quicklook", "p2sw"),
        default="helioviewer",
        help="look of the calibrated images: 'helioviewer' as the P2SC's JPEG 2000 images on Helioviewer (tone curve "
        "and sharpening; default), 'quicklook' as their quicklook PNGs and movies (plain square root, softer), "
        "'p2sw' as the SWAP team's p2sw_write_png (shows more faint corona)",
    )

    pipeline = parser.add_argument_group("pipeline")
    pipeline.add_argument(
        "--raw",
        action="store_true",
        help="no calibration: only orient and align the decoded images, keeping their own grey levels",
    )
    pipeline.add_argument("--fits", action="store_true", help="also keep the level 1 FITS file of every image")
    pipeline.add_argument(
        "--archive-headers",
        type=Path,
        help="folder of the archive's level 0 headers or FITS files: their detector temperature and LAR timing are "
        "used for the same images",
    )
    pipeline.add_argument(
        "--detector-temp",
        type=float,
        default=DETECTOR_TEMP,
        help=f"detector temperature (C) for the dark current of images without an archive header "
        f"(default: {DETECTOR_TEMP})",
    )
    args = parser.parse_args()

    if args.cadu and args.in_path and args.cadu != args.in_path:
        parser.error("give the input once: as an argument or with -i")
    args.in_path = args.in_path or args.cadu
    if args.in_path is None:
        parser.error("no input: give a .cadu dump or a folder of them (-i)")
    if not args.in_path.exists():
        parser.error(f"no such file or folder: {args.in_path}")
    if args.raw and args.fits:
        parser.error("--fits needs the calibration: it can't go with --raw")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if not 0 <= args.stretch < 50:
        parser.error("--stretch is a percentage per end, from 0 to under 50")
    if args.out_path is None:
        args.out_path = args.in_path.parent / OUT_DIR  # alongside the dump, or alongside the folder of dumps
    return args


def make_jobs(args: argparse.Namespace, cadu_files: list[Path]) -> tuple[list[Level1Job], int]:
    """One job per decodable image, in time order, and the number of images without a usable first packet."""
    images, clocks = collect(cadu_files)
    archive = load_archive_headers(args.archive_headers) if args.archive_headers else []
    jobs: list[Level1Job] = []
    names: set[str] = set()
    unreadable = 0
    for _, image in sorted(images.items()):
        for keep_bit_errors in (True, False):  # bit errors are usually harmless, but can wreck a decode
            assembled = assemble(image, keep_bit_errors)
            pixels = decode(assembled.jpeg) if assembled else None
            if pixels is not None:
                break
        if assembled is None or pixels is None:
            unreadable += 1
            continue
        start = clocks[image.dump].utc(assembled.header.start)
        name = base = image_name(start)
        count = 0
        while name in names:  # two images in the same second
            name = f"{base}-{count}"
            count += 1
        names.add(name)
        jobs.append(
            Level1Job(
                pixels,
                assembled.header,
                start,
                find_archive_header(archive, start),
                args.detector_temp,
                args.out_path,
                name,
                write_fits=args.fits,
                scaling=args.scaling,
                save_png=False,
            )
        )
    return jobs, unreadable


def main() -> int:
    args = parse_args()
    cadu_files = find_cadu_files([args.in_path])
    if not cadu_files:
        print(f"no .cadu files in {args.in_path}", file=sys.stderr)
        return 1
    print(f"reading {len(cadu_files)} dump{'s' * (len(cadu_files) > 1)}", file=sys.stderr)
    jobs, unreadable = make_jobs(args, cadu_files)
    args.out_path.mkdir(parents=True, exist_ok=True)

    names, notes, frames = [], [], []
    with ProcessPoolExecutor(max_workers=LEVEL1_WORKERS) as pool:  # a few seconds per image: side by side
        for job, (_, frame, note) in zip(jobs, pool.map(make_aligned if args.raw else make_level1, jobs), strict=True):
            if frame is None:
                print(f"{job.name}: {note}")
                continue
            names.append(job.name)
            notes.append(note)
            frames.append(frame)

    if frames:
        # the same stretch for every image (no flicker); the tint is the palette they're saved with
        low, high = render.stretch_limits(frames, args.stretch)
        frames = [render.apply_stretch(frame, low, high) for frame in frames]
        palette = render.tint_palette(args.tint) if args.color == "tint" else None
        for name, note, frame in zip(names, notes, frames, strict=True):
            path = args.out_path / f"{name}.png"
            render.save_png(frame, path, palette)
            print(f"{path} ({note})")
        if not args.no_gif:
            render.save_gif(frames, args.out_path / GIF_NAME, args.fps, palette)
            print(f"{args.out_path / GIF_NAME} ({len(frames)} frames)")
        if args.make_mp4:
            render.save_mp4(frames, args.out_path / MP4_NAME, args.fps, palette)
            print(f"{args.out_path / MP4_NAME} ({len(frames)} frames)")

    print(
        f"saved {len(frames)} {'aligned' if args.raw else 'calibrated'} images in {args.out_path}; "
        f"{len(jobs) - len(frames)} refused, {unreadable} without a usable first packet",
        file=sys.stderr,
    )

    if args.decode_lyra:
        import lyra  # only when asked: it pulls in the LYRA pipeline

        return lyra.main([str(args.in_path)])
    return 0


if __name__ == "__main__":
    sys.exit(main())

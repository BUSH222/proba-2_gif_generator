"""Calibrated SWAP images from one PROBA-2 .cadu dump: every image goes through the archive's level 1 pipeline
(`p2sw_prep.py`, a port of SolarSoft's p2sw_prep.pro: dark current, flat field, bad pixels, despiking, sun centered,
square pixels, solar north up, DN/s) and is saved as a PNG, by default with the brightness of the P2SC's quicklook
images and daily movies (--scaling p2sw: the SWAP team's p2sw_write_png, brighter off the disk).

With --raw the calibration is skipped: the decoded images keep their own grey levels and are only oriented and aligned
the way the pipeline does it (90 degree roll, sun center from the limb, mount offset and solar north, square pixels).

    python process_cadu.py cadus/proba2_dump.cadu                 # -> SWAP_pro/SWAP_<start of acquisition>.png
    python process_cadu.py cadus/proba2_dump.cadu --gif --mp4     # + SWAP_pro/swap.gif and swap.mp4, in time order

The detector temperature and the LAR timing are not in the dump. By default the dark current is computed for
DETECTOR_TEMP and images blurred by a LAR are kept; --archive-headers takes both from the archive's level 0 headers of
the same images where there are some (see the README, "Level 1").
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from extract_images import (
    DETECTOR_TEMP,
    GIF_NAME,
    LEVEL1_WORKERS,
    MP4_NAME,
    Level1Job,
    assemble,
    collect,
    decode,
    find_archive_header,
    image_name,
    load_archive_headers,
    make_aligned,
    make_level1,
    save_gif,
    save_mp4,
)

OUT_DIR = Path("SWAP_pro")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrated (level 1) SWAP PNG images from one PROBA-2 .cadu dump")
    parser.add_argument("cadu", type=Path, help="the .cadu dump")
    parser.add_argument("-o", "--out_path", type=Path, default=OUT_DIR, help=f"output folder (default: {OUT_DIR})")
    parser.add_argument("--gif", action="store_true", help=f"also stitch the images, in time order, into {GIF_NAME}")
    parser.add_argument("--mp4", action="store_true", help=f"also render the images, in time order, as {MP4_NAME}")
    parser.add_argument("--fits", action="store_true", help="also keep the level 1 FITS file of every image")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="no calibration: only orient and align the decoded images (90 degree roll, sun center, mount offset, "
        "solar north up), keeping their own grey levels",
    )
    parser.add_argument(
        "--scaling",
        choices=("quicklook", "p2sw"),
        default="quicklook",
        help="PNG brightness: 'quicklook' as the P2SC's quicklook images and daily movies (square root, dark "
        "background; default), 'p2sw' as the SWAP team's p2sw_write_png (1/4 power, shows more faint corona)",
    )
    parser.add_argument(
        "--archive-headers",
        type=Path,
        help="folder of the archive's level 0 headers or FITS files: their detector temperature and LAR timing are "
        "used for the same images",
    )
    parser.add_argument(
        "--detector-temp",
        type=float,
        default=DETECTOR_TEMP,
        help=f"detector temperature (C) for the dark current of images without an archive header "
        f"(default: {DETECTOR_TEMP})",
    )
    args = parser.parse_args()
    if not args.cadu.is_file():
        parser.error(f"not a file: {args.cadu}")
    if args.raw and args.fits:
        parser.error("--fits needs the calibration: it can't go with --raw")

    images, clocks = collect([args.cadu])
    archive = load_archive_headers(args.archive_headers) if args.archive_headers else []
    args.out_path.mkdir(parents=True, exist_ok=True)

    jobs: list[Level1Job] = []
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
        while name in {job.name for job in jobs}:  # two images in the same second
            name = f"{base}-{sum(job.name.startswith(base) for job in jobs)}"
        telemetry = find_archive_header(archive, start)
        jobs.append(
            Level1Job(
                pixels,
                assembled.header,
                start,
                telemetry,
                args.detector_temp,
                args.out_path,
                name,
                write_fits=args.fits,
                scaling=args.scaling,
            )
        )

    frames = []
    with ProcessPoolExecutor(max_workers=LEVEL1_WORKERS) as pool:  # a few seconds per image: side by side
        for job, (path, frame, note) in zip(
            jobs, pool.map(make_aligned if args.raw else make_level1, jobs), strict=True
        ):
            print(f"{path or job.name} ({note})")
            if frame is not None:
                frames.append(frame)

    if frames:
        for enabled, name, save in ((args.gif, GIF_NAME, save_gif), (args.mp4, MP4_NAME, save_mp4)):
            if enabled:
                save(frames, args.out_path / name)
                print(f"{args.out_path / name} ({len(frames)} frames)")

    print(
        f"saved {len(frames)} {'aligned' if args.raw else 'calibrated'} images; "
        f"{len(jobs) - len(frames)} refused, "
        f"{unreadable} without a usable first packet",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

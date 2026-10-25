"""PROBA-2 LYRA irradiance from raw .cadu dumps (SatDump): FITS files of levels 0 to 2 and a plot.

    uv run lyra.py demo_lyra.cadu                 # -> LYRA/ next to the dump: level 0, 1 and 2 FITS + a level 2 plot
    uv run lyra.py -i cadus --level 1             # every dump in the folder, levels 0 and 1, plot of level 1
    uv run lyra.py demo_lyra.cadu --refresh       # first take the day's degradation correction from the archive

LYRA's samples are in APID 21, at its full cadence of 20 Hz. Level 0 is the raw counts per integration, level 1 the
counts per millisecond (kHz) and level 2 the irradiance in W/m², as the PROBA2 Science Center's lev1_std and lev2_std
files (the same samples, to about one count; see lyra_pipeline/NOTES.md). Every level up to --level is written as FITS,
and --level itself is plotted.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from lyra_pipeline import archive, plot, products
from lyra_pipeline.calibrate import Calibration, sun_d2
from lyra_pipeline.decode import LyraDump, read_dump
from swap_pipeline.extract_images import find_cadu_files

OUT_DIR = "LYRA_pro"
LEVELS = (0, 1, 2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PROBA-2 LYRA irradiance (levels 0 to 2) from .cadu dumps.")
    parser.add_argument("cadu", nargs="?", type=Path, help="a .cadu dump, or a folder of them (same as -i)")
    parser.add_argument("-i", "--in_path", type=Path, help="a .cadu dump, or a folder of them")
    parser.add_argument(
        "-o", "--out_path", type=Path, help=f"output folder (default: {OUT_DIR} next to the dump or the folder)"
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        choices=LEVELS,
        default=2,
        help="data level: 0 raw counts, 1 kHz, 2 irradiance in W/m² (default: 2). FITS files are written for every "
        "level up to it, and it is plotted",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="before calibrating, fetch the dump day's degradation correction from the PROBA2 archive (internet; the "
        "archive has a day a day or two later) and add it to lyra_pipeline/lyra_caldb/degradation.csv",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="temperature of LYRA's unit 2 in C, for a dump without the housekeeping reports that give it (default: "
        "the calibration's typical value)",
    )
    parser.add_argument("--no_plot", action="store_true", help="only write the FITS files")
    args = parser.parse_args(argv)

    if args.cadu and args.in_path and args.cadu != args.in_path:
        parser.error("give the input once: as an argument or with -i")
    args.in_path = args.in_path or args.cadu
    if args.in_path is None:
        parser.error("no input: give a .cadu dump or a folder of them (-i)")
    if not args.in_path.exists():
        parser.error(f"no such file or folder: {args.in_path}")
    if args.out_path is None:
        args.out_path = args.in_path.parent / OUT_DIR
    return args


def unit_temperature(dump: LyraDump, calibration: Calibration, fallback: float | None) -> tuple[np.ndarray, str]:
    """Unit 2 temperature at every sample, and where it came from: "housekeeping" or "assumed"."""
    if len(dump.report_seconds) == 0:
        celsius = calibration.default_temperature if fallback is None else fallback
        print(f"{dump.name}: no temperature reports, assuming {celsius} C (--temperature)", file=sys.stderr)
        return np.full(dump.seconds.shape, celsius), "assumed"
    reports = calibration.celsius(dump.report_raw)
    low, high = calibration.temperature_range
    if reports.min() < low - 1 or reports.max() > high + 1:
        print(
            f"{dump.name}: temperature {reports.min():.1f} to {reports.max():.1f} C, outside the {low}-{high} C "
            "where its conversion was fitted",
            file=sys.stderr,
        )
    return np.interp(dump.seconds, dump.report_seconds, reports), "housekeeping"


def process(dump: LyraDump, args: argparse.Namespace, calibration: Calibration) -> list[Path]:
    levels: dict[int, np.ndarray] = {0: dump.counts}
    integration = {"INTEGRAT": (calibration.integration_ms, "ms, integration time of a sample")}
    keywords: dict[int, dict[str, tuple[object, str]]] = {0: dict(integration), 1: dict(integration), 2: {}}
    history: list[str] = []
    note = ""
    if args.level >= 1:
        levels[1] = calibration.level1(dump.counts)
    if args.level >= 2:
        if dump.day < calibration.epoch:
            print(
                f"{dump.name}: {dump.day} is before {calibration.epoch}, when the archive's calibration changed; "
                "level 2 may be off by a few tenths of a percent (--refresh takes the day's G, which helps)",
                file=sys.stderr,
            )
        temperature, temperature_source = unit_temperature(dump, calibration, args.temperature)
        degradation = calibration.degradation_for(dump.day)
        levels[2] = calibration.level2(levels[1], temperature, sun_d2(dump.day, dump.seconds), degradation.values)
        keywords[2] = {
            "CAL_EPOC": (f"{calibration.epoch}", "calibration constants valid from this day"),
            "TEMPSRC": (temperature_source, "unit 2 temperature: housekeeping or assumed"),
            "TEMP": (round(float(np.mean(temperature)), 3), "C, mean unit 2 temperature"),
            "DEGRSRC": (degradation.source, "G: table, interpolated or nearest"),
            "DEGRGAP": (degradation.gap_days, "days from the nearest day of the G table"),
        }
        for channel, value in enumerate(degradation.values):
            keywords[2][f"DEGR_G{channel + 1}"] = (float(f"{value:.10e}"), f"W/m**2, G of CHANNEL{channel + 1}")
        history = [
            "level 2 = k (d/1 AU)^2 (level 1 - D0 - A exp(B T)) + G",
            "the numbers: lyra_pipeline/lyra_caldb, see lyra_pipeline/NOTES.md",
            degradation.note,
        ]
        note = f"{temperature_source} temperature, {float(np.mean(temperature)):.2f} C; {degradation.note}"
        print(f"{dump.name}: {degradation.note}", file=sys.stderr)

    written = []
    for level in range(args.level + 1):
        product = products.Product(level, levels[level], keywords[level], history if level == 2 else [])
        written.append(products.write(args.out_path, dump, product))
    if not args.no_plot:
        path = args.out_path / f"lyra_{dump.start:%Y%m%d-%H%M%S}_lev{args.level}.png"
        written.append(plot.plot(path, dump, args.level, levels[args.level], note if args.level == 2 else ""))
    return written


def main(argv: list[str] | None = None) -> int:
    """argv: the command line arguments (default: sys.argv[1:])."""
    args = parse_args(argv)
    cadu_files = find_cadu_files([args.in_path])
    if not cadu_files:
        print(f"no .cadu files in {args.in_path}", file=sys.stderr)
        return 1
    calibration = Calibration.load()
    found = 0
    for path in cadu_files:
        print(f"reading {path}", file=sys.stderr)
        dump = read_dump(path)
        if dump is None:
            print(f"{path.name}: no LYRA data", file=sys.stderr)
            continue
        found += 1
        print(
            f"{path.name}: {dump.counts.shape[1]} samples per channel, {dump.start:%Y-%m-%d %H:%M:%S} to "
            f"{dump.end:%H:%M:%S} UTC"
            + (f", {dump.skipped} packets left out (bad CRC or not samples)" if dump.skipped else ""),
            file=sys.stderr,
        )
        if not dump.gps_clock:
            print(f"{path.name}: no GPS records, the times may be off by a minute", file=sys.stderr)
        if args.refresh and args.level >= 2:
            print(f"{path.name}: {archive.refresh(calibration, dump.day)}", file=sys.stderr)
        args.out_path.mkdir(parents=True, exist_ok=True)
        for written in process(dump, args, calibration):
            print(written)
    if not found:
        print("no LYRA data in the input", file=sys.stderr)
        return 1
    print(f"LYRA data of {found} dump{'s' * (found > 1)} in {args.out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

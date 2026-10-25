"""FITS files of LYRA levels 0, 1 and 2, in the layout of the archive's `lev1_std` and `lev2_std` files.

TIME is in seconds of the UTC day, as in the archive. DATE-OBS is the midnight that starts that day, because sunpy's
LYRA reader adds TIME to DATE-OBS; the first and last samples are in DATE-BEG and DATE-END. Level 0 has no archive
counterpart: the same layout with the raw counts per integration.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

from .decode import CHANNEL_NAMES, CHANNELS

if TYPE_CHECKING:
    from pathlib import Path

    from .decode import LyraDump

EXTENSIONS = {0: "COUNTS LEVEL 0", 1: "FREQ LEVEL 1", 2: "IRRAD LEVEL 2"}
UNITS = {0: "counts", 1: "kHz", 2: "W/m**2"}
UNIT_NAMES = {0: "counts per integration", 1: "kiloHertz", 2: "watt per square metre"}
FORMATS = {0: "J", 1: "D", 2: "D"}


@dataclass
class Product:
    level: int
    values: np.ndarray  # (4, n)
    keywords: dict[str, tuple[object, str]]  # extra primary header keywords: name -> (value, comment)
    history: list[str] = field(default_factory=list)


def file_stem(dump: LyraDump, level: int) -> str:
    return f"lyra_{dump.start:%Y%m%d-%H%M%S}_lev{level}_std"


def primary_header(dump: LyraDump, product: Product, filename: str) -> fits.Header:
    midnight = datetime(dump.day.year, dump.day.month, dump.day.day, tzinfo=UTC)
    header = fits.Header()
    header["ORIGIN"] = ("proba-2_gif_generator", "lyra.py, from a raw .cadu dump")
    header["TELESCOP"] = "PROBA2"
    header["INSTRUME"] = "LYRA"
    header["OBJECT"] = "EUV solar irrad"
    header["OBS_MODE"] = ("nominal standard", "nominal unit (unit 2), standard acquisition")
    header["DATE"] = (f"{datetime.now(UTC):%Y-%m-%d}", "FITS file creation date (yyyy-mm-dd)")
    header["DATE-OBS"] = (f"{midnight:%Y-%m-%dT%H:%M:%S.%f}", "UTC start of the day of TIME")
    header["DATE-BEG"] = (f"{dump.start:%Y-%m-%dT%H:%M:%S.%f}", "UTC of the first sample")
    header["DATE-END"] = (f"{dump.end:%Y-%m-%dT%H:%M:%S.%f}", "UTC of the last sample")
    header["DATASRC"] = (dump.name, "SatDump .cadu dump")
    header["LEVEL"] = (str(product.level), "Calibration level")
    header["FILENAME"] = (filename, "Name of this FITS file")
    header["CLOCK"] = ("GPS" if dump.gps_clock else "fixed offset", "tick to UTC: GPS records (APID 18) of the dump")
    for name, (value, comment) in product.keywords.items():
        header[name] = (value, comment)
    for line in product.history:
        header.add_history(line)
    return header


def table_hdu(dump: LyraDump, product: Product) -> fits.BinTableHDU:
    level = product.level
    values = product.values.astype(np.int32 if level == 0 else np.float64)
    columns = [fits.Column(name="TIME", format="D", unit="s", array=dump.seconds)]
    columns += [
        fits.Column(name=f"CHANNEL{n + 1}", format=FORMATS[level], unit=UNITS[level], array=values[n])
        for n in range(CHANNELS)
    ]
    if level == 2:
        warning = np.char.add(dump.qfactor.astype(str), "0000")  # "30000", or "40000" at a time mark, as the archive
        columns.append(fits.Column(name="WARNING", format="5A", array=warning))
    else:
        columns.append(fits.Column(name="QFACTOR", format="B", array=dump.qfactor))
    hdu = fits.BinTableHDU.from_columns(columns, name=EXTENSIONS[level])
    assert isinstance(hdu, fits.BinTableHDU)
    hdu.header.comments["TTYPE1"] = "TIME in seconds of the day"
    for n in range(CHANNELS):
        hdu.header.comments[f"TTYPE{n + 2}"] = f"CHANNEL{n + 1}: {CHANNEL_NAMES[n]}"
        hdu.header.comments[f"TUNIT{n + 2}"] = UNIT_NAMES[level]
    last = CHANNELS + 2
    hdu.header.comments[f"TTYPE{last}"] = (
        "WARNING: quality of time and channels" if level == 2 else "QFACTOR: TIME quality factor (4 at a time mark)"
    )
    return hdu


def write(folder: Path, dump: LyraDump, product: Product) -> Path:
    path = folder / f"{file_stem(dump, product.level)}.fits"
    primary = fits.PrimaryHDU(header=primary_header(dump, product, path.name))
    fits.HDUList([primary, table_hdu(dump, product)]).writeto(path, overwrite=True)
    return path

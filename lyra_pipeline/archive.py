"""The degradation correction of a day from the PROBA2 Science Center's LYRA archive (for lyra.py --refresh).

G is the one number of the calibration the archive changes as it goes, and the dump alone can't give it. A day's
level 1 and level 2 files hold the same samples before and after the calibration, so G is level 2 minus the rest of
the formula. Four stretches of 250 s of each file, read with HTTP range requests from their binary tables, and the
day's met file (temperature and Sun distance) are enough: 2 MB instead of 150.
"""

import io
import sys
import time
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from astropy.io import fits

from .calibrate import AU_KM, Calibration

ARCHIVE = "https://proba2.sidc.be/lyra/data"
CHUNK_ROWS = 5000  # 250 s at 20 Hz
CHUNK_STARTS = (0.1, 0.35, 0.6, 0.85)  # fractions of the day's rows
BLOCK = 2880
CARD = 80
ROW = {
    "lev1_std": np.dtype([("TIME", ">f8"), ("CHANNELS", ">f8", 4), ("QFACTOR", "u1")]),
    "lev2_std": np.dtype([("TIME", ">f8"), ("CHANNELS", ">f8", 4), ("WARNING", "S5")]),
}
FILL_STEP = 10  # days between the days added to the table when the dump's own day isn't in the archive yet
FILL_MAX = 12


def url(day: date, kind: str) -> str:
    folder = "bsd" if kind.startswith("lev2") else "eng"  # level 2 is in `bsd`, level 1 and the met files in `eng`
    return f"{ARCHIVE}/{folder}/{day:%Y/%m/%d}/lyra_{day:%Y%m%d}-000000_{kind}.fits"


def fetch(address: str, first: int | None = None, last: int | None = None, tries: int = 4) -> bytes:
    headers = {} if first is None else {"Range": f"bytes={first}-{last}"}
    for attempt in range(tries):
        try:
            with urlopen(Request(address, headers=headers), timeout=60) as response:
                data = response.read()
                assert isinstance(data, bytes)
                return data
        except HTTPError:
            raise
        except URLError, TimeoutError:  # the server resets a connection now and then
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def table_rows(address: str, kind: str) -> np.ndarray:
    """Stretches of CHUNK_ROWS rows of the binary table of a std file, spread over the day."""
    head = fetch(address, 0, 4 * BLOCK - 1)
    cards = [head[i : i + CARD] for i in range(0, len(head), CARD)]
    ends = [i for i, card in enumerate(cards) if card.startswith(b"END ")]
    if len(ends) < 2:
        raise ValueError(f"no table header in the first {4 * BLOCK} bytes of {address}")
    data_start = (ends[1] * CARD // BLOCK + 1) * BLOCK  # the table starts at the block after the second END
    rows = next(int(card[10:30]) for card in cards[ends[0] :] if card.startswith(b"NAXIS2"))
    size = ROW[kind].itemsize
    parts = []
    for fraction in CHUNK_STARTS:
        first = int(fraction * max(rows - CHUNK_ROWS, 0))
        count = min(CHUNK_ROWS, rows)
        data = fetch(address, data_start + first * size, data_start + (first + count) * size - 1)
        parts.append(np.frombuffer(data, dtype=ROW[kind]))
    return np.concatenate(parts)


def day_degradation(calibration: Calibration, day: date) -> list[float] | None:
    """G of the 4 channels on a day, from the archive; None if the archive doesn't have the day."""
    try:
        level1 = table_rows(url(day, "lev1_std"), "lev1_std")
        level2 = table_rows(url(day, "lev2_std"), "lev2_std")
        with fits.open(io.BytesIO(fetch(url(day, "lev1_met")))) as hdus:
            table = hdus[1].data
            met_time = np.asarray(table["TIME"], dtype=np.float64)
            temperature = np.asarray(table["TEMPERATURE"], dtype=np.float64)[:, 1]  # unit 2
            distance = np.asarray(table["DISTANCE"], dtype=np.float64)
    except HTTPError as error:
        if error.code in (404, 416):
            return None
        raise
    if not np.array_equal(level1["TIME"], level2["TIME"]):
        return None
    khz = level1["CHANNELS"].astype(np.float64).T
    irradiance = level2["CHANNELS"].astype(np.float64).T
    good = np.isfinite(khz).all(axis=0) & np.isfinite(irradiance).all(axis=0)
    seconds = level1["TIME"].astype(np.float64)[good]
    d2 = (np.interp(seconds, met_time, distance) / AU_KM) ** 2
    celsius = np.interp(seconds, met_time, temperature)
    rest = calibration.level2(khz[:, good], celsius, d2, [0.0] * 4)
    return [float(value) for value in np.mean(irradiance[:, good] - rest, axis=1)]


def refresh(calibration: Calibration, day: date) -> str:
    """Add the day's G to the table from the archive. If the archive doesn't have the day yet, add the latest days it
    has back to the end of the table instead. Saves the table; returns what was done."""
    print(f"refresh: asking the archive for {day}", file=sys.stderr)
    if (values := day_degradation(calibration, day)) is not None:
        calibration.degradation[day] = values
        calibration.save_degradation()
        return f"G of {day} from the archive, added to the table"
    last = max(calibration.degradation)
    if day <= last:
        return f"the archive has no {day}; G stays interpolated from the table"
    added: list[date] = []
    candidate = min(day, date.today()) - timedelta(days=1)
    while candidate > last and len(added) < FILL_MAX:
        if (values := day_degradation(calibration, candidate)) is not None:
            calibration.degradation[candidate] = values
            added.append(candidate)
            print(f"refresh: added {candidate}", file=sys.stderr)
            candidate -= timedelta(days=FILL_STEP)
        else:
            candidate -= timedelta(days=1)
    if not added:
        return f"the archive has nothing for {day} or after {last} yet"
    calibration.save_degradation()
    return f"the archive has no {day} yet; added {len(added)} later days to the table, the last {added[0]}"

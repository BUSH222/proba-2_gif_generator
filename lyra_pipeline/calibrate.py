"""LYRA level 0 (counts) to level 1 (kHz) to level 2 (W/m²), as the PROBA2 Science Center's archive does it.

    level 1 = counts / integration time, rounded to 6 decimals (the archive's numbers, bit for bit)
    level 2 = k (d / 1 AU)² (level 1 - D0 - A exp(B T)) + G

d is the Sun distance, T the temperature of LYRA's nominal unit (unit 2), D0 + A exp(B T) its dark current in kHz,
k the conversion to W/m² and G the degradation correction, which the archive updates as it goes. The numbers were
fitted on pairs of the archive's level 1 and level 2 files (calibrate_lyra.ipynb, experimental branch) and are kept in
lyra_caldb/: calibration.json for the constants, degradation.csv for G on sampled days. See NOTES.md.
"""

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
from astropy.time import Time, TimeDelta
from sunpy.coordinates import sun

CALDB = Path(__file__).parent / "lyra_caldb"
AU_KM = 149_597_870.7
STALE_DAYS = 30  # beyond this from the nearest day of the table, G is flagged as a guess


@dataclass(frozen=True)
class Channel:
    k: float  # W/m² per kHz at 1 AU
    d0: float  # kHz: constant part of the dark current
    a: float  # kHz
    b: float  # 1/C

    def dark(self, temperature: np.ndarray) -> np.ndarray:
        return self.d0 + self.a * np.exp(self.b * temperature)


@dataclass(frozen=True)
class Degradation:
    values: list[float]  # G of the 4 channels, W/m²
    source: str  # "table", "interpolated" or "nearest"
    gap_days: int  # from the nearest day of the table
    note: str


@dataclass
class Calibration:
    integration_ms: float
    temperature_poly: list[float]  # raw word of unit 2 -> C (numpy polyval order)
    temperature_range: tuple[float, float]  # C: where that conversion was fitted
    default_temperature: float  # C: for a dump without temperature reports
    epoch: date  # first day the constants are good for
    channels: list[Channel]
    degradation: dict[date, list[float]]  # day -> G of the 4 channels, W/m²
    folder: Path = CALDB

    @classmethod
    def load(cls, folder: Path = CALDB) -> Calibration:
        constants = json.loads((folder / "calibration.json").read_text())
        degradation = {}
        with (folder / "degradation.csv").open() as file:
            for row in csv.DictReader(line for line in file if not line.startswith("#")):
                degradation[date.fromisoformat(row["date"])] = [float(row[f"G{n}"]) for n in range(1, 5)]
        return cls(
            integration_ms=constants["integration_ms"],
            temperature_poly=constants["temperature_poly"],
            temperature_range=tuple(constants["temperature_range"]),
            default_temperature=constants["default_temperature"],
            epoch=date.fromisoformat(constants["epoch"]),
            channels=[Channel(**channel) for channel in constants["channels"]],
            degradation=dict(sorted(degradation.items())),
            folder=folder,
        )

    def save_degradation(self) -> None:
        path = self.folder / "degradation.csv"
        header = [line for line in path.read_text().splitlines() if line.startswith("#")]
        with path.open("w", newline="") as file:
            file.write("".join(line + "\n" for line in header))
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["date", "G1", "G2", "G3", "G4"])
            for day, values in sorted(self.degradation.items()):
                writer.writerow([day.isoformat(), *(f"{value:.10e}" for value in values)])

    def celsius(self, raw: np.ndarray) -> np.ndarray:
        return np.polyval(self.temperature_poly, raw)

    def degradation_for(self, day: date) -> Degradation:
        """G of a day: from the table, interpolated between its days, or its nearest day outside it."""
        days = sorted(self.degradation)
        gap = min(abs((day - d).days) for d in days)
        if day in self.degradation:
            return Degradation(self.degradation[day], "table", 0, f"G of {day} from the table")
        x = np.array([(d - self.epoch).days for d in days], dtype=float)
        target = (day - self.epoch).days
        values = [float(np.interp(target, x, [self.degradation[d][c] for d in days])) for c in range(4)]
        if days[0] < day < days[-1]:
            return Degradation(
                values, "interpolated", gap, f"G interpolated, {plural(gap, 'day')} from a day of the table"
            )
        note = f"G of {days[-1] if day > days[-1] else days[0]}, the table's nearest day, {plural(gap, 'day')} away"
        if gap > STALE_DAYS:
            note += " (a guess: --refresh fetches the day's own from the archive)"
        return Degradation(values, "nearest", gap, note)

    def level1(self, counts: np.ndarray) -> np.ndarray:
        return np.round(counts / self.integration_ms, 6)

    def level2(self, khz: np.ndarray, temperature: np.ndarray, d2: np.ndarray, g: list[float]) -> np.ndarray:
        out = np.empty(khz.shape, dtype=np.float64)
        for index, channel in enumerate(self.channels):
            out[index] = channel.k * d2 * (khz[index] - channel.dark(temperature)) + g[index]
        return out


def plural(count: int, word: str) -> str:
    return f"{count} {word}{'s' * (count != 1)}"


def sun_d2(day: date, seconds: np.ndarray) -> np.ndarray:
    """(d / 1 AU)² at the given seconds of a UTC day, d the Earth's distance from the Sun (PROBA2's is within 1e-5)."""
    grid = np.arange(np.min(seconds) - 60, np.max(seconds) + 120, 60.0)
    times = Time(datetime(day.year, day.month, day.day), scale="utc") + TimeDelta(grid, format="sec")
    return (np.interp(seconds, grid, sun.earth_distance(times).to_value("km")) / AU_KM) ** 2

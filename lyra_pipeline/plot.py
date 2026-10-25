"""A plot of one level of a dump: the four channels stacked, at full cadence with a 10 s mean over it."""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .decode import CHANNEL_NAMES, CHANNELS

if TYPE_CHECKING:
    from pathlib import Path

    from .decode import LyraDump

UNITS = {0: "counts", 1: "kHz", 2: "W/m²"}
TITLES = {0: "level 0, raw counts", 1: "level 1, count rate", 2: "level 2, irradiance"}
MEAN_SECONDS = 10
GAP_SECONDS = 1.0  # a longer step between samples breaks the line

# light surface, recessive grid, ink-coloured text; one hue in two steps for the samples and their mean
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e2e0da", "#fcfcfb"
SAMPLES_COLOUR, MEAN_COLOUR = "#9ec5f4", "#184f95"
STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "text.color": INK,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def with_gaps(seconds: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN between samples further apart than GAP_SECONDS, so a gap isn't drawn as a straight line."""
    breaks = np.flatnonzero(np.diff(seconds) > GAP_SECONDS) + 1
    return np.insert(seconds, breaks, np.nan), np.insert(values.astype(np.float64), breaks, np.nan)


def running_mean(seconds: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Means over whole MEAN_SECONDS bins, at the bin centres; NaN for empty bins."""
    index = np.floor((seconds - seconds[0]) / MEAN_SECONDS).astype(int)
    totals, number = np.bincount(index, values), np.bincount(index)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = totals / number
    return seconds[0] + (np.arange(len(totals)) + 0.5) * MEAN_SECONDS, means


def power_of_ten(values: np.ndarray) -> int:
    """The power of ten that brings small values (W/m²) to between 1 and 10; 0 for values from 0.01 up."""
    typical = float(np.nanmedian(np.abs(values)))
    return int(np.floor(np.log10(typical))) if 0 < typical < 1e-2 else 0


def superscript(number: int) -> str:
    return str(number).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def as_dates(day_start: datetime, seconds: np.ndarray) -> np.ndarray:
    """Matplotlib date numbers (days), NaN kept."""
    return mdates.date2num(day_start) + seconds / 86400


def plot(path: Path, dump: LyraDump, level: int, values: np.ndarray, note: str = "") -> Path:
    plt.switch_backend("Agg")  # files only, no window
    midnight = datetime(dump.day.year, dump.day.month, dump.day.day, tzinfo=UTC)
    with plt.rc_context(STYLE):  # ty: ignore[invalid-argument-type] - plain names of rcParams
        fig, axes = plt.subplots(CHANNELS, 1, figsize=(12, 9.5), sharex=True)
        for channel, ax in enumerate(axes):
            power = power_of_ten(values[channel])
            scaled = values[channel] / 10.0**power
            x, y = with_gaps(dump.seconds, scaled)
            ax.plot(as_dates(midnight, x), y, color=SAMPLES_COLOUR, linewidth=0.5)
            x, y = running_mean(dump.seconds, scaled)
            ax.plot(as_dates(midnight, x), y, color=MEAN_COLOUR, linewidth=1.4)
            unit = f"10{superscript(power)} {UNITS[level]}" if power else UNITS[level]
            ax.set_ylabel(f"{CHANNEL_NAMES[channel]}\n({unit})")
            ax.ticklabel_format(axis="y", useOffset=False, style="plain")
        handles = [
            Line2D([], [], color=SAMPLES_COLOUR, linewidth=4, label="20 Hz samples"),
            Line2D([], [], color=MEAN_COLOUR, linewidth=2, label=f"{MEAN_SECONDS} s mean"),
        ]
        fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.99, 0.995))
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1].set_xlabel(f"UTC, {dump.day:%Y-%m-%d}")
        span = f"{dump.start:%Y-%m-%d %H:%M} to {(dump.end + timedelta(seconds=30)):%H:%M} UTC"
        fig.suptitle(f"LYRA {TITLES[level]}, {span}", color=INK, fontsize=12, x=0.01, ha="left")
        footer = f"{dump.name}" + (f" · {note}" if note else "")
        fig.text(0.01, 0.005, footer, color=INK_2, fontsize=8, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return path

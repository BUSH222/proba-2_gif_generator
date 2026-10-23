"""Python port of p2sw_prep.pro, the PROBA2/SWAP calibration pipeline of SolarSoft (v1.5): level 0 FITS to level 1.

Every function below is one of the IDL routines in `swap_prep/` and keeps its name, its order in the pipeline and its
arithmetic (float32 where IDL works in float, IDL's rounding, medians and edge handling), so that the archive's level 1
images come out again. The calibration files are those of `swap_caldb/` (`$SSW/proba2/swap/caldb`).

Arrays are indexed [row, column] = [y, x], as astropy reads FITS: IDL's image[x, y] is image[y, x] here. Pixel
coordinates (sun center, transformation matrices) are (x, y) = (column, row), 0-based, as in IDL.

The defaults are those of the P2SC pipeline that makes the archive (`/DESPIKE, /NORMALIZE, /FLOAT`):

    python p2sw_prep.py swap_lv0_*.fits -o level1
"""

import argparse
import math
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from scipy import ndimage
from scipy.io import readsav

VERSION = "v1.5"  # of p2sw_prep.pro, written to CREATOR and HISTORY
CALDB = Path(__file__).resolve().parent / "swap_caldb"
SIZE = 1024  # detector side

F32 = np.float32


class PrepError(Exception):
    """The image can't be (or shouldn't be) made into level 1: a LAR, a calibration image, missing information."""


@dataclass
class Options:
    """The keywords of P2SW_PREP (and of the routines it passes them to). The defaults are the archive's."""

    caldir: Path = CALDB
    dark_form: str = "temp"  # 'temp': dark modelled from the detector temperature, 'fits': a dark frame
    no_calibration: bool = False
    no_pmcdiv: bool = False
    no_pixlist: bool = False
    no_satrep: bool = False
    no_dark: bool = False
    no_pmcrep: bool = False
    no_flat: bool = False
    despike: bool = True
    spike_sig: float = 6.0
    apply_gentle_filter: bool = False
    psf_correct: bool = False
    remove_horizontal: bool = False
    no_sun_center: bool = False
    no_rescale: bool = False
    no_rotate: bool = False
    no_solar_ep: bool = False
    no_swap_mount_offset: bool = False
    preserve_radiometric: bool = False
    force_sun_center: bool = False
    sun_cen_x: float | None = None  # a sun center (0-based) to use instead of the limb fit
    sun_cen_y: float | None = None
    normalize: bool = True
    keep_float: bool = True  # IDL's /FLOAT: no rounding to integers
    permit_negatives: bool = False
    ignore_lar: bool = False
    force_calibrate: bool = False
    temp_dt_override: bool = False


# ---------------------------------------------------------------------------------------------------------------------
# IDL building blocks


def idl_round(values):
    """IDL's ROUND: halves away from zero."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def idl_median_filter(image: np.ndarray, width: int) -> np.ndarray:
    """IDL's MEDIAN(image, width): the pixels within width / 2 of the edges are left as they are."""
    filtered = ndimage.median_filter(image, size=width, mode="nearest")
    half = width // 2
    filtered[:half], filtered[-half:] = image[:half], image[-half:]
    filtered[:, :half], filtered[:, -half:] = image[:, :half], image[:, -half:]
    return filtered


def idl_nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    """IDL's MEDIAN(values, DIMENSION=...): NaN left out, and the upper of the two middle values for an even count."""
    ordered = np.sort(values, axis=axis)  # NaN last
    count = np.sum(~np.isnan(values), axis=axis, keepdims=True)
    middle = np.take_along_axis(ordered, np.minimum(count // 2, values.shape[axis] - 1), axis=axis)
    return np.where(count > 0, middle, np.nan).squeeze(axis)


def keys_weights(f: np.ndarray, a: float = -0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weights of the samples at -1, 0, 1 and 2 for a point a fraction f in [0, 1) past sample 0: the cubic
    convolution kernel of IDL's CUBIC=-0.5 (INTERPOLATE and POLY_2D)."""
    f2 = f * f
    f3 = f2 * f
    return (
        a * (f3 - 2 * f2 + f),
        (a + 2) * f3 - (a + 3) * f2 + 1,
        -(a + 2) * f3 + (2 * a + 3) * f2 - a * f,
        a * (f2 - f3),
    )


def interpolate_cubic(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, missing: float = 0.0, poly_2d: bool = False
) -> np.ndarray:
    """IDL's INTERPOLATE(image, x, y, CUBIC=-0.5, MISSING=missing): points outside [0, n - 1] get `missing`. With
    poly_2d, POLY_2D's bounds: [0, n), the last half-open pixel extrapolated (measured on the archive's images)."""
    rows, cols = image.shape
    if poly_2d:
        outside = ~((x >= 0) & (x < cols) & (y >= 0) & (y < rows))  # NaN is outside too
    else:
        outside = ~((x >= 0) & (x <= cols - 1) & (y >= 0) & (y <= rows - 1))
    x = np.where(outside, 0.0, x)
    y = np.where(outside, 0.0, y)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    weights_x, weights_y = keys_weights(x - x0), keys_weights(y - y0)
    columns = [np.clip(x0 + i, 0, cols - 1) for i in range(-1, 3)]
    result = np.zeros(x.shape)
    for j, weight_y in enumerate(weights_y):
        row = np.clip(y0 + j - 1, 0, rows - 1)
        across = sum(weight_x * image[row, column] for weight_x, column in zip(weights_x, columns, strict=True))
        result += weight_y * across
    result[outside] = missing
    return result


def poly_2d_coordinates(inverse: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Input coordinates of every output pixel for POLY_2D with a linear (degree 1) transformation. IDL starts each
    row exactly and then adds the x step in float32, so the coordinates drift by up to ~0.01 pixel along a row; the
    archive's images have that drift (with it 99.6 % of their pixels come out identical, without it 58 %)."""
    y = np.arange(shape[0], dtype=np.float64)
    coordinates = []
    for axis in (0, 1):
        start, step = inverse[axis, 1] * y + inverse[axis, 2], inverse[axis, 0]
        increments = np.empty(shape, dtype=F32)
        increments[:, 0] = start
        increments[:, 1:] = F32(step)
        coordinates.append(np.cumsum(increments, axis=1, dtype=F32).astype(np.float64))
    return coordinates[0], coordinates[1]


def rebin_2x2(array: np.ndarray) -> np.ndarray:
    """REBIN(REBIN(a, nx / 2, ny / 2), nx, ny, /SAMPLE): the means of 2 x 2 blocks, each spread back over its block."""
    rows, cols = array.shape
    means = array.reshape(rows // 2, 2, cols // 2, 2).mean(axis=(1, 3), dtype=array.dtype)
    return np.repeat(np.repeat(means, 2, axis=0), 2, axis=1)


# ---------------------------------------------------------------------------------------------------------------------
# Header helpers


def date_obs(header: fits.Header) -> Time:
    return Time(header["DATE-OBS"], scale="utc")


def window(header: fits.Header) -> tuple[slice, slice]:
    """The read-out detector area as [rows, columns]."""
    return slice(header["FIRSTROW"] - 1, header["LAST_ROW"]), slice(header["FIRSTCOL"] - 1, header["LAST_COL"])


def rebinned(header: fits.Header) -> bool:
    return str(header.get("REBIN", "off")).strip().upper().startswith("ON")


def capture_mode(header: fits.Header) -> str:
    return str(header["CAP_MODE"]).strip().lower()


def update_history(header: fits.Header, routine: str, text: str) -> None:
    """UPDATE_HISTORY, as it appears in the archive's files: 'p2sw_drksub v1.5  Subtracted ...'."""
    header.add_history(f"{routine.lower()} {VERSION}  {text}")


def previously(header: fits.Header, routine: str) -> bool:
    """GET_HISTORY: whether the routine has already been applied to this file."""
    return any(str(line).lower().startswith(routine.lower() + " ") for line in header.get("HISTORY", []))


def calibration_file(caldir: Path, prefix: str, header: fits.Header, suffix: str = ".fits") -> Path:
    """The newest calibration file older than the image (the oldest one if none is)."""
    files = sorted(caldir.glob(f"{prefix}_????????_??????{suffix}"))
    if not files:
        raise PrepError(f"No {prefix} files found in {caldir}")
    observed = date_obs(header).datetime.replace(tzinfo=UTC)

    def made(path: Path) -> datetime:
        stamp = path.name[len(prefix) + 1 : len(prefix) + 16]
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)

    older = [path for path in files if observed > made(path)]
    if not older:
        print(f"warning: no {prefix} file older than the image, using {files[0].name}", file=sys.stderr)
    return older[-1] if older else files[0]


@cache
def read_map(path: Path) -> np.ndarray:
    """A calibration FITS image (cached: the same few files serve every image)."""
    data = np.asarray(fits.getdata(path))
    data.flags.writeable = False
    return data


@cache
def read_save(path: Path, name: str) -> np.ndarray:
    """An array from a calibration IDL save file (cached)."""
    data = np.asarray(readsav(str(path))[name])
    data.flags.writeable = False
    return data


# ---------------------------------------------------------------------------------------------------------------------
# The steps of P2SW_PREP


def p2sw_hdrchk(header: fits.Header, opts: Options) -> fits.Header:
    """Check that the image can be processed and complete its header: LARs and calibration images are refused, the
    pixel scales fixed and the detector temperature for the dark (TEMPDARK) worked out."""
    header = header.copy()
    exptime = float(header["EXPTIME"])

    lar_known = "DTPLAR1" in header and "DTPLAR2" in header
    if not opts.ignore_lar and lar_known and (header["DTPLAR1"] < 180.0 or header["DTPLAR2"] < 5.0 + exptime):
        raise PrepError("Image acquired within LAR, aborting prep")

    if not opts.force_calibrate:
        led = str(header.get("LED_POW", "off")).strip().upper().startswith("ON")
        if led or abs(header.get("PAV_ROT0", 0.0)) > 0.02 or abs(header.get("PAV_ROT1", 0.0)) > 0.02:
            raise PrepError("Image identified as LED or DARK acquisition, aborting prep")
        tracking = str(header.get("OBS_MODE", "")).strip() == "CME tracking"
        if "P2_X0" in header and not tracking and (abs(header["P2_X0"]) > 2.3 or abs(header["P2_Y0"]) > 2.3):
            raise PrepError("Spacecraft not sun-centered, inconsistent with obs. mode and commanded pointing")

    # Pixel scales: swapped, or from before their 2011 correction
    xscl, yscl = header["CDELT1"], header["CDELT2"]
    if xscl > yscl:
        header["CDELT1"], header["CDELT2"] = yscl, xscl
        for key, scale in (("CD1_1", yscl), ("CD2_1", yscl), ("CD1_2", xscl), ("CD2_2", xscl)):
            if header[key] != 0:
                header[key] = math.copysign(scale, header[key])
        update_history(header, "P2SW_HDRCHK", "Fixed incorrect CDELT and CDij values")
    old_new = {3.16226783968713: 3.1646941, 3.17123287671252: 3.1760467}
    changed = False
    for key in ("CDELT1", "CDELT2", "CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        value = float(header[key])
        if abs(value) in old_new:
            header[key] = math.copysign(old_new[abs(value)], value)
            changed = True
    if changed:
        update_history(header, "P2SW_HDRCHK", "Outdated CDELT and CDi_j keywords corrected")

    # Detector temperature: the sample(s) within 10 minutes of the image, interpolated in time
    valid = 3 * 86400.0 if opts.temp_dt_override else 600.0
    samples = [
        (header[temp], abs((Time(header[when], scale="utc") - date_obs(header)).sec))
        for temp, when in (("TEMP1DET", "TTEMP1"), ("TEMP2DET", "TTEMP2"))
        if temp in header and when in header
    ]
    recent = [(temp, dt) for temp, dt in samples if dt <= valid]
    if not samples:
        raise PrepError("Level-0 header determined as insufficiently complete (no detector temperature)")
    if not recent:
        raise PrepError("No sufficiently recent Detector Temp. in level-0 header")
    if len(recent) == 2:
        (t1, dt1), (t2, dt2) = recent
        det_temp = (t1 + t2) / 2.0 if dt1 + dt2 == 0 else (t2 - t1) / (dt2 + dt1) * dt1 + t1
    elif len(samples) == 2:
        det_temp = min(samples, key=lambda sample: sample[1])[0]
    else:
        det_temp = recent[0][0]
    after = "TTEMP2" if "TTEMP2" in header else "TTEMP1"
    header.set("TEMPDARK", float(det_temp), "[Celsius] temperature used in dark subtraction", after=after)
    update_history(header, "P2SW_HDRCHK", f"Computed detector temp.: {det_temp:.8g} Celsius")

    if header["FIRSTROW"] < 1 or header["LAST_ROW"] > SIZE or header["FIRSTCOL"] < 1 or header["LAST_COL"] > SIZE:
        raise PrepError("Invalid detector sub-field area, aborting prep")
    if capture_mode(header) not in ("ds", "cds"):
        raise PrepError("Image capture mode neither DS nor CDS, aborting prep")

    header.set("SWXCEN", -10000.0, "[pixel] axis 1 location of solar center in lv0", after="CROTA2")
    header.set("SWYCEN", -10000.0, "[pixel] axis 2 location of solar center in lv0", after="SWXCEN")
    return header


def p2sw_pmcdiv(header: fits.Header, image: np.ndarray, opts: Options) -> np.ndarray:
    """Undo the on-board multiplication by 2 or 4 of the pixels marked 2 or 3 in the pixel map."""
    if rebinned(header):
        return image
    path = calibration_file(opts.caldir, "swap_pmc", header)
    pmc = np.ones(image.shape, dtype=np.int16)
    rows, cols = window(header)
    pmc[rows, cols] = read_map(path)[rows, cols]
    image = image.copy()
    for mark, factor in ((2, 2.0), (3, 4.0)):
        values = image[pmc == mark] / factor
        image[pmc == mark] = np.trunc(values) if np.issubdtype(image.dtype, np.integer) else values
    update_history(header, "P2SW_PMCDIV", f"Reverted wrt {path.name}")
    return image


def p2sw_pixrep(header: fits.Header, image: np.ndarray, opts: Options, fits_path: Path | None = None) -> np.ndarray:
    """Put back the pixels preserved on board (from the FITS extension) and set saturated pixels to 4400."""
    image = image.copy()
    if not opts.no_pixlist and header.get("EXTEND", False) and fits_path is not None:
        with fits.open(fits_path) as hdus:
            if len(hdus) > 1 and hdus[1].data is not None:
                table = np.asarray(hdus[1].data)  # IDL [3, n]: x, y, value
                for x, y, value in table.reshape(-1, 3):
                    image[int(y), int(x)] = value
                update_history(header, "P2SW_PIXREP", f"Replaced {len(table):4d} pixels with preprocessed values")
    if not opts.no_satrep and not rebinned(header):
        path = calibration_file(opts.caldir, "swap_sat" + capture_mode(header), header)
        saturation = np.full(image.shape, 5500.0, dtype=F32)
        rows, cols = window(header)
        saturation[rows, cols] = read_map(path)[rows, cols]
        image[image >= F32(0.95) * saturation] = 4400
        update_history(header, "P2SW_PIXREP", f"Replaced wrt {path.name}")
    return image


def dark_model(coefs: np.ndarray, kelvin: np.float32) -> np.ndarray:
    """Dark current per pixel (DN/s): c0 + c1 T + c2 T^2. (+ c3 T^3.), negatives set to 0. The terms are ~1e5 and
    cancel to ~1, so the float32 rounding is IDL's: left to right, each power rounded once."""
    model = coefs[0] + coefs[1] * kelvin
    for power in range(2, len(coefs)):
        model = model + coefs[power] * F32(np.float64(kelvin) ** power)
    return np.maximum(model, F32(0))


def p2sw_drksub(header: fits.Header, image: np.ndarray, opts: Options) -> np.ndarray:
    """Subtract the bias (DS mode only) and the dark current, by default modelled per pixel as a polynomial in the
    detector temperature. Then clip the strongly negative pixels left by bad pixels."""
    mode = capture_mode(header)
    rows, cols = window(header)

    def fitted(frame: np.ndarray) -> np.ndarray:
        """The calibration frame on the image's read-out area, rebinned like the image."""
        full = np.zeros(image.shape, dtype=F32)
        part = frame[rows, cols].astype(F32)
        full[rows, cols] = rebin_2x2(part) if rebinned(header) else part
        return full

    result = image.astype(F32)
    if mode == "ds":
        path = calibration_file(opts.caldir, "swap_biasds", header)
        result = result - fitted(read_map(path))
        update_history(header, "P2SW_DRKSUB", f"Subtracted {path.name}")

    exptime, gain = F32(header["EXPTIME"]), F32(header.get("PGA_GAIN", 1))
    if opts.dark_form == "fits":
        path = calibration_file(opts.caldir, "swap_dark" + mode, header)
        dark = fitted(read_map(path)) * exptime * gain
        update_history(header, "P2SW_DRKSUB", f"Subtracted {path.name}")
    else:
        path = calibration_file(opts.caldir, "swap_dark_coefs", header, suffix=".save")
        coefs = read_save(path, "dark_coefs").astype(F32)  # [power, y, x]
        kelvin = F32(header["TEMPDARK"] + float(F32(273.15)))  # IDL's 273.15 is a float32 literal: it matters
        dark = fitted(dark_model(coefs, kelvin)) * exptime * gain
        update_history(header, "P2SW_DRKSUB", "Subtracted modeled dark with coefficients in")
        update_history(header, "P2SW_DRKSUB", f" {path.name} using")
        update_history(header, "P2SW_DRKSUB", f" T={float(kelvin):6.2f} K, dt={header['EXPTIME']:g} s")
    result = result - dark

    return np.maximum(result, dark_cutoff(result))


def dark_cutoff(image: np.ndarray) -> np.float32:
    """The clipping level after the dark subtraction: the lower edge of the 0.1 DN histogram bin holding the darkest
    1 % of the pixels, or 0 if that is positive."""
    low = float(image.min())
    bins = np.floor((image.astype(np.float64) - low) / 0.1).astype(np.int64).ravel()
    cumulative = np.cumsum(np.bincount(bins))
    first = int(np.argmax(cumulative / cumulative[-1] >= 0.01))
    return F32(min(0.0, low + first * 0.1))


def p2sw_pmcrep(header: fits.Header, image: np.ndarray, opts: Options) -> np.ndarray:
    """Replace the pixels the board didn't read (0 in the pixel map) by their 3 x 3 median."""
    path = calibration_file(opts.caldir, "swap_pmc", header)
    pmc_full = read_map(path)
    if rebinned(header):
        unread = np.repeat(np.repeat(pmc_full.reshape(SIZE // 2, 2, SIZE // 2, 2).min(axis=(1, 3)) == 0, 2, 0), 2, 1)
        pmc_full = np.where(unread, 0, 1)
    pmc = np.ones(image.shape, dtype=np.int16)
    rows, cols = window(header)
    pmc[rows, cols] = pmc_full[rows, cols]
    image = image.copy()
    unread = pmc == 0
    image[unread] = idl_median_filter(image, 3)[unread]
    update_history(header, "P2SW_PMCREP", f"Reverted wrt {path.name}")
    return image


def p2sw_ffcorr(header: fits.Header, image: np.ndarray, opts: Options) -> np.ndarray:
    """Divide by the flat field."""
    path = calibration_file(opts.caldir, "swap_flat" + capture_mode(header), header)
    flat = np.ones(image.shape, dtype=F32)
    rows, cols = window(header)
    part = read_map(path)[rows, cols].astype(F32)
    flat[rows, cols] = rebin_2x2(part) if rebinned(header) else part
    update_history(header, "P2SW_FFCORR", f"Flat-field: {path.name}")
    return image / flat


def gentle_filter(image: np.ndarray, second_max: bool = False) -> np.ndarray:
    """Clamp every pixel between the minimum and maximum of its 8 neighbours (or the second ones, with second_max)."""
    shifts = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # (dy, dx), IDL's order
    neighbours = np.stack([np.roll(image, shift, axis=(0, 1)) for shift in shifts]).astype(np.float64)
    if second_max:
        index = np.indices(image.shape)
        brightest, faintest = np.argmax(neighbours, axis=0), np.argmin(neighbours, axis=0)
        neighbours[brightest, index[0], index[1]] = np.nan
        neighbours[faintest, index[0], index[1]] = np.nan
        high, low = np.nanmax(neighbours, axis=0), np.nanmin(neighbours, axis=0)
    else:
        high, low = neighbours.max(axis=0), neighbours.min(axis=0)
    return np.minimum(np.maximum(image, low), high).astype(image.dtype)


def p2sw_despike(header: fits.Header, image: np.ndarray, opts: Options) -> np.ndarray:
    """Replace the pixels more than spike_sig robust sigmas (1.483 x the median absolute deviation, both over 5 x 5)
    from their 5 x 5 median by that median: cosmic ray hits, and dark pixels too."""
    rows, cols = window(header)
    image = image.copy()
    sub = image[rows, cols]
    median = ndimage.median_filter(sub, size=5, mode="mirror")  # FILTER_IMAGE(MEDIAN=5, /ALL_PIXELS)
    deviation = np.abs(sub - median)
    shifted = np.stack(
        [np.abs(np.roll(sub, (dy, dx), axis=(0, 1)) - median) for dy in range(-2, 3) for dx in range(-2, 3)]
    )
    mad = np.partition(shifted, 12, axis=0)[12]
    spikes = deviation > F32(opts.spike_sig) * F32(1.483) * mad
    sub[spikes] = median[spikes]
    image[rows, cols] = sub
    update_history(header, "P2SW_DESPIKE", f"Despiked at the {opts.spike_sig:3.1f}-sigma level ({spikes.sum()} pix)")
    if opts.apply_gentle_filter:
        image = gentle_filter(image, second_max=True)
        update_history(header, "P2SW_DESPIKE", "Gentle filter applied")
    return image


def p2sw_psfcor(header: fits.Header, image: np.ndarray, opts: Options, psf: np.ndarray | None = None) -> np.ndarray:
    """Remove stray light by deconvolving with the PSF measured from the 2011 eclipse; optionally subtract the
    horizontal bands that this brings out in dark regions."""
    if psf is None:
        paths = sorted(opts.caldir.glob("swap_psf_????????_??????.save"))
        if not paths:
            raise PrepError(f"No PSF file found in {opts.caldir}")
        psf, used = read_save(paths[-1], "psf"), paths[-1].name
    else:
        used = "a user-provided PSF"
    if psf.shape != (SIZE, SIZE) or image.shape != (SIZE, SIZE):
        raise PrepError("PSF deconvolution needs a 1024 x 1024 PSF and image")

    def padded(array: np.ndarray) -> np.ndarray:
        out = np.zeros((2 * SIZE, 2 * SIZE))
        out[SIZE:, :SIZE] = array
        return out

    psf_pad = np.roll(padded(psf), (SIZE // 2, -SIZE // 2), axis=(0, 1))
    img_pad = padded(image)
    # IDL: FFT(FFT(img, 1) / FFT(psf, 1), -1); FFT(x, 1) = N ifft2(x) and FFT(x, -1) = fft2(x) / N
    result = np.fft.fft2(np.fft.ifft2(img_pad) / np.fft.ifft2(psf_pad)).real / img_pad.size
    result = result[SIZE:, :SIZE]
    update_history(header, "P2SW_PSFCOR", f"Deconvolved PSF using {used}.")

    if opts.remove_horizontal:
        threshold = 0.7 * float(header["EXPTIME"])
        mask = np.zeros((SIZE, SIZE))
        if np.count_nonzero(result > threshold) > 1:
            premask = ndimage.uniform_filter((result >= threshold).astype(np.float64), size=31, mode="nearest")
            mask[premask > 0.25] = 1
            masked = np.where(mask == 1, np.nan, result)
        else:
            masked = image.astype(np.float64)
        rows_median = idl_nanmedian(masked, axis=1)
        overall = idl_nanmedian(masked.ravel(), axis=0)
        cross_section = (rows_median - overall) * ((1 - mask).sum(axis=1) > 75)
        cross_section[~np.isfinite(cross_section)] = 0.0
        result = result - cross_section[:, None]
        update_history(header, "P2SW_PSFCOR", "Horizontal banding cleaned.")
    return result.astype(F32)


# --- Limb fit (p2sw_fit_limb and its helpers)


def p2sw_polar_map(
    image: np.ndarray, theta_size: int, rho_size: int, x_center: float, y_center: float, rho_min: float, rho_max: float
) -> np.ndarray:
    """The image on a polar grid [theta, rho] around (x_center, y_center); theta from +y towards +x."""
    rho = rho_min + (rho_max - rho_min) * np.arange(rho_size, dtype=F32) / rho_size
    theta = (np.arange(theta_size, dtype=F32) * F32(360.0) / F32(theta_size)) * F32(np.pi) / F32(180.0)
    # INTERPOLATE works in single precision (no /DOUBLE): the coordinates are float32
    x = (np.outer(np.sin(theta), rho) + x_center).astype(F32).astype(np.float64)
    y = (np.outer(np.cos(theta), rho) + y_center).astype(F32).astype(np.float64)
    return interpolate_cubic(image, x, y, missing=0.0)


def p2sw_limb_profile(
    image: np.ndarray, param: np.ndarray, theta_size: int, rho_size: int, band: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every angle, the radius where the brightness rises fastest outwards (type 'max'), and whether it is
    usable (the whole radial cut inside the image, the maximum not at the ends of the band). No sector is masked
    (/HOLES)."""
    rho_min, rho_max = param[2] * band[0], param[2] * band[1]
    angles = F32(2.0 * np.pi) * np.arange(theta_size, dtype=np.float64) / theta_size
    polar = p2sw_polar_map(image, theta_size, rho_size, param[0], param[1], rho_min, rho_max)
    slope = (polar[:, 2:] - polar[:, :-2]) / 2.0  # DERIV along rho, at rho = 1 .. rho_size - 2
    polar = polar[:, 1:-1]
    steepest = np.argmax(slope, axis=1)
    inside = np.all(polar != 0, axis=1)
    usable = inside & (steepest < rho_size - 3) & (steepest > 0)
    radius = rho_min + (steepest + 1.5) * (rho_max - rho_min) / rho_size
    return angles, radius, usable


def p2sw_polar_circle_limb(x: np.ndarray, a0: float, a1: float, radius: float) -> np.ndarray:
    """Distance from the polar origin to a circle of the given radius centred at (a0, a1) from it, along angle x."""
    return a0 * np.sin(x) + a1 * np.cos(x) + np.sqrt(radius**2 - (a0 * np.cos(x) - a1 * np.sin(x)) ** 2)


def _circle_jacobian(x: np.ndarray, a0: float, a1: float, radius: float) -> np.ndarray:
    cross = a0 * np.cos(x) - a1 * np.sin(x)
    root = np.sqrt(radius**2 - cross**2)
    return np.column_stack([np.sin(x) - np.cos(x) * cross / root, np.cos(x) + np.sin(x) * cross / root, radius / root])


def idl_curvefit(x: np.ndarray, y: np.ndarray, a: np.ndarray, tol: float = 1e-3, itmax: int = 20) -> np.ndarray:
    """IDL's CURVEFIT (Marquardt's gradient-expansion least squares, unit weights) of p2sw_polar_circle_limb. It stops
    once chi-square improves by less than tol (relative), short of full convergence; the sun centers of the archive
    follow it to the 0.01 pixel they are rounded to."""
    a = np.array(a, dtype=np.float64)
    nfree = len(y) - len(a)
    flambda = 1e-3
    for _ in range(itmax):
        yfit, pder = p2sw_polar_circle_limb(x, *a), _circle_jacobian(x, *a)
        beta = (y - yfit) @ pder
        alpha = pder.T @ pder
        chisq1 = float(np.sum((y - yfit) ** 2)) / nfree
        if chisq1 < np.sum(np.abs(y)) / 1e7 / nfree:
            break
        scale = np.sqrt(np.outer(np.diag(alpha), np.diag(alpha)))
        tries = 0
        while True:
            tries += 1
            normalized = alpha / scale
            normalized[np.diag_indices_from(normalized)] *= 1.0 + flambda
            b = a + (np.linalg.inv(normalized) / scale) @ beta
            chisq = float(np.sum((y - p2sw_polar_circle_limb(x, *b)) ** 2)) / nfree
            if not math.isfinite(chisq) or (tries > 30 and chisq >= chisq1):
                return a
            flambda *= 10.0
            if chisq <= chisq1:
                break
        flambda /= 100.0
        a = b
        if (chisq1 - chisq) / chisq1 <= tol:
            break
    return a


def p2sw_fit_limb(
    image: np.ndarray,
    param: np.ndarray,
    theta_size: int = 10000,
    rho_size: int = 40,
    band: tuple[float, float] = (0.99, 1.01),
    tol: float = 1.5e-3,
    maxiter: int = 30,
    nbcol: int = 100,
) -> bool:
    """Fit the limb with a circle, updating param = [x_center, y_center, radius] in place. Each pass takes the limb
    profile around the current center (the first in a wider band) and fits r(theta) of an offset circle to it, until
    the correction is below tol. False if too few angles were usable."""
    outer_band = (0.97, 1.03)
    correction = np.array([0.0, 0.0, param[2]])
    for iteration in range(maxiter):
        angles, radius, usable = p2sw_limb_profile(
            image, param, theta_size, rho_size, outer_band if iteration == 0 else band
        )
        count = int(usable.sum())
        if count <= nbcol:
            return False
        x, profile = angles[usable], radius[usable]
        correction = idl_curvefit(x, profile, correction)
        change = math.sqrt(correction[0] ** 2 + correction[1] ** 2 + (param[2] - correction[2]) ** 2)
        param[0:2] += correction[0:2]
        param[2] = correction[2]
        if change < tol:
            break
    return True


def find_sun_center(header: fits.Header, image: np.ndarray) -> tuple[float, float]:
    """The sun center the way P2SW_IMGCOR finds it: a ring of the solar radius (+-3 pixels) correlated with the 5 x 5
    median of the image for a start, then P2SW_FIT_LIMB, rounded to 0.01 pixel."""
    radius = header["RSUN_ARC"] / header["CDELT1"]
    yy, xx = np.indices((SIZE, SIZE))
    distance = np.hypot(xx - SIZE // 2, yy - SIZE // 2).astype(F32)
    ring = ((distance < radius + 3) & (distance > radius - 3)).astype(np.float64)
    smoothed = idl_median_filter(image, 5).astype(np.float64)
    correlation = np.abs(
        np.roll(np.fft.ifft2(np.fft.fft2(ring) * np.fft.fft2(smoothed)), (SIZE // 2, SIZE // 2), (0, 1))
    )
    row, col = np.unravel_index(np.argmax(correlation), correlation.shape)
    param = np.array([float(col), float(row), radius])
    p2sw_fit_limb(image.astype(np.float64), param)
    x, y = (F32(idl_round(value * 100)) / F32(100.0) for value in param[:2])
    return float(x), float(y)


def p2sw_imgcor(header: fits.Header, image: np.ndarray, opts: Options) -> np.ndarray:
    """Center the sun, make the pixels square and turn solar north up, in one cubic interpolation. The matrix is
    uncenter . solar_ep . translate . (mount offset . s/c roll) . pixel scale . center, on (x, y, 1)."""
    no_center_offpoint = False
    if date_obs(header).mjd >= 55256 and (header.get("PAV_ROT0", 0.0) != 0 or header.get("PAV_ROT1", 0.0) != 0):
        no_center_offpoint = not opts.force_sun_center

    cx, cy = (image.shape[1] - 1.0) / 2.0, (image.shape[0] - 1.0) / 2.0  # IDL: NAXIS1/2
    identity = np.eye(3)
    center = np.array([[1.0, 0, -cx], [0, 1.0, -cy], [0, 0, 1.0]])
    uncenter = np.array([[1.0, 0, cx], [0, 1.0, cy], [0, 0, 1.0]])
    xscl, yscl = float(header["CDELT1"]), float(header["CDELT2"])

    def rotation(degrees: float) -> np.ndarray:
        c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    if opts.no_rotate:
        roll = identity
    else:
        roll = np.array(
            [
                [header["CD1_1"] / xscl, header["CD1_2"] / yscl, 0],
                [header["CD2_1"] / xscl, header["CD2_2"] / yscl, 0],
                [0, 0, 1.0],
            ]
        )
    mount = identity if (opts.no_swap_mount_offset or opts.no_rotate) else rotation(-0.3)
    roll = mount @ roll
    scale = identity if opts.no_rescale else np.array([[1.0, 0, 0], [0, yscl / xscl, 0], [0, 0, 1.0]])
    solar_ep = float(header["SOLAR_EP"])
    ecliptic = identity if (opts.no_rotate or opts.no_solar_ep) else rotation(solar_ep)

    if opts.sun_cen_x is None or opts.sun_cen_y is None:
        found_x, found_y = find_sun_center(header, image)
    else:  # IDL fits the limb anyway, for SWXCEN/SWYCEN; here they record the center given
        found_x, found_y = opts.sun_cen_x, opts.sun_cen_y
    sun_x = found_x if opts.sun_cen_x is None else opts.sun_cen_x
    sun_y = found_y if opts.sun_cen_y is None else opts.sun_cen_y
    header["SWXCEN"] = round(found_x + 1.0, 2)  # float32 in IDL; written with 2 decimals as the archive shows
    header["SWYCEN"] = round(found_y + 1.0, 2)

    moved = uncenter @ roll @ scale @ center @ np.array([sun_x, sun_y, 1.0])
    translate = identity.copy()
    if not (opts.no_sun_center or no_center_offpoint):
        translate[0:2, 2] = np.array([cx, cy]) - moved[0:2]
    matrix = uncenter @ ecliptic @ translate @ roll @ scale @ center

    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        result = image
    else:
        x_in, y_in = poly_2d_coordinates(inverse, image.shape)
        result = interpolate_cubic(image.astype(np.float64), x_in, y_in, 0.0, poly_2d=True).astype(image.dtype)
        if not opts.no_rescale and opts.preserve_radiometric:
            result = result * F32(yscl / xscl)

    if opts.no_rotate:
        if not opts.no_rescale:
            header["CD1_2"] *= xscl / yscl
            header["CD2_2"] *= xscl / yscl
    elif opts.no_solar_ep:
        c, s = math.cos(math.radians(solar_ep)), math.sin(math.radians(solar_ep))
        y_scale = xscl if not opts.no_rescale else yscl
        header["CD1_1"], header["CD1_2"] = xscl * c, -y_scale * s
        header["CD2_1"], header["CD2_2"] = xscl * s, y_scale * c
        header["CROTA1"] = header["CROTA2"] = math.radians(solar_ep)  # sic: IDL writes the angle in radians here
    else:
        header["CD1_1"], header["CD1_2"], header["CD2_1"], header["CD2_2"] = xscl, 0.0, 0.0, xscl
        header["CROTA1"] = header["CROTA2"] = 0.0
    if not opts.no_rescale:
        header["CDELT2"] = xscl
    if opts.no_sun_center or no_center_offpoint:
        transformed = matrix @ np.array([sun_x, sun_y, 1.0])
        header["CRPIX1"], header["CRPIX2"] = transformed[0] + 1, transformed[1] + 1
    else:
        update_history(header, "P2SW_IMGCOR", "Image centred on solar disk centre")
    if not opts.no_rescale:
        update_history(
            header,
            "P2SW_IMGCOR",
            "Image scaled to square pixels" + (" total intensity preserved" * opts.preserve_radiometric),
        )
    if not opts.no_rotate:
        text = (
            "Spacecraft roll corrected, Ecliptic North up." if opts.no_solar_ep else "Image rotated to Solar North up"
        )
        update_history(header, "P2SW_IMGCOR", text)
    return result


def p2sw_expnorm(header: fits.Header, image: np.ndarray) -> np.ndarray:
    """Divide by the exposure time: DN/s."""
    header["BUNIT"] = "DN/s"
    update_history(header, "P2SW_EXPNORM", "Exposure time normalized data (DN/s)")
    return image / F32(header["EXPTIME"])


# ---------------------------------------------------------------------------------------------------------------------
# The driver


def prep(image: np.ndarray, header: fits.Header, opts: Options | None = None, fits_path: Path | None = None):
    """Level 0 image and header -> level 1 image (float32, or int16 without opts.keep_float) and header. Steps already
    in the file's HISTORY are skipped, as in IDL. Raises PrepError for an image that isn't made into level 1."""
    opts = replace(opts or Options())
    if opts.remove_horizontal:
        opts.psf_correct = True  # banding can only be removed from deconvolved images
    header = header.copy()
    if not previously(header, "P2SW_HDRCHK"):
        header = p2sw_hdrchk(header, opts)

    steps = []
    if not opts.no_calibration:
        steps += [
            ("P2SW_PMCDIV", not opts.no_pmcdiv, lambda img: p2sw_pmcdiv(header, img, opts)),
            ("P2SW_PIXREP", True, lambda img: p2sw_pixrep(header, img, opts, fits_path)),
            ("P2SW_DRKSUB", not opts.no_dark, lambda img: p2sw_drksub(header, img, opts)),
            ("P2SW_PMCREP", not opts.no_pmcrep, lambda img: p2sw_pmcrep(header, img, opts)),
            ("P2SW_FFCORR", not opts.no_flat, lambda img: p2sw_ffcorr(header, img, opts)),
        ]
    steps.append(("P2SW_DESPIKE", opts.despike, lambda img: p2sw_despike(header, img, opts)))
    for routine, wanted, step in steps:
        if wanted and not previously(header, routine):
            image = step(image)

    image = image.astype(F32)
    if "DATAMAX" in header:  # before the PSF and the transformations, in DN/s
        header.set(
            "SWAVINT",
            float(image.mean(dtype=np.float64)) / float(header["EXPTIME"]),
            "[DN/s/pixel] avg intensity in calibrated image",
            after="DATAMAX",
        )

    transformed_before = previously(header, "P2SW_IMGCOR")
    if (opts.psf_correct or opts.remove_horizontal) and not previously(header, "P2SW_PSFCOR"):
        if transformed_before:
            print("warning: PSF deconvolution can't be applied to an image already transformed", file=sys.stderr)
        else:
            image = p2sw_psfcor(header, image, opts)
    if not transformed_before:
        image = p2sw_imgcor(header, image, opts)
    if opts.normalize and not previously(header, "P2SW_EXPNORM"):
        image = p2sw_expnorm(header, image)

    if not opts.keep_float:
        image = idl_round(image).astype(np.int16)
    if not opts.permit_negatives:
        image = np.maximum(image, 0)

    header["LEVEL"] = 1
    header["CREATOR"] = f"P2SW_PREP.PRO {VERSION}"
    header["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    header["FILENAME"] = level1_name(header)
    header["WCSNAME"] = (
        "Helioprojective-cartesian",
        f"aligned with {'ecliptic' if opts.no_rotate else 'solar'} North",
    )
    header["DATAMAX"], header["DATAMIN"] = float(image.max()), float(image.min())
    return image, header


def level1_name(header: fits.Header) -> str:
    stamp = date_obs(header).datetime.strftime("%Y%m%d_%H%M%S")
    return f"swap_lv1_{stamp}.fits"


def to_fits_integers(image: np.ndarray, header: fits.Header, normalize: bool = True, permit_negatives: bool = False):
    """The int16 array, BSCALE and BZERO that P2SW_PREP writes: 1/16 DN (or DN/s) steps, 0 at -32768."""
    exptime = F32(header["EXPTIME"])
    bscale = F32(1.0) / (F32(16.0) * (exptime if normalize else F32(1.0)))
    if not permit_negatives or image.min() >= 0:
        bzero = F32(2.0**15) * bscale
    else:
        span = float(image.max()) - float(image.min())
        if 2.0**16 - span / bscale < 0:
            bscale = F32(1.0 / ((2.0**16 - 1.0) / span))
        bzero = F32(2.0**15) * bscale - F32(abs(image.min()))
    values = idl_round((image.astype(F32) - bzero) / bscale)
    return np.clip(values, -(2**15), 2**15 - 1).astype(np.int16), float(bscale), float(bzero)


def write_level1(path: Path, image: np.ndarray, header: fits.Header, opts: Options | None = None) -> None:
    """Write the level 1 FITS file as the archive does: int16 with BSCALE and BZERO (or float32 if the image is)."""
    opts = opts or Options()
    header = header.copy()
    for key in ("BSCALE", "BZERO", "CHECKSUM", "DATASUM", "BITPIX", "EXTEND"):
        header.remove(key, ignore_missing=True, remove_all=True)
    if np.issubdtype(image.dtype, np.floating):
        data, bscale, bzero = to_fits_integers(image, header, opts.normalize, opts.permit_negatives)
        header["DATAMAX"] = float(data.max() * F32(bscale) + F32(bzero))
        header["DATAMIN"] = float(data.min() * F32(bscale) + F32(bzero))
    else:
        data, bscale, bzero = image.astype(np.int16), 1.0, 0.0
    hdu = fits.PrimaryHDU(data=data, header=header)
    # IDL writes these float32 values with 6 significant digits (0.00625, 204.8), which is what readers apply
    hdu.header.set("BSCALE", float(f"{bscale:.6g}"), "ratio of physical to array value at 0 offset", after="EXPTIME")
    hdu.header.set("BZERO", float(f"{bzero:.6g}"), "physical value for the array value 0", after="BSCALE")
    hdu.writeto(path, overwrite=True, checksum=True)


def read_level0(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdus:
        return np.asarray(hdus[0].data), hdus[0].header.copy()


# ---------------------------------------------------------------------------------------------------------------------
# Display: p2sw_write_png


def display_bytes(image: np.ndarray, low: float = 0.5, high: float = 4.2) -> np.ndarray:
    """The SWAP team's PNG scaling (p2sw_write_png): BYTSCL(image ^ 0.25, 0.5, 4.2) of a level 1 image in DN/s."""
    scaled = np.power(np.maximum(image.astype(np.float64), 0.0), 0.25)
    return np.clip(np.floor(255.9999 * (scaled - low) / (high - low)), 0, 255).astype(np.uint8)


def quicklook_bytes(image: np.ndarray) -> np.ndarray:
    """The scaling of the P2SC's quicklook PNGs and daily movies (proba2.sidc.be/swap/data/qlviewer, .../mpg/movies),
    measured on 4 of them against their level 1 files: grey = 13.42 sqrt(DN/s) + 9.7, the same for every image to
    0.5 grey level rms. Darker than p2sw_write_png's 1/4 power, above all off the disk (1 DN/s: 23 against 35)."""
    grey = 13.42 * np.sqrt(np.maximum(image.astype(np.float64), 0.0)) + 9.7
    return np.clip(np.round(grey), 0, 255).astype(np.uint8)


SCALINGS = {"quicklook": quicklook_bytes, "p2sw": display_bytes}


def main() -> int:
    parser = argparse.ArgumentParser(description="SWAP level 0 FITS -> level 1, a port of p2sw_prep.pro")
    parser.add_argument("files", nargs="+", type=Path, help="level 0 FITS files")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("."), help="output folder")
    parser.add_argument("--caldir", type=Path, default=CALDB, help="calibration files (default: swap_caldb)")
    parser.add_argument("--no-despike", action="store_true")
    parser.add_argument("--psf", action="store_true", help="deconvolve the PSF (stray light)")
    parser.add_argument(
        "--remove-horizontal", action="store_true", help="also remove horizontal bands (implies --psf)"
    )
    parser.add_argument("--gentle-filter", action="store_true")
    parser.add_argument("--png", action="store_true", help="also write PNG previews with the SWAP team's scaling")
    args = parser.parse_args()

    opts = Options(
        caldir=args.caldir,
        despike=not args.no_despike,
        psf_correct=args.psf,
        remove_horizontal=args.remove_horizontal,
        apply_gentle_filter=args.gentle_filter,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path in args.files:
        image, header = read_level0(path)
        try:
            level1, header1 = prep(image, header, opts, fits_path=path)
        except PrepError as error:
            print(f"{path.name}: {error}", file=sys.stderr)
            continue
        out = args.outdir / header1["FILENAME"]
        write_level1(out, level1, header1, opts)
        if args.png:
            from PIL import Image as PILImage

            PILImage.fromarray(np.flipud(display_bytes(level1))).save(out.with_suffix(".png"))
        print(f"{out} (sun at {header1['SWXCEN']:.2f}, {header1['SWYCEN']:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

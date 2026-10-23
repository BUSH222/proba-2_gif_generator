"""Extract the SWAP images (APID 20) from raw PROBA-2 .cadu dumps and save them as JPEG files.

The packet layout is in the README ("APID 20") and `APID_20.ipynb`. In short: every image packet has an 8 byte header
(`01`, end of acquisition as a 32 bit tick and a 16 bit fraction, `00`) and ends in a CRC-16; the first packet of an
image has the 84 byte image header after that and then the JPEG, the others JPEG data only. An image whose packets are
not all there is cut at the first missing or garbled packet: the JPEG is progressive, so the part that came through is
still the whole image, only at lower quality.

Files are named after the UTC of the start of acquisition (the DATE-OBS of the archive's level 0 files), from the tick
and the GPS receiver records (APID 18) of the same dump.

With --process every image is also turned north up (90 degree steps, from the commanded roll in its header), its sun
center is found by fitting the limb, and it is shifted to put the sun in the middle. These go to `<out>/processed/` as
PNG, with the centers in `centers.csv`.

With --level1 every image also goes through the archive's calibration pipeline (`p2sw_prep.py`, a port of SolarSoft's
p2sw_prep.pro): dark current, flat field, bad and unread pixels, despiking, then sun centered, square pixels and solar
north up in one interpolation, in DN/s. These go to `<out>/level1/` as level 1 FITS files, with PNG previews in the
SWAP team's scaling.

With --gif all the images are also stitched, in time order, into `<out>/swap.gif`: the level 1 ones with --level1, the
processed ones with --process, the JPEGs as they came down otherwise. --mp4 does the same into `<out>/swap.mp4`.
"""

import argparse
import binascii
import bisect
import csv
import io
import math
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image as PILImage
from PIL import ImageFile

from cadu import HEADER_LEN, iter_packets, packet_apid, packet_sequence_count

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ty: ignore[invalid-assignment] - a cut JPEG still decodes, as far as it goes

SWAP_APID = 20
GPS_APID = 18
SEQUENCE_MODULO = 1 << 14

# Image packets. Offsets are into the payload (the packet minus its 6 byte primary header), all big-endian.
IMAGE_MARKER = 0x01  # first payload byte of an image packet (the event and verification reports have 0x10)
IMAGE_HEADER_LEN = 8  # 01, tick (4), fraction (2), 00
METADATA_END = 92  # the image header of the first packet ends here, and the JPEG starts
STREAM_START = 88  # SIZCOMPI counts from here: JPEG length (4), the JPEG, padding to a multiple of 16
CRC_LEN = 2
PACKET_LEN = 65536  # all packets of an image but the last
DATA_PER_PACKET = PACKET_LEN - HEADER_LEN - IMAGE_HEADER_LEN - CRC_LEN
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"

# Time: the stamps count 1/65536 of a tick. One tick is about a second of GPS time, but the on-board clock gains
# 1.064 s a day on GPS, so the offset is taken from the GPS records of each dump (see the README, "Time").
FRACTION = 65536
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
GPS_MINUS_UTC = 18  # leap seconds
GPS_MINUS_TICK = 1_209_613_891  # fallback for a dump without GPS records: right in mid-August 2025 only
TICK_DRIFT = 1.064 / 86400  # GPS minus tick grows by this per tick
GPS_HEADER = bytes.fromhex("100889")

# Image processing
PIXEL_SCALE = 3.1646941  # arcsec per pixel along x (CDELT1 of the archive)
LIMB_ABOVE_PHOTOSPHERE = 2.0  # pixels: the brightest point of the limb at 17.4 nm, measured against the archive
RING_HALF_WIDTH = 3  # pixels, as in the pipeline's p2sw_imgcor
LIMB_SEARCH = (0.97, 1.03)  # radii searched for the limb, relative to the expected one
LIMB_ANGLES = 720
MAX_RADIUS_ERROR = 4.0  # pixels: a fit further off the expected limb radius is not trusted
MIN_LIMB_POINTS = 0.5  # fraction of the radii that must give a limb point consistent with the circle

GIF_NAME = "swap.gif"
MP4_NAME = "swap.mp4"
GIF_FRAME_MS = 100  # also the frame time of the mp4

# Grey level -> detector value (DN) of the archive's level 0 images, measured in APID_20.ipynb (section 9). The board
# maps 10-3600 DN to 8 bits before the JPEG compression ("fixed recoding"); this is the ground's inverse.
# fmt: off
RECODE_TABLE = np.array([
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 32, 35, 38, 41, 44, 47, 50, 53,
    56, 59, 62, 65, 68, 71, 74, 77, 80, 84, 89, 94, 99, 104, 109, 114, 119, 124, 129, 134, 139, 144, 149, 154,
    159, 164, 169, 174, 179, 185, 192, 199, 206, 213, 220, 227, 234, 241, 248, 255, 262, 269, 276, 283, 291, 300,
    309, 318, 327, 336, 345, 354, 363, 372, 381, 390, 399, 408, 417, 426, 435, 444, 453, 462, 471, 480, 490, 501,
    512, 523, 534, 545, 556, 567, 578, 589, 600, 611, 622, 633, 644, 656, 669, 682, 695, 708, 721, 734, 747, 760,
    773, 786, 799, 812, 825, 838, 851, 864, 877, 890, 903, 916, 930, 945, 960, 975, 990, 1005, 1020, 1035, 1050,
    1065, 1080, 1095, 1110, 1125, 1140, 1155, 1170, 1186, 1203, 1220, 1237, 1254, 1271, 1288, 1305, 1322, 1339,
    1356, 1373, 1390, 1407, 1424, 1441, 1458, 1475, 1493, 1512, 1531, 1550, 1569, 1588, 1607, 1626, 1645, 1664,
    1683, 1702, 1721, 1740, 1759, 1778, 1797, 1816, 1836, 1857, 1878, 1899, 1920, 1941, 1962, 1983, 2004, 2025,
    2046, 2067, 2088, 2109, 2130, 2151, 2172, 2193, 2214, 2235, 2257, 2280, 2303, 2326, 2349, 2372, 2395, 2418,
    2441, 2464, 2487, 2510, 2533, 2556, 2579, 2603, 2628, 2653, 2678, 2703, 2728, 2753, 2778, 2803, 2828, 2853,
    2878, 2903, 2928, 2953, 2978, 3003, 3028, 3053, 3078, 3103, 3128, 3154, 3181, 3208, 3235, 3262, 3289, 3316,
    3343, 3370, 3397, 3424, 3451, 3478, 3505, 3532, 3560, 3588,
], dtype=np.float64)
# fmt: on


def crc_ok(packet: bytes) -> bool:
    """CRC-16/CCITT-FALSE over the packet, in its last 2 bytes."""
    return binascii.crc_hqx(packet[:-CRC_LEN], 0xFFFF) == int.from_bytes(packet[-CRC_LEN:])


@dataclass
class ImageHeader:
    """The image header of the first packet (payload bytes 0-91), with the names of the level 0 FITS keywords."""

    end: int  # 1-6, EACQTIME / TRANTIME: end of acquisition, tick x 65536 + fraction
    start: int  # 8-13, SACQTIME: start of acquisition
    obs_mode: int  # 22, OBS_MODE: 0 sun centered, 1 fixed off-pointing, 2 variable off-pointing, 3 CME tracking
    lang_rot: float  # 23-30, LANG_ROT: commanded large angle rotation component, sin(roll / 2)
    pav_rot: tuple[float, float]  # 31-46, PAV_ROT0/1: commanded paving (off-pointing) rotation components
    capture_mode: int  # 47, CAP_MODE: 0 DS, 1 CDS
    exposure: int  # 48-51, EXPTIME (s)
    window: tuple[int, int, int, int]  # 52-59: first row, last row, first column, last column (0-based)
    priority: int  # 75, PN
    compressed_size: int  # 76-79, SIZCOMPI: bytes from payload offset 88 on (JPEG length, JPEG, padding)
    jpeg_length: int  # 88-91: SOI to EOI

    @classmethod
    def parse(cls, payload: bytes) -> ImageHeader:
        def u(offset: int, size: int) -> int:
            return int.from_bytes(payload[offset : offset + size])

        def f64(offset: int) -> float:
            return struct.unpack(">d", payload[offset : offset + 8])[0]

        return cls(
            end=u(1, 6),
            start=u(8, 6),
            obs_mode=payload[22],
            lang_rot=f64(23),
            pav_rot=(f64(31), f64(39)),
            capture_mode=payload[47],
            exposure=u(48, 4),
            window=(u(52, 2), u(54, 2), u(56, 2), u(58, 2)),
            priority=payload[75],
            compressed_size=u(76, 4),
            jpeg_length=u(88, 4),
        )

    @property
    def packet_count(self) -> int:
        return math.ceil((STREAM_START - IMAGE_HEADER_LEN + self.compressed_size) / DATA_PER_PACKET)

    def packet_length(self, index: int) -> int:
        """Length of the packet with this index in the image."""
        if index < self.packet_count - 1:
            return PACKET_LEN
        last_data = STREAM_START - IMAGE_HEADER_LEN + self.compressed_size - DATA_PER_PACKET * (self.packet_count - 1)
        return HEADER_LEN + IMAGE_HEADER_LEN + last_data + CRC_LEN

    @property
    def roll(self) -> int | None:
        """Commanded spacecraft roll in degrees (0, 90, 180 or 270), None if the value is damaged."""
        if not -1 <= self.lang_rot <= 1:
            return None
        roll = math.degrees(2 * math.asin(self.lang_rot))
        if abs(roll - 90 * round(roll / 90)) > 0.01:
            return None
        return round(roll) % 360

    @property
    def north_up_steps(self) -> int | None:
        """k for np.rot90(jpeg, k) that turns the image north up (CROTA1 / 90 of the archive)."""
        return None if self.roll is None else (-(self.roll + 90) // 90) % 4


@dataclass
class Image:
    dump: str
    packets: dict[int, bytes] = field(default_factory=dict)  # sequence count -> packet

    def add(self, packet: bytes) -> None:
        count = packet_sequence_count(packet)
        if count not in self.packets or (crc_ok(packet) and not crc_ok(self.packets[count])):
            self.packets[count] = packet  # a copy that passes the CRC beats one that doesn't

    def first_packet(self) -> bytes | None:
        firsts = [p for p in self.packets.values() if p[HEADER_LEN + METADATA_END :].startswith(JPEG_SOI)]
        return max(firsts, key=crc_ok) if firsts else None


@dataclass
class Assembled:
    header: ImageHeader
    header_ok: bool  # the first packet passed its CRC
    jpeg: bytes
    packets: int  # used
    packets_expected: int
    bad_crc: int  # packets used despite a failed CRC (bit errors)

    @property
    def complete(self) -> bool:
        return self.packets == self.packets_expected


def assemble(image: Image, keep_bit_errors: bool = True) -> Assembled | None:
    """The JPEG of an image, cut at the first missing or garbled packet (or, without keep_bit_errors, at any packet
    that fails its CRC). None without a first packet."""
    first = image.first_packet()
    if first is None:
        return None
    header = ImageHeader.parse(first[HEADER_LEN:])
    first_count = packet_sequence_count(first)
    chunks, bad = [], 0
    for index in range(header.packet_count):
        packet = image.packets.get((first_count + index) % SEQUENCE_MODULO)
        if packet is None or len(packet) != header.packet_length(index):
            break  # missing, or glued together from frames of other packets (lost frames)
        if not crc_ok(packet):
            if not keep_bit_errors:
                break
            bad += 1
        payload = packet[HEADER_LEN:-CRC_LEN]
        chunks.append(payload[METADATA_END if index == 0 else IMAGE_HEADER_LEN :])
    stream = b"".join(chunks)
    if len(chunks) == header.packet_count and stream[: header.jpeg_length].endswith(JPEG_EOI):
        jpeg = stream[: header.jpeg_length]  # leave out the padding
    else:
        jpeg = stream + JPEG_EOI  # makes the decoder finish the image with what it has
    return Assembled(header, crc_ok(first), jpeg, len(chunks), header.packet_count, bad)


def decode(jpeg: bytes) -> np.ndarray | None:
    """Grey levels of a JPEG, None if it doesn't decode to an image (damaged data can leave it all black)."""
    try:
        pixels = np.asarray(PILImage.open(io.BytesIO(jpeg)).convert("L"))
    except OSError, SyntaxError, ValueError:
        return None
    return pixels if pixels.std() > 1 else None


@dataclass
class Clock:
    """Tick -> UTC for one dump, from the GPS receiver's F4 records (APID 18): week and second against the tick."""

    reference_tick: float = 0.0
    gps_minus_tick: float = GPS_MINUS_TICK
    measured: bool = False

    @classmethod
    def from_records(cls, records: list[tuple[float, float]]) -> Clock:
        if not records:
            return cls()
        ticks, offsets = np.array(records).T
        return cls(float(np.median(ticks)), float(np.median(offsets)), measured=True)

    def utc(self, stamp: int) -> datetime:
        tick = stamp / FRACTION
        offset = self.gps_minus_tick + (TICK_DRIFT * (tick - self.reference_tick) if self.measured else 0.0)
        return GPS_EPOCH + timedelta(seconds=tick + offset - GPS_MINUS_UTC)


def gps_record(packet: bytes) -> tuple[float, float] | None:
    """(tick, GPS seconds - tick) of an F4 record of the GPS receiver."""
    if packet[6:9] != GPS_HEADER or not crc_ok(packet):
        return None
    text = packet[24:-5].decode("latin1")  # after STX, before the 2 checksum digits, ETX and the CRC
    if not text.startswith("F4") or len(text) != 44:
        return None
    try:
        gps_seconds = int(text[3:7]) * 604800 + float(text[7:15])
    except ValueError:
        return None
    tick = int.from_bytes(packet[11:15]) + int.from_bytes(packet[15:17]) / FRACTION
    return tick, gps_seconds - tick


def collect(cadu_files: list[Path]) -> tuple[dict[int, Image], dict[str, Clock]]:
    """The images (by end of acquisition) and the clock of every dump."""
    images: dict[int, Image] = {}
    clocks: dict[str, Clock] = {}
    for path in cadu_files:
        records = []
        for _, packet in iter_packets(path):
            apid = packet_apid(packet)
            if apid == GPS_APID:
                if record := gps_record(packet):
                    records.append(record)
            elif apid == SWAP_APID and len(packet) > HEADER_LEN + METADATA_END and packet[HEADER_LEN] == IMAGE_MARKER:
                end = int.from_bytes(packet[HEADER_LEN + 1 : HEADER_LEN + 7])
                images.setdefault(end, Image(path.name)).add(packet)
        clocks[path.name] = Clock.from_records(records)
        if not records:
            print(f"warning: no GPS records in {path.name}, its image times may be off by a minute", file=sys.stderr)
    return images, clocks


def solar_radius(time: datetime) -> float:
    """Radius of the photosphere in pixels."""
    from sunpy.coordinates import sun  # slow to import, only needed with --process

    return float(sun.angular_radius(time).to_value("arcsec")) / PIXEL_SCALE


def ring_center(image: np.ndarray, radius: float) -> tuple[float, float]:
    """Coarse sun center: the shift that best matches a ring of the given radius, by FFT."""
    size = image.shape[0]
    rows, cols = np.mgrid[:size, :size]
    ring = np.abs(np.hypot(cols - size // 2, rows - size // 2) - radius) < RING_HALF_WIDTH
    correlation = np.fft.ifft2(np.fft.fft2(image) * np.conj(np.fft.fft2(ring))).real
    row, col = np.unravel_index(np.argmax(correlation), correlation.shape)
    return float(size // 2 + (col + size // 2) % size - size // 2), float(
        size // 2 + (row + size // 2) % size - size // 2
    )


def limb_circle(image: np.ndarray, x: float, y: float, radius: float) -> tuple[float, float, float, float]:
    """Circle (x, y, radius) through the brightest point of the limb on each of a fan of radii around (x, y), and the
    fraction of the radii that agree with it."""
    theta = np.linspace(0, 2 * np.pi, LIMB_ANGLES, endpoint=False)
    r = np.arange(radius * LIMB_SEARCH[0], radius * LIMB_SEARCH[1], 0.25)
    xs, ys = x + np.outer(np.cos(theta), r), y + np.outer(np.sin(theta), r)
    x0 = np.clip(np.floor(xs).astype(int), 0, image.shape[1] - 2)
    y0 = np.clip(np.floor(ys).astype(int), 0, image.shape[0] - 2)
    fx, fy = xs - x0, ys - y0
    profile = (
        image[y0, x0] * (1 - fx) * (1 - fy)
        + image[y0, x0 + 1] * fx * (1 - fy)
        + image[y0 + 1, x0] * (1 - fx) * fy
        + image[y0 + 1, x0 + 1] * fx * fy
    )
    inside = ((xs >= 0) & (ys >= 0) & (xs < image.shape[1] - 1) & (ys < image.shape[0] - 1)).all(axis=1)
    brightest = np.argmax(profile, axis=1)
    px, py = xs[np.arange(LIMB_ANGLES), brightest][inside], ys[np.arange(LIMB_ANGLES), brightest][inside]
    keep = np.ones(len(px), dtype=bool)
    cx, cy, fitted = x, y, radius
    for _ in range(3):  # least squares circle, then drop the points far from it
        if keep.sum() < 3:
            return x, y, radius, 0.0
        design = np.column_stack([2 * px[keep], 2 * py[keep], np.ones(keep.sum())])
        (cx, cy, c), *_ = np.linalg.lstsq(design, px[keep] ** 2 + py[keep] ** 2, rcond=None)
        fitted = math.sqrt(max(c + cx**2 + cy**2, 0.0))
        residual = np.hypot(px - cx, py - cy) - fitted
        keep = np.abs(residual) < 2.5 * residual[keep].std() + 0.5
    return float(cx), float(cy), fitted, float(keep.sum()) / LIMB_ANGLES


@dataclass
class SunFit:
    x: float  # 0-based pixel coordinates in the image that was fitted
    y: float
    radius: float
    expected_radius: float  # where the limb should be
    agreeing: float  # fraction of the limb radii on the circle

    @property
    def ok(self) -> bool:
        return abs(self.radius - self.expected_radius) < MAX_RADIUS_ERROR and self.agreeing >= MIN_LIMB_POINTS


def fit_sun(pixels: np.ndarray, photosphere: float) -> SunFit:
    """Sun center by limb fitting, in the spirit of the pipeline's p2sw_imgcor: a ring correlation for a start, then
    a circle through the limb maximum, twice. Works on detector values (DN), like the pipeline."""
    image = RECODE_TABLE[pixels]
    x, y = ring_center(image, photosphere)
    radius, agreeing = photosphere, 0.0
    for _ in range(2):
        x, y, radius, agreeing = limb_circle(image, x, y, photosphere)
    return SunFit(x, y, radius, photosphere + LIMB_ABOVE_PHOTOSPHERE, agreeing)


def rotated_point(x: float, y: float, steps: int, size: int) -> tuple[float, float]:
    """Where pixel (x, y) of a size x size image ends up after np.rot90(image, steps)."""
    for _ in range(steps % 4):
        x, y = y, size - 1 - x
    return x, y


def center_sun(pixels: np.ndarray, x: float, y: float) -> np.ndarray:
    """Shift the image so that (x, y) lands on its center, with bicubic interpolation (outside is black)."""
    size = pixels.shape[0]
    middle = (size - 1) / 2
    image = PILImage.fromarray(pixels)
    shifted = image.transform(
        image.size, PILImage.Transform.AFFINE, (1, 0, x - middle, 0, 1, y - middle), PILImage.Resampling.BICUBIC
    )
    return np.asarray(shifted)


def image_name(time: datetime) -> str:
    return "SWAP_" + time.strftime("%Y%m%dT%H%M%SZ")


def unique_path(folder: Path, name: str, suffix: str) -> Path:
    path = folder / f"{name}{suffix}"
    count = 0
    while path.exists():
        path = folder / f"{name}-{count}{suffix}"
        count += 1
    return path


def find_cadu_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        files += sorted(path.glob("*.cadu")) if path.is_dir() else [path]
    return files


CSV_COLUMNS = [
    "file",
    "start_utc",
    "dump",
    "packets",
    "roll",
    "rotation_ccw",
    "sun_x",
    "sun_y",
    "limb_radius",
    "expected_radius",
    "fit_ok",
    "swxcen",
    "swycen",
]


def process(pixels: np.ndarray, header: ImageHeader, start: datetime) -> tuple[np.ndarray, dict[str, object]]:
    """Turn north up, fit the sun and center it. Returns the image and its row of centers.csv."""
    steps = header.north_up_steps
    turned = np.ascontiguousarray(np.rot90(pixels, steps or 0))
    fit = fit_sun(turned, solar_radius(start))
    # the same center in the archive's level 0 convention (SWXCEN/SWYCEN): the JPEG flipped up-down, 1-based
    x, y = rotated_point(fit.x, fit.y, -(steps or 0), pixels.shape[0])
    row = {
        "roll": header.roll,
        "rotation_ccw": None if steps is None else 90 * steps,
        "sun_x": round(fit.x, 2),
        "sun_y": round(fit.y, 2),
        "limb_radius": round(fit.radius, 2),
        "expected_radius": round(fit.expected_radius, 2),
        "fit_ok": fit.ok,
        "swxcen": round(x + 1, 2),
        "swycen": round(pixels.shape[0] - y, 2),
    }
    return (center_sun(turned, fit.x, fit.y) if fit.ok else turned), row


# Level 1: the archive's calibration pipeline (p2sw_prep.py, a port of SolarSoft's p2sw_prep.pro) on our images. They
# get the level 0 header the ground would give them: the fields of the image header, the pointing from LANG_ROT, the
# ephemeris computed from the date. The detector temperature (for the dark current) and the LAR timing are not in our
# telemetry: they come from the archive's level 0 header of the same image if one is given (--archive-headers), and
# otherwise the temperature is DETECTOR_TEMP and images taken during a LAR are not caught.
LEVEL1_DIR = "level1"
CDELT = (3.1646941, 3.1760467)  # arcsec per pixel along x and y (CDELT1/2 of the archive; the pixels aren't square)
ARCHIVE_SOLAR_RADIUS = 696_000.0  # km: the archive's RSUN_ARC uses it (sunpy's default is 695 700)
DETECTOR_TEMP = -0.73  # deg C: median TEMPDARK of the archive's 281 level 1 files of Aug-Oct 2025 (-1.3 to +1.5)
LEVEL1_WORKERS = max(1, (os.cpu_count() or 2) // 2)  # ~1 GB each at the peak: all the cores would swap
ARCHIVE_MATCH = 1.0  # s: largest difference of DATE-OBS for an archive header to be the same image
OBS_MODES = {0: "Sun centered", 1: "Fixed off-pointing", 2: "Variable off-pointing", 3: "CME tracking"}
TELEMETRY_KEYS = (  # taken from the archive's header of the same image when there is one
    *("TEMP1DET", "TEMP2DET", "TTEMP1", "TTEMP2", "DTPLAR1", "DTPLAR2"),
    *("P2_X0", "P2_Y0", "P2_ROLL", "SOLAR_EP", "RSUN_ARC"),
)


def solar_ep(time: datetime) -> float:
    """Angle from ecliptic north to solar north as seen from the Earth, in degrees (SOLAR_EP of the archive, which
    this reproduces to 0.001 degree)."""
    import astropy.units as u
    from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord
    from astropy.time import Time
    from sunpy.coordinates import frames, get_earth

    t = Time(time)
    view = frames.Helioprojective(observer=get_earth(t), obstime=t)
    pole = SkyCoord(0 * u.deg, 90 * u.deg, radius=695_700 * u.km, frame=frames.HeliographicStonyhurst, obstime=t)
    ecliptic = GeocentricTrueEcliptic(equinox=t, obstime=t)
    sun = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=view).transform_to(ecliptic)
    north = SkyCoord(sun.lon, sun.lat + 0.1 * u.deg, distance=sun.distance, frame=ecliptic)

    def position_angle(point: SkyCoord) -> float:
        on_sky = point.transform_to(view)
        return math.degrees(math.atan2(-on_sky.Tx.to_value(u.arcsec), on_sky.Ty.to_value(u.arcsec)))

    return (position_angle(north) - position_angle(pole)) % 360


def rsun_arc(time: datetime) -> float:
    """Angular radius of the photosphere from the Earth, in arcsec, as the archive computes it (RSUN_ARC)."""
    from sunpy.coordinates import sun

    distance = float(sun.earth_distance(time).to_value("km"))
    return math.degrees(math.asin(ARCHIVE_SOLAR_RADIUS / distance)) * 3600


def load_archive_headers(folder: Path) -> list[tuple[datetime, fits.Header]]:
    """The level 0 headers of the archive in a folder (FITS files, or headers saved alone), by DATE-OBS."""
    headers = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in (".fits", ".fts"):
            header = fits.getheader(path)
        elif path.suffix.lower() == ".hdr":
            raw = path.read_bytes()
            end = raw.find(b"END" + b" " * 77)
            header = fits.Header.fromstring(raw[: end + 80].decode("latin1")) if end >= 0 else None
        else:
            continue
        if header is not None and header.get("LEVEL") == 0 and "DATE-OBS" in header:
            headers.append((datetime.fromisoformat(header["DATE-OBS"]).replace(tzinfo=UTC), header))
    return sorted(headers, key=lambda item: item[0])


def find_archive_header(headers: list[tuple[datetime, fits.Header]], start: datetime) -> fits.Header | None:
    """The archive header of the image that started at this time, if there is one."""
    index = bisect.bisect_left(headers, start, key=lambda item: item[0])
    near = [headers[i] for i in (index - 1, index) if 0 <= i < len(headers)]
    best = min(near, key=lambda item: abs((item[0] - start).total_seconds()), default=None)
    return best[1] if best and abs((best[0] - start).total_seconds()) <= ARCHIVE_MATCH else None


def level0(
    pixels: np.ndarray,
    header: ImageHeader,
    start: datetime,
    telemetry: fits.Header | None = None,
    detector_temp: float = DETECTOR_TEMP,
) -> tuple[np.ndarray, fits.Header]:
    """The level 0 image (detector values, FITS orientation) and header of a JPEG, as the archive's level 0 file."""
    image = np.flipud(RECODE_TABLE[pixels]).astype(np.int16)  # FITS row 1 is the last JPEG row
    roll = header.roll or 0
    crota = (270 - roll) % 360  # the archive's CROTA1 for each commanded roll
    cos, sin = round(math.cos(math.radians(crota))), round(math.sin(math.radians(crota)))
    first_row, last_row, first_col, last_col = header.window
    date_obs = start.replace(tzinfo=None).isoformat(timespec="milliseconds")
    fits_header = fits.Header()
    for key, value, comment in (
        ("DATE-OBS", date_obs, "UTC time of observation"),
        ("LEVEL", 0, "data processing level"),
        ("TELESCOP", "PROBA2", "satellite name"),
        ("INSTRUME", "SWAP", "instrument name"),
        ("WAVELNTH", 174, "[Angstrom] bandpass peak response"),
        ("OBS_MODE", OBS_MODES.get(header.obs_mode, str(header.obs_mode)), "sun_cen, fix_off, var_off, cme_track"),
        ("CAP_MODE", "CDS" if header.capture_mode == 1 else "DS", "(DS,CDS) capture mode"),
        ("EXPTIME", float(header.exposure), "[s] commanded exposure time"),
        ("BUNIT", "DN", "unit of physical value"),
        ("DATAMIN", float(image.min()), "minimum valid physical value"),
        ("DATAMAX", float(image.max()), "maximum valid physical value"),
        ("FIRSTROW", first_row + 1, "first read-out detector row"),
        ("LAST_ROW", last_row + 1, "last  read-out detector row"),
        ("FIRSTCOL", first_col + 1, "first read-out detector column"),
        ("LAST_COL", last_col + 1, "last  read-out detector column"),
        ("REBIN", "off", "on-board rebin (2x2 pixel average)"),
        ("WCSNAME", "Helioprojective-cartesian", "level < 1: aligned with ecliptic North"),
        ("CTYPE1", "HPLN-TAN", "WCS axis X"),
        ("CTYPE2", "HPLT-TAN", "WCS axis Y"),
        ("CUNIT1", "arcsec", "WCS axis X units"),
        ("CUNIT2", "arcsec", "WCS axis Y units"),
        ("CD1_1", CDELT[0] * cos, "WCS coordinate description matrix"),
        ("CD1_2", -CDELT[1] * sin, "WCS coordinate description matrix"),
        ("CD2_1", CDELT[0] * sin, "WCS coordinate description matrix"),
        ("CD2_2", CDELT[1] * cos, "WCS coordinate description matrix"),
        ("CDELT1", CDELT[0], "[arcsec] average pixel scale along axis 1"),
        ("CDELT2", CDELT[1], "[arcsec] average pixel scale along axis 2"),
        ("CRVAL1", 0.0, "[arcsec] level < 1: unknown"),
        ("CRVAL2", 0.0, "[arcsec] level < 1: unknown"),
        ("CRPIX1", 512.5, "[pixel] reference point axis 1"),
        ("CRPIX2", 512.5, "[pixel] reference point axis 2"),
        ("CROTA1", float(crota), "[deg] axis 1 to WCS rotation angle"),
        ("CROTA2", float(crota), "[deg] axis 2 to WCS rotation angle"),
        ("SOLAR_EP", solar_ep(start), "[deg] s/c ecliptic North to solar North angle"),
        ("RSUN_ARC", rsun_arc(start), "[arcsec] photospheric solar radius"),
        ("TEMP1DET", detector_temp, "[Celsius] detector temperature (assumed)"),
        ("TEMP2DET", detector_temp, "[Celsius] detector temperature (assumed)"),
        ("TTEMP1", date_obs, "UTC time of detector temp 1st sample"),
        ("TTEMP2", date_obs, "UTC time of detector temp 2nd sample"),
        ("SACQTIME", header.start // 4 + 2**42, "[ct] OBET start acquisition time"),
        ("EACQTIME", header.end // 4 + 2**42, "[ct] OBET end   acquisition time"),
        ("LANG_ROT", header.lang_rot, "commanded large angle rotation component"),
        ("PAV_ROT0", header.pav_rot[0], "commanded paving rotation component 0"),
        ("PAV_ROT1", header.pav_rot[1], "commanded paving rotation component 1"),
        ("PGA_GAIN", 1, "PGA gain"),
        ("LED_POW", "off", "calibration LED power"),
        ("RECODING", "fixed", "(off,fixed,adaptive) on-board recoding"),
        ("COMPRESS", "jpeg", "(off,lzw,jpeg) on-board compression algorithm"),
        ("PN", header.priority, "on-board priority number"),
        ("SIZCOMPI", header.compressed_size, "size of on-board compressed image"),
        ("NPRESCR", 0, "# preserved pixels - cosmic rays"),
    ):
        fits_header[key] = (value, comment)
    if telemetry is not None:
        for key in TELEMETRY_KEYS:
            if key in telemetry:
                fits_header.set(key, telemetry[key], telemetry.comments[key])
        fits_header.add_history(f"Telemetry from the archive's {telemetry.get('FILENAME', 'level 0 header')}")
    else:
        fits_header.add_history(f"No telemetry: detector temperature assumed {detector_temp} C, LAR timing unknown")
    return image, fits_header


@dataclass
class Level1Job:
    pixels: np.ndarray
    header: ImageHeader
    start: datetime
    telemetry: fits.Header | None
    detector_temp: float
    folder: Path
    name: str
    write_fits: bool = True  # the level 1 FITS file next to the PNG preview
    scaling: str = "quicklook"  # of the PNG: "quicklook" (the P2SC's quicklooks and movies) or "p2sw" (p2sw_write_png)


def make_level1(job: Level1Job) -> tuple[Path | None, np.ndarray | None, str]:
    """Run the pipeline on one image: (FITS file, or the PNG without write_fits; display frame; note), or
    (None, None, why not)."""
    import p2sw_prep

    image, header = level0(job.pixels, job.header, job.start, job.telemetry, job.detector_temp)
    try:
        level1, header1 = p2sw_prep.prep(image, header)
    except p2sw_prep.PrepError as error:
        return None, None, f"no level 1: {error}"
    path = job.folder / f"{job.name}.fits"
    if job.write_fits:
        p2sw_prep.write_level1(path, level1, header1)
    frame = np.flipud(p2sw_prep.SCALINGS[job.scaling](level1))
    PILImage.fromarray(frame).save(path.with_suffix(".png"))
    path = path if job.write_fits else path.with_suffix(".png")
    source = "archive telemetry" if job.telemetry is not None else f"assumed {job.detector_temp} C"
    return path, frame, f"sun at ({header1['SWXCEN']}, {header1['SWYCEN']}), dark at {source}"


def make_aligned(job: Level1Job) -> tuple[Path | None, np.ndarray | None, str]:
    """Only the geometry of the pipeline (p2sw_imgcor) on one image, without its calibration: the JPEG's own grey
    levels, turned by the commanded roll (90 degree steps), the sun center found by the pipeline's limb fit (on the
    detector values), then the -0.3 degree mount offset and ecliptic -> solar north (SOLAR_EP) rotations, the pixels
    made square and the sun put at the center, all in one cubic interpolation. (PNG, frame, note) or (None, None,
    why not)."""
    from dataclasses import replace

    import p2sw_prep

    image, header = level0(job.pixels, job.header, job.start, job.telemetry, job.detector_temp)
    opts = p2sw_prep.Options()
    try:
        header = p2sw_prep.p2sw_hdrchk(header, opts)  # refuses LAR images if the archive's header says so
    except p2sw_prep.PrepError as error:
        return None, None, f"not aligned: {error}"
    x, y = p2sw_prep.find_sun_center(header, image.astype(np.float32))
    grey = np.flipud(job.pixels).astype(np.float32)  # FITS orientation, like the level 0 image
    aligned = p2sw_prep.p2sw_imgcor(header, grey, replace(opts, sun_cen_x=x, sun_cen_y=y))
    frame = np.flipud(np.clip(np.round(aligned), 0, 255).astype(np.uint8))
    path = job.folder / f"{job.name}.png"
    PILImage.fromarray(frame).save(path)
    return path, frame, f"sun at ({x + 1:.2f}, {y + 1:.2f}), roll {job.header.roll}, solar_ep {header['SOLAR_EP']:.2f}"


def same_size(frames: list[np.ndarray]) -> list[PILImage.Image]:
    """Frames as PIL images; those of another size than the first are resized to it."""
    first = PILImage.fromarray(frames[0])
    return [first] + [
        PILImage.fromarray(frame).resize(first.size, PILImage.Resampling.BICUBIC)
        if frame.shape != frames[0].shape
        else PILImage.fromarray(frame)
        for frame in frames[1:]
    ]


def save_gif(frames: list[np.ndarray], path: Path) -> None:
    """Animated GIF of grey level frames."""
    first, *rest = same_size(frames)
    first.save(path, save_all=True, append_images=rest, duration=GIF_FRAME_MS, loop=0)


def save_mp4(frames: list[np.ndarray], path: Path) -> None:
    """H.264 video of grey level frames, at the frame rate of the GIF."""
    import imageio.v2 as imageio  # brings its own ffmpeg, only needed with --mp4

    with imageio.get_writer(path, fps=1000 / GIF_FRAME_MS, codec="libx264", macro_block_size=2) as writer:
        for frame in same_size(frames):
            writer.append_data(np.asarray(frame.convert("RGB")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help=".cadu files, or folders containing them")
    parser.add_argument("-o", "--out_path", type=Path, default=Path("images"), help="output folder (default: images)")
    parser.add_argument(
        "--process",
        action="store_true",
        help="also save the images turned north up and with the sun centered by limb fitting, in <out>/processed",
    )
    parser.add_argument(
        "--level1",
        action="store_true",
        help=f"also make level 1 FITS files (and PNG previews) with the archive's calibration pipeline, in "
        f"<out>/{LEVEL1_DIR}",
    )
    parser.add_argument(
        "--archive-headers",
        type=Path,
        help="folder of the archive's level 0 headers or FITS files: with --level1, their detector temperature and "
        "LAR timing are used for the same images",
    )
    parser.add_argument(
        "--detector-temp",
        type=float,
        default=DETECTOR_TEMP,
        help=f"with --level1, the detector temperature (C) for the dark current of images without an archive header "
        f"(default: {DETECTOR_TEMP})",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help=f"also stitch all the images, in time order, into <out>/{GIF_NAME} (the level 1 ones with --level1, "
        "else the processed ones with --process)",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help=f"also render all the images, in time order, as <out>/{MP4_NAME} (the level 1 ones with --level1, else "
        "the processed ones with --process)",
    )
    args = parser.parse_args()

    cadu_files = find_cadu_files(args.inputs)
    if missing := [path for path in cadu_files if not path.is_file()]:
        parser.error(f"not a file: {', '.join(map(str, missing))}")

    images, clocks = collect(cadu_files)
    args.out_path.mkdir(parents=True, exist_ok=True)
    processed_dir = args.out_path / "processed"
    level1_dir = args.out_path / LEVEL1_DIR
    rows: list[dict[str, object]] = []
    frames: list[np.ndarray] = []
    jobs: list[Level1Job] = []
    if args.process:
        processed_dir.mkdir(exist_ok=True)
    archive = load_archive_headers(args.archive_headers) if args.level1 and args.archive_headers else []
    if args.level1:
        level1_dir.mkdir(exist_ok=True)

    saved = skipped = 0
    for _, image in sorted(images.items()):
        for keep_bit_errors in (True, False):  # bit errors are usually harmless, but can wreck a decode
            assembled = assemble(image, keep_bit_errors)
            pixels = decode(assembled.jpeg) if assembled else None
            if pixels is not None:
                break
        if assembled is None or pixels is None:
            skipped += 1
            continue

        start = clocks[image.dump].utc(assembled.header.start)
        path = unique_path(args.out_path, image_name(start), ".jpeg")
        path.write_bytes(assembled.jpeg)
        state = "complete" if assembled.complete else f"cut after {assembled.packets} of {assembled.packets_expected}"
        notes = [state] + ([f"{assembled.bad_crc} with bit errors"] if assembled.bad_crc else [])
        notes += [] if assembled.header_ok else ["header has bit errors"]

        if args.level1:
            telemetry = find_archive_header(archive, start)
            name = base = "swap_lv1_" + start.strftime("%Y%m%d_%H%M%S")
            while name in {job.name for job in jobs}:  # two images in the same second
                name = f"{base}-{sum(job.name.startswith(base) for job in jobs)}"
            jobs.append(Level1Job(pixels, assembled.header, start, telemetry, args.detector_temp, level1_dir, name))

        if args.process:
            centered, row = process(pixels, assembled.header, start)
            PILImage.fromarray(centered).save(processed_dir / f"{path.stem}.png")
            rows.append(
                {"file": path.name, "start_utc": start.isoformat(), "dump": image.dump, "packets": state} | row
            )
            fit = f"sun at ({row['sun_x']}, {row['sun_y']})" if row["fit_ok"] else "limb fit failed, not shifted"
            notes.append(f"roll {row['roll']}, {fit}")
            pixels = centered

        if (args.gif or args.mp4) and not args.level1:
            frames.append(pixels)

        print(f"{path} ({', '.join(notes)})")
        saved += 1

    if jobs:  # the pipeline takes a few seconds per image: run them side by side
        made = 0
        with ProcessPoolExecutor(max_workers=LEVEL1_WORKERS) as pool:
            for job, (fits_path, frame, note) in zip(jobs, pool.map(make_level1, jobs), strict=True):
                print(f"{fits_path or level1_dir / (job.name + '.fits')} ({note})")
                if frame is not None:
                    made += 1
                    if args.gif or args.mp4:
                        frames.append(frame)
        print(f"made {made} level 1 images of {len(jobs)}", file=sys.stderr)

    if args.process:
        with (processed_dir / "centers.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    if frames:
        for enabled, name, save in ((args.gif, GIF_NAME, save_gif), (args.mp4, MP4_NAME, save_mp4)):
            if enabled:
                save(frames, args.out_path / name)
                print(f"{args.out_path / name} ({len(frames)} frames)")

    print(f"saved {saved} images, skipped {skipped} without a usable first packet", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

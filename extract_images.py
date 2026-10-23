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

With --gif all the images are also stitched, in time order, into `<out>/swap.gif`: the processed ones with --process,
the JPEGs as they came down otherwise. --mp4 does the same into `<out>/swap.mp4`.
"""

import argparse
import binascii
import csv
import io
import math
import struct
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
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
        "--gif",
        action="store_true",
        help=f"also stitch all the images, in time order, into <out>/{GIF_NAME} (the processed ones with --process)",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help=f"also render all the images, in time order, as <out>/{MP4_NAME} (the processed ones with --process)",
    )
    args = parser.parse_args()

    cadu_files = find_cadu_files(args.inputs)
    if missing := [path for path in cadu_files if not path.is_file()]:
        parser.error(f"not a file: {', '.join(map(str, missing))}")

    images, clocks = collect(cadu_files)
    args.out_path.mkdir(parents=True, exist_ok=True)
    processed_dir = args.out_path / "processed"
    rows: list[dict[str, object]] = []
    frames: list[np.ndarray] = []
    if args.process:
        processed_dir.mkdir(exist_ok=True)

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

        if args.process:
            centered, row = process(pixels, assembled.header, start)
            PILImage.fromarray(centered).save(processed_dir / f"{path.stem}.png")
            rows.append(
                {"file": path.name, "start_utc": start.isoformat(), "dump": image.dump, "packets": state} | row
            )
            fit = f"sun at ({row['sun_x']}, {row['sun_y']})" if row["fit_ok"] else "limb fit failed, not shifted"
            notes.append(f"roll {row['roll']}, {fit}")
            pixels = centered

        if args.gif or args.mp4:
            frames.append(pixels)

        print(f"{path} ({', '.join(notes)})")
        saved += 1

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

"""Extract the SWAP images (APID 20) from raw PROBA-2 .cadu dumps and save them as JPEG files.

Follows satdump's SWAPReader: the packets of one image share the timestamp in the first 6 payload bytes, the first
packet has 78 bytes of metadata in front of the JPEG data. An image that libjpeg can't make sense of is skipped, as
satdump writes a RAW image for those. satdump's masks are not applied.
"""

import argparse
import io
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image as PILImage

from cadu import HEADER_LEN, iter_packets, packet_apid, packet_sequence_count

SWAP_APID = 20
SECONDARY_HEADER_LEN = 14
METADATA_LEN = 78  # only present in the first packet of an image
TIME_OFFSET_DAYS = 18630
TIME_SHIFT_SECONDS = 4 * 3600  # satdump shifts every timestamp by this, kept so the file names match
MAX_PLAUSIBLE_DAYS = 1000  # other packets with APID 20 (housekeeping) have garbage where the time should be
JPEG_SOI = b"\xff\xd8"


@dataclass
class Image:
    metadata: bytes = b""  # secondary header and metadata block of the first packet
    jpeg: bytearray = field(default_factory=bytearray)
    segments: int = 0
    sequence_counts: set[int] = field(default_factory=set)


def packet_timestamp(payload: bytes) -> int | None:
    days = int.from_bytes(payload[0:2])
    milliseconds = int.from_bytes(payload[2:6])
    if days > MAX_PLAUSIBLE_DAYS:
        return None
    return (TIME_OFFSET_DAYS + days) * 86400 + milliseconds // 1000 + TIME_SHIFT_SECONDS


def collect_images(cadu_files: list[Path]) -> dict[int, Image]:
    images: dict[int, Image] = {}
    for path in cadu_files:
        for _, packet in iter_packets(path):
            payload = packet[HEADER_LEN:]
            if packet_apid(packet) != SWAP_APID or len(payload) < SECONDARY_HEADER_LEN + 2:
                continue
            timestamp = packet_timestamp(payload)
            if timestamp is None:
                continue

            image = images.setdefault(timestamp, Image())
            sequence_count = packet_sequence_count(packet)
            if sequence_count in image.sequence_counts:
                continue  # the same packet in two overlapping dumps
            image.sequence_counts.add(sequence_count)

            start = SECONDARY_HEADER_LEN + (METADATA_LEN if image.segments == 0 else 0)
            if image.segments == 0:
                image.metadata = payload[:start]
            image.jpeg += payload[start:-1]  # satdump leaves out the last byte of every packet
            image.segments += 1
    return images


def decodes_like_satdump(jpeg: bytes) -> bool:
    """Truncated JPEGs are fine for satdump, corrupt ones are not."""
    try:
        PILImage.open(io.BytesIO(jpeg)).load()
    except OSError as error:
        return "truncated" in str(error)
    return True


def image_name(timestamp: int) -> str:
    return "SWAP_" + datetime.fromtimestamp(timestamp, UTC).strftime("%Y%m%dT%H%M%SZ")


def unique_path(folder: Path, name: str) -> Path:
    path = folder / f"{name}.jpeg"
    suffix = 0
    while path.exists():
        path = folder / f"{name}-{suffix}.jpeg"
        suffix += 1
    return path


def find_cadu_files(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        files += sorted(path.glob("*.cadu")) if path.is_dir() else [path]
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help=".cadu files, or folders containing them")
    parser.add_argument("-o", "--out_path", type=Path, default=Path("images"), help="output folder (default: images)")
    args = parser.parse_args()

    cadu_files = find_cadu_files(args.inputs)
    if missing := [path for path in cadu_files if not path.is_file()]:
        parser.error(f"not a file: {', '.join(map(str, missing))}")

    images = collect_images(cadu_files)
    args.out_path.mkdir(parents=True, exist_ok=True)

    saved = skipped = 0
    for timestamp, image in sorted(images.items()):
        # without its first packet an image has no JPEG header
        if not image.jpeg.startswith(JPEG_SOI) or not decodes_like_satdump(bytes(image.jpeg)):
            skipped += 1
            continue
        path = unique_path(args.out_path, image_name(timestamp))
        path.write_bytes(image.jpeg)
        print(f"{path} ({image.segments} segments)")
        saved += 1

    print(f"saved {saved} images, skipped {skipped} that can't be decoded", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

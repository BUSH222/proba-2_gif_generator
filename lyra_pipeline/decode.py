"""LYRA's raw samples (APID 21) and its temperature (APID 2) out of a .cadu dump.

The layout is in NOTES.md. In short: an APID 21 packet holds 50 s of the four channels at 10 Hz, as a first value and
500 differences per channel, compressed with Unix `compress`. The even and odd samples of LYRA's 20 Hz series go in
alternate packets, so the two packet streams together are the full cadence. Time marks at the end of a packet tie the
sample counter to the on-board clock, and the GPS records (APID 18) tie the clock to UTC. The temperature of the
nominal unit, which sets its dark current, is in the housekeeping reports with structure id 70.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from swap_pipeline.cadu import iter_packets, packet_apid
from swap_pipeline.extract_images import (
    FRACTION,
    GPS_APID,
    GPS_EPOCH,
    GPS_MINUS_UTC,
    TICK_DRIFT,
    Clock,
    crc_ok,
    gps_record,
)

if TYPE_CHECKING:
    from pathlib import Path

LYRA_APID = 21
HK_APID = 2
CHANNELS = 4
CHANNEL_NAMES = ("Lyman alpha", "Herzberg", "Aluminium filter", "Zirconium filter")

# APID 21 packet: bytes into the packet
SECONDARY_HEADER = bytes.fromhex("10088b")
FIRST_COUNTER_OFFSET = 19  # 3 bytes: sample counter of the first sample
KIND_OFFSET = 40  # 2 bytes: 0 for a data packet (4 packets in our dumps hold something else)
BLOCK_OFFSET = 96  # the `compress` block, up to the CRC
COMPRESS_MAGIC = b"\x1f\x9d"
CRC_LEN = 2
SAMPLES = 501  # per channel and packet, 2 counter units apart
SAMPLE_BYTES = 3
SAMPLE_SECTION = CHANNELS * SAMPLES * SAMPLE_BYTES
MARKS_OFFSET = SAMPLE_SECTION + 4 + 500 * 4  # after the samples, the first counter again and 500 flag words
MARK_BYTES = 7  # tick (4), fraction (2), 00; then one 4 byte counter per mark
COUNTER_RATE = 20  # sample counter units per second

# housekeeping report (PUS 3/25) with LYRA's temperatures
TEMPERATURE_REPORT_LEN = 150
TEMPERATURE_SID = 70
TEMPERATURE_OFFSET = 55  # 2 bytes: unit 2, the nominal unit (units 1 and 3 are at 47 and 63)

QFACTOR_MARK = 4  # the archive's QFACTOR at a sample with a time mark
QFACTOR_NORMAL = 3

CLEAR = 256
CODES_PER_GROUP = 8


def uncompress(data: bytes) -> bytes:
    """Decode a Unix `compress` (.Z) stream: magic 1F 9D, a flags byte, then LZW codes of 9 bits and up, packed least
    significant bit first in groups of 8 (a change of width or a table reset skips the rest of the group)."""
    if data[:2] != COMPRESS_MAGIC or len(data) < 3:
        raise ValueError("not a compress stream")
    max_bits, block_mode, body = data[2] & 0x1F, bool(data[2] & 0x80), data[3:]
    total = len(body) * 8
    table = {i: bytes([i]) for i in range(256)}
    free = 257 if block_mode else 256
    bits, group_start, index = 9, 0, 0
    previous: bytes | None = None
    out = bytearray()
    while (position := group_start + index * bits) + bits <= total:
        chunk = int.from_bytes(body[position >> 3 : (position >> 3) + 3], "little")
        code = (chunk >> (position & 7)) & ((1 << bits) - 1)
        index += 1
        if index == CODES_PER_GROUP:
            group_start, index = group_start + CODES_PER_GROUP * bits, 0
        if block_mode and code == CLEAR:
            table = {i: bytes([i]) for i in range(256)}
            free, previous = 257, None
            if index:
                group_start += CODES_PER_GROUP * bits
            bits, index = 9, 0
            continue
        if code in table:
            entry = table[code]
        elif code == free and previous is not None:
            entry = previous + previous[:1]
        else:
            raise ValueError(f"bad code {code} at bit {position}")
        out += entry
        if previous is not None and free < (1 << max_bits):
            table[free] = previous + entry[:1]
            free += 1
        previous = entry
        if free > (1 << bits) - 1 and bits < max_bits:
            if index:
                group_start, index = group_start + CODES_PER_GROUP * bits, 0
            bits += 1
    return bytes(out)


def signed24(data: bytes) -> np.ndarray:
    values = np.frombuffer(data, dtype=np.uint8).reshape(-1, SAMPLE_BYTES).astype(np.int64)
    unsigned = values[:, 0] << 16 | values[:, 1] << 8 | values[:, 2]
    return np.where(unsigned >= 1 << 23, unsigned - (1 << 24), unsigned)


def is_data_packet(packet: bytes) -> bool:
    return (
        len(packet) > BLOCK_OFFSET + 3 + CRC_LEN
        and packet[6:9] == SECONDARY_HEADER
        and int.from_bytes(packet[KIND_OFFSET : KIND_OFFSET + 2]) == 0
        and packet[BLOCK_OFFSET : BLOCK_OFFSET + 2] == COMPRESS_MAGIC
    )


def is_temperature_report(packet: bytes) -> bool:
    return (
        len(packet) == TEMPERATURE_REPORT_LEN
        and packet[7:9] == b"\x03\x19"  # PUS service 3, subtype 25: housekeeping report
        and packet[21] == TEMPERATURE_SID
        and crc_ok(packet)
    )


def packet_samples(packet: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample counters (501), counts (4, 501), clock stamps (ticks x 65536) and whether a time mark is at a sample."""
    data = uncompress(packet[BLOCK_OFFSET:-CRC_LEN])
    tail = data[MARKS_OFFSET:]
    marks = len(tail) // (MARK_BYTES + 4)
    if len(data) < SAMPLE_SECTION or marks == 0 or len(tail) != marks * (MARK_BYTES + 4):
        raise ValueError(f"unexpected block of {len(data)} bytes")
    counts = np.cumsum(signed24(data[:SAMPLE_SECTION]).reshape(CHANNELS, SAMPLES), axis=1)  # first value + differences
    counters = int.from_bytes(packet[FIRST_COUNTER_OFFSET : FIRST_COUNTER_OFFSET + 3]) + 2 * np.arange(SAMPLES)
    mark_stamps = np.array([int.from_bytes(tail[i * MARK_BYTES : i * MARK_BYTES + 6]) for i in range(marks)])
    mark_counters = np.frombuffer(tail[marks * MARK_BYTES :], dtype=">u4").astype(np.int64)
    stamps = mark_stamps[0] + (counters - mark_counters[0]) * FRACTION / COUNTER_RATE
    return counters, counts, stamps, np.isin(counters, mark_counters)


def utc_seconds(clock: Clock, stamps: np.ndarray, day: date) -> np.ndarray:
    """Clock.utc for arrays of stamps, in seconds from the midnight that starts `day` (UTC)."""
    ticks = np.asarray(stamps, dtype=np.float64) / FRACTION
    offset = clock.gps_minus_tick + (TICK_DRIFT * (ticks - clock.reference_tick) if clock.measured else 0.0)
    midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return ticks + offset - GPS_MINUS_UTC - (midnight - GPS_EPOCH).total_seconds()


@dataclass
class LyraDump:
    """The LYRA samples of one dump: level 0, at LYRA's full cadence (20 Hz)."""

    name: str
    day: date  # UTC day of the first sample; the times count seconds from its midnight, as the archive's TIME
    seconds: np.ndarray  # (n,)
    counts: np.ndarray  # (4, n) counts per integration
    qfactor: np.ndarray  # (n,) 4 at a time mark, 3 elsewhere
    report_seconds: np.ndarray  # times of the temperature reports
    report_raw: np.ndarray  # their raw unit 2 temperature words
    gps_clock: bool  # False: no GPS records, the times are off by up to a minute
    skipped: int  # LYRA packets that failed their CRC or couldn't be decoded

    @property
    def start(self) -> datetime:
        return datetime(self.day.year, self.day.month, self.day.day, tzinfo=UTC) + timedelta(seconds=self.seconds[0])

    @property
    def end(self) -> datetime:
        return datetime(self.day.year, self.day.month, self.day.day, tzinfo=UTC) + timedelta(seconds=self.seconds[-1])


def read_dump(path: Path) -> LyraDump | None:
    """The LYRA samples of a dump, both packet streams merged in counter order; None without LYRA packets."""
    lyra: list[bytes] = []
    gps: list[tuple[float, float]] = []
    reports: list[bytes] = []
    for _, packet in iter_packets(path):
        apid = packet_apid(packet)
        if apid == GPS_APID:
            if record := gps_record(packet):
                gps.append(record)
        elif apid == LYRA_APID:
            lyra.append(packet)
        elif apid == HK_APID and is_temperature_report(packet):
            reports.append(packet)

    parts = []
    skipped = 0
    for packet in lyra:
        if not is_data_packet(packet) or not crc_ok(packet):
            skipped += 1
            continue
        try:
            parts.append(packet_samples(packet))
        except ValueError:
            skipped += 1
    if not parts:
        return None

    counters = np.concatenate([part[0] for part in parts])
    order = np.argsort(counters, kind="stable")
    order = order[np.r_[True, np.diff(counters[order]) != 0]]  # each sample once
    counts = np.concatenate([part[1] for part in parts], axis=1)[:, order]
    stamps = np.concatenate([part[2] for part in parts])[order]
    at_mark = np.concatenate([part[3] for part in parts])[order]

    clock = Clock.from_records(gps)
    day = clock.utc(int(stamps[0])).date()
    report_stamps = np.array([int.from_bytes(packet[11:17]) for packet in reports], dtype=np.float64)
    report_seconds = utc_seconds(clock, report_stamps, day)
    report_order = np.argsort(report_seconds)
    return LyraDump(
        name=path.name,
        day=day,
        seconds=utc_seconds(clock, stamps, day),
        counts=counts,
        qfactor=np.where(at_mark, QFACTOR_MARK, QFACTOR_NORMAL).astype(np.uint8),
        report_seconds=report_seconds[report_order],
        report_raw=np.array(
            [int.from_bytes(packet[TEMPERATURE_OFFSET : TEMPERATURE_OFFSET + 2]) for packet in reports], dtype=float
        )[report_order],
        gps_clock=clock.measured,
        skipped=skipped,
    )

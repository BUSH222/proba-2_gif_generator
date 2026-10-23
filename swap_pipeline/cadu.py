"""Reading CCSDS source packets out of raw .cadu dumps."""

from typing import TYPE_CHECKING

import numpy as np

from . import rs

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

ASM_LEN = 4
CADU_LEN = 1279  # 4 byte ASM + 1275 byte RS(255,223) codeblock (interleave 5)
FRAME_LEN = 1115  # transfer frame length, excluding the 160 bytes of Reed-Solomon parity
HEADER_LEN = 6
MPDU_DATA_SIZE = 1103  # the frame minus its 6 byte header, 4 byte operational control field and 2 byte CRC
NO_PACKET_START = 2047  # first header pointer: no packet starts in this frame
IDLE_FRAME = 2046


INTERLEAVE = 5
CHUNK_FRAMES = 2000


def iter_frames(path: Path, correct_errors: bool = False) -> Iterator[bytes]:
    """Yield the transfer frames of a dump, optionally Reed-Solomon corrected (uncorrectable ones are dropped)."""
    data = path.read_bytes()
    count = len(data) // CADU_LEN
    if not correct_errors:
        for start in range(0, count * CADU_LEN, CADU_LEN):
            yield data[start + ASM_LEN : start + ASM_LEN + FRAME_LEN]
        return
    for chunk_start in range(0, count, CHUNK_FRAMES):
        chunk_end = min(chunk_start + CHUNK_FRAMES, count)
        raw = data[chunk_start * CADU_LEN : chunk_end * CADU_LEN]
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, CADU_LEN)[:, ASM_LEN:]
        conventional = np.frombuffer(frames.tobytes().translate(rs.TO_CONVENTIONAL), dtype=np.uint8)
        codewords = conventional.reshape(-1, rs.N, INTERLEAVE).transpose(0, 2, 1)  # (frames, codeword, symbol)
        damaged = rs.syndromes(codewords, count=2).any(
            axis=-1
        )  # a damaged codeword fails two syndromes (almost) always

        for index in range(len(frames)):
            frame = bytearray(conventional[index * (rs.N * INTERLEAVE) : (index + 1) * (rs.N * INTERLEAVE)].tobytes())
            if damaged[index].any():
                for j in np.nonzero(damaged[index])[0]:
                    codeword = bytearray(frame[j::INTERLEAVE])
                    if not rs.correct(codeword):
                        break
                    frame[j::INTERLEAVE] = codeword
                else:
                    yield bytes(frame[:FRAME_LEN]).translate(rs.TO_DUAL)
                continue
            yield bytes(frame[:FRAME_LEN]).translate(rs.TO_DUAL)


class Demuxer:
    """Builds the source packets of one virtual channel out of its frames, quirks included.

    This is a port of satdump's ccsds_tm Demuxer: it never looks at the frame counter, doesn't pad lost frames and
    just keeps appending to the open packet, which is what makes its output (and its images) what they are.
    """

    def __init__(self, data_size: int = MPDU_DATA_SIZE) -> None:
        self.data_size = data_size
        self.header_buffer = bytearray(HEADER_LEN)
        self.in_header_buffer = 0
        self.in_header = False
        self.working = False
        self.header = b""
        self.payload = bytearray()
        self.remaining = 0
        self.total_length = 0

    def _read_packet(self, header: bytes | bytearray) -> None:
        self.working = True
        self.header = bytes(header)
        payload_length = int.from_bytes(header[4:6]) + 1
        self.total_length = payload_length + HEADER_LEN
        self.remaining = payload_length

    def _push_payload(self, data: bytes, start: int, length: int) -> None:
        if length > 0:
            self.payload += data[start : start + length]
        self.remaining -= length

    def _abort(self) -> None:
        self.working = False
        self.payload.clear()
        self.remaining = 0

    def _finish(self, packets: list[bytes]) -> None:
        packets.append(self.header + bytes(self.payload))
        self.payload.clear()
        self.remaining = 0
        self.working = False

    def _start_partial_header(self, data: bytes, start: int) -> None:
        self.in_header = True
        self.header_buffer[: self.data_size - start] = data[start : self.data_size]
        self.in_header_buffer = self.data_size - start

    def work(self, data: bytes, first_header_ptr: int) -> list[bytes]:
        """Feed the data zone of one frame, returns the packets completed by it."""
        packets: list[bytes] = []
        size = self.data_size
        if first_header_ptr < NO_PACKET_START and first_header_ptr >= size:
            return packets
        if first_header_ptr == IDLE_FRAME:
            return packets

        offset = 0
        if self.in_header:
            self.in_header = False
            missing = HEADER_LEN - self.in_header_buffer
            self.header_buffer[self.in_header_buffer :] = data[:missing]
            offset = missing
            self.in_header_buffer = HEADER_LEN
            self._read_packet(self.header_buffer)

        if self.remaining > 0 and self.working:
            if first_header_ptr < NO_PACKET_START:  # a header takes priority
                to_write = (
                    first_header_ptr + 1 - offset if self.remaining + offset > first_header_ptr + 1 else self.remaining
                )
                self._push_payload(data, offset, to_write)
                self.remaining = 0
            else:
                to_write = size - offset if self.remaining + offset > size - offset else self.remaining
                self._push_payload(data, offset, to_write)

        if self.remaining == 0 and self.working:
            self._finish(packets)

        if first_header_ptr >= NO_PACKET_START:
            return packets
        if first_header_ptr + HEADER_LEN < size:
            self._read_packet(data[first_header_ptr : first_header_ptr + HEADER_LEN])
            if size > first_header_ptr + self.total_length:  # more than one packet starts in this frame
                if first_header_ptr + self.total_length < size:
                    self._push_payload(data, first_header_ptr + HEADER_LEN, self.total_length - HEADER_LEN)
                    self._finish(packets)
                else:
                    self._abort()
                next_header = first_header_ptr + self.total_length
                while next_header < size:
                    if next_header + HEADER_LEN < size:
                        self._read_packet(data[next_header : next_header + HEADER_LEN])
                        room = size - (next_header + HEADER_LEN)
                        self._push_payload(data, next_header + HEADER_LEN, min(self.remaining, room))
                    elif next_header < size:
                        self._start_partial_header(data, next_header)
                        break
                    if self.remaining == 0 and self.working:
                        self._finish(packets)
                    next_header += self.total_length
            elif self.working:
                room = size - (first_header_ptr + HEADER_LEN)
                self._push_payload(data, first_header_ptr + HEADER_LEN, min(self.remaining, room))
        elif first_header_ptr < size:
            self._start_partial_header(data, first_header_ptr)
        return packets


def iter_packets(path: Path, correct_errors: bool = False) -> Iterator[tuple[int, bytes]]:
    """Yield (vcid, packet bytes) for every source packet in a dump."""
    demuxers: dict[int, Demuxer] = {}
    for frame in iter_frames(path, correct_errors):
        vcid = (frame[1] >> 1) & 0x7
        first_header_ptr = int.from_bytes(frame[4:6]) & 0x7FF
        data = frame[HEADER_LEN : HEADER_LEN + MPDU_DATA_SIZE]
        for packet in demuxers.setdefault(vcid, Demuxer()).work(data, first_header_ptr):
            yield vcid, packet


def packet_apid(packet: bytes) -> int:
    return int.from_bytes(packet[0:2]) & 0x7FF


def packet_sequence_count(packet: bytes) -> int:
    return int.from_bytes(packet[2:4]) & 0x3FFF

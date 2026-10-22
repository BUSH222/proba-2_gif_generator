# Experimental environment
Trying to extract LYRA and the calibration data from proba-2 cadus

TODO: To settle APID 22, compare it with a LYRA file for August–October 2025 from the P2SC archive. A matching LED or calibration cycle would confirm it.
TODO: Check if PROBA-V GPS log format matches PROBA-2's to get more out of APID 18

## APID 20 (SWAP images)

All offsets are bytes into the packet payload, i.e. after the 6 byte CCSDS primary header, and all numbers are
big-endian. `extract_images.py` follows satdump's SWAPReader and `image_rotation.ipynb` is where the metadata is poked at.
Everything below was measured on the 13 dumps in `cadus/`, 324 images that still have their first packet.

### From CADU to packet

The dumps are not randomized and the frames are not Reed-Solomon corrected. `cadu.py` does what satdump's PROBA module
does with them, and the images that `extract_images.py` writes match satdump's: for `proba2_dump.cadu` it saves the same
25 images, 24 of them pixel for pixel and one with a difference of at most 1 grey level in 2 % of the pixels.

- A CADU is 1279 bytes: 4 byte sync marker, 1115 byte CCSDS TM transfer frame, 160 bytes of Reed-Solomon parity.
- The frame is 6 byte header (the last two bytes hold the 11 bit first header pointer, 2047 = no packet starts here), 1103
  bytes of data, 4 bytes OCF and a 2 byte CRC-16 (CCITT, init `0xFFFF`) over the 1113 before it. Virtual channel 3 carries
  SWAP.
- Packets are cut from the data of consecutive frames of a virtual channel. satdump's demuxer (ported in `cadu.py`) ignores
  the frame counter, so a lost frame is not padded and not noticed: the next frames just continue the open packet. Zero
  padding the hole instead gets slightly different images in the dumps with lost frames.
- Taking the CRC (2 bytes) for packet data was the bug that cost us every second packet and put stray bytes in the JPEGs.
- The parity is a CCSDS RS(255,223) with interleaving 5, dual basis symbols, field polynomial `0x187` and first root
  `α^(11*112)`. About 3.5 % of the codewords of `proba2_dump.cadu` have errors, mostly 1-2 symbols. `rs.py` corrects them
  and `iter_packets(path, correct_errors=True)` uses it, but satdump doesn't, so it is off.

### Packet types

- **Image packets**, up to 65536 bytes. An image is 12 of them in 266 of the 324 images (the others lost packets), all sharing the time stamp in payload bytes 0-5. Each
  starts with a 14 byte secondary header, the very first one is followed by a 78 byte metadata block, and then the
  progressive JPEG data follows (`FF D8` at offset 92 of the first packet).
- **Housekeeping packets**, 22-24 bytes, evenly spaced in time. Layout: `10 05 01 00`, a time stamp, the constant `0x012c`
  (300) and 2 bytes that look random. These carry a garbage day count in bytes 0-1 and are ignored when counting images.

### Metadata of the first image packet

| Offset | Size | Meaning | Status |
|---|---|---|---|
| 0-5 | 6 | Image time stamp: `0x010d` days + 32 bit milliseconds (satdump adds 18630 days and 4 h to this) | known, taken from satdump |
| 6 | 1 | Fraction of the time stamp (the only byte of 6-7 that changes) | guessed, byte 7 is always 0 |
| 8-13 | 6 | A second, earlier time stamp, about 3.5 s before the first one (assuming milliseconds) | unknown, exposure start? |
| 14-21 | 8 | The image time stamp again, with a `00` where the first copy has `01` | known |
| 22 | 1 | `0x02` in every image | unknown |
| 23-30 | 8 | Float64 which is only ever `1.0`, `0.0` or `±0.7071067819` | see below |
| 31-47 | 17 | Zeros, bar a constant `01` at 47 | unknown, probably unused |
| 48-51 | 4 | `10` | unknown |
| 52-59 | 8 | `1023`, `1023` | window of the image, 1024 x 1024 pixels |
| 60-74 | 15 | Constants: `3b 00 00 00 04 0b 01 01 0e 10 00 0a 00 01 00` (includes `0x0e10` = 3600 and `0x0a` = 10) | unknown |
| 75 | 1 | One of 100, 101, 103, 110, 120, 130 | unknown, constant over runs of consecutive images |
| 76-79 | 4 | About 733 000-750 000, changes from image to image | unknown, see below |
| 80-87 | 8 | Zeros | unknown, probably unused |
| 88-91 | 4 | The value at 76-79 minus 4 to 19 | unknown |

Of all 78 bytes, only the time stamps, offset 23-30, 75, 76-79 and 88-91 differ from image to image. Everything else is
the same in all but two or three images, which have random bytes everywhere (bit errors, the frames are not corrected).
There is no room for further per-image values.

**Orientation (23-30).** On `proba2_dump.cadu` the value matches the rotation of the sun measured by hand on all 12 images:

| Value | Sun rotated clockwise relative to the first image |
|---|---|
| -0.7071 | 0 |
| 1.0 | 90 |
| 0.7071 | 180 |
| 0.0 | 270 |

This is cos(psi/2) of a roll psi in 90 degree steps (with psi = rotation - 90), which would make it the w component of a
quaternion. That is a fit to four values, and the other dumps have not been measured by hand. Over all dumps the
flag predicts 98 images at 0, 83 at 90, 54 at 180 and 89 at 270 degrees.

**76-79 and 88-91.** They move together, between 726 256 and 769 072, and the first is always 4-19 larger than the
second. Over the 258 complete images they don't correlate with the mean brightness (r = 0.06), the area of the sun disk
(r = 0.21) or either coordinate of the disk center (|r| <= 0.03; disk found by thresholding a 128 x 128 downsample, so
that is rough). Whatever they are, they are not the sun position.

### Where the sun center, attitude and position are not

The sun center was the obvious candidate for an onboard autonomy parameter, but the metadata above has no field which
could hold it: the center of the disk moves between images (roughly 470-530 in x and 450-650 in y, measured roughly on the 258 complete
images), while the only bytes which change are the ones listed, and none of them correlates with it. No position or
attitude vector fits either. The rest of the metadata is constant.
The F4/F6 text records in APID 18 are the only position-like numbers found so far (see `find_apids.ipynb`).

Still to look at: the secondary headers of the later image packets, the housekeeping packets, the other APIDs (2 and 22 are done, see
below) and the JPEG stream itself.

## APID 2 (housekeeping telemetry)

Explored in `APID_2.ipynb`, on the same 13 dumps. 18 399 packets, 12 280 on virtual channel 2, 6 113 on channel 1 and 6 on channel 5.
Offsets here are bytes into the whole packet, including the 6 byte primary header.

- **Almost all PUS housekeeping reports.** Byte 6 is `0x10` (PUS version 1), 7 is the service and 8 the subtype. 18 064 packets are
  service 3 / subtype 25 (housekeeping parameter report), 141 + 139 are service 1 / subtype 1 and 7 (telecommand acceptance and
  completion). The other 43 (service, subtype) pairs are singletons: packets reassembled across lost frames, with garbage bytes and
  absurd lengths.
- **Layout of a report:** bytes 9-10 are `00 01`, 11-14 a 32 bit tick counter, 15-16 a 16 bit fraction, 17-20 zeros, 21 a structure id
  (SID), and the data from 22 on. `(length, SID)` identifies one fixed layout: 123 of them, 44 with at least 50 reports. The biggest
  are 104 bytes / SID 30 (2 101 reports), 61 / 55 (2 042), 292 / 100 (1 199) and 132 / 0 (900).
- **Time:** the counter grows by a fixed step per layout, 2 ticks for 292 / 100, 5 for 132 / 0, 10 for 61 / 55 and 104 / 30, up to
  250 for the slow ones. One tick is one second (see APID 18: it runs together with the GPS time). So the reports come every 2 s to
  about 4 minutes. A first calibration against the SWAP timestamps gave 0.256 s per tick; that was wrong, the SWAP timestamps are what
  runs slow (see APID 18).
- **Losses:** 97.8 % of the sequence count steps are 1.
- **LYRA:** no packet contains the string "LYRA", and no column looks like a four channel science stream. The reports are at best every
  2 s, which rules out 20 Hz channel data. If LYRA is in here it is housekeeping (temperatures, voltages). Several layouts have groups
  of 4-7 float columns whose changes are correlated (for example offsets 62-90 of 292 / 1), which would fit, but nothing confirms it.
- **Attitude:** no group of 4 (quaternion) or 3 (direction) big-endian float32 with unit norm in any layout. Fixed point or float64
  encodings would not have been found.

Conclusion: APID 2 is spacecraft telemetry and not useful for the sun center. No ICD, so none of the fields are named.

## APID 22 (fast sampled int16 data)

Explored in `APID_22.ipynb`. 11 186 packets, 9 632 on virtual channel 2 and 1 551 on channel 1, 10 of the 13 dumps have some (2 and 6 hold
most). Offsets are bytes into the whole packet.

- **Packet:** the same header as the APID 2 reports. 11 170 packets have secondary header `10 08 8e`, the others are garbage from lost
  frames. Bytes 9-10 are `00 01`, 11-14 the tick counter, 15-16 its fraction, 21-22 the data length (packet length minus 25), the data
  starts at 23 and the last 2 bytes are a CRC-16/CCITT (init `0xFFFF`) over the whole packet, which 99.0 % of the packets pass (the rest
  have bit errors, the dumps are not RS corrected). Lengths vary from 59 to 1489 bytes, 215 distinct values.
- **Cadence:** one packet per tick, i.e. per second, in 99.3 % of the steps, sequence counts to match.
- **Chunks:** the data field is cut from self-describing chunks of 154 or 162 bytes, found by their header: size (word 0),
  index (1), a type word (2), a 16 bit counter that grows by about 14 per chunk (3), a source id 49 or 50 (4), flags `0x..05` / `0x..0D` (5),
  then 70 (154 byte chunks) or 74 (162 byte chunks) big-endian int16 samples and a CRC-16/CCITT of the chunk, which 100 % of the found
  chunks pass. Chunks are not aligned with the packets, about half of the packets start at offset 0.
- **162 byte chunks** (32 513 found): one continuous stream of 8 interleaved channels, 592 samples/s, so 74 per channel. The stream
  continues across chunks without a gap (headers and CRCs are not part of it): 74 mod 8 = 2, so the channel of the first sample moves back
  by 2 per chunk, which is what the data does. Index word 1 counts 0-13 and shows missing chunks. In the longest gap free stretch
  (444 frames, 6 s) all 8 channels sit at 0 +- 300 and then ramp up together over about 30 frames, one channel up to 29 000 and the others to
  about 10 000, and fall back at once; the pulse repeats every 129-130 frames (1.75 s).
- **154 byte chunks** (11 329 found, index 0 and 1 in pairs): one channel, each chunk a linear ramp (for example -550 to -200) that
  restarts in the next chunk, a sawtooth. 140 samples/s.
- **LYRA:** possible, not confirmed. The rate (74 Hz per channel) is below LYRA's 100 Hz, and LYRA has two calibration LEDs per detector,
  which a cycled ramp pulse could be. But 8 channels (LYRA has 4 per unit, 3 units) that are dark between pulses are not a solar irradiance.
  A LYRA file of the same dates or the ICD is needed to decide.
- **Sun position / attitude:** nothing in it.

## APID 18 (GPS receiver text)

Explored in `APID_18.ipynb`. 5 025 packets, 4 466 on virtual channel 2. Same packet header as APID 2 and 22, secondary header `10 08 89`.

- **Framing:** data from byte 23 is STX (`0x02`), a text, 2 hex digits holding the XOR of the text bytes, ETX (`0x03`), and the packet CRC-16
  at the end. 99.5 % of the packets pass both checks. Other packets in this APID are PUS telecommand verification reports (24 bytes,
  `10 01 01` and `10 01 07`) and 23 byte event reports (`10 05 01`).
- **Records:** `F4` (44 chars, 2 448 good), `F6` (597 chars, 2 443) and `F9` (one line, "Phoenix s/w version D09Ep (Proba-2 171204)", 17
  of them). F4 and F6 alternate every 10 ticks. The text is a GPS receiver's (Phoenix is a GPS receiver family).
- **F4:** `F4`, a digit, a 4 digit week and a second of the week (8 chars), a field `18` + week, a signed counter, a slow counter (1150-1193), a
  constant `510` and a second time of the week which stays the same for several records and then jumps (it is 2 s to 12 h older than the
  first one).
- **F6:** `F6`, a digit, the week and second of the week, 6 chars (`000718`, `747218`), a float (-0.014 to 0.074, clock-like) and 12 channel
  blocks. A block is: satellite number (0 = unused), a range in metres (19 to 53 million; the receiver clock offset is in it), a carrier
  phase in cycles, a range rate in m/s (+-8 600), a signal strength (30 to 46) and a status (`00`, `01`, `03`, `07`, `0F`, `1F`). Only
  `1F` blocks have data (3 052 of them).
- **It is a GPS clock and the units check out:** the change of the phase is 5.2550 times the change of the range between two records of the
  same satellite (1 / L1 wavelength = 5.2550), and the change of the range divided by the range rate is 10.00 s, the 10 ticks between the
  records.
- **Time:** the week and second give GPS dates in August to October 2025 (the 13 dumps span about 2 months), and the GPS time runs at
  1.0000 s per tick across all files. So the tick counter of APIDs 2, 22 and 18 is in seconds. The SWAP times of `extract_images.py`
  (2021-10-31 to 2021-11-16) advance only 0.256 s per GPS second between the files, so they are wrong in date and scale (suspects:
  satdump's day offset 18630 and the unit of the 32 bit field). The image file names are affected.
- **Position:** none. F4 and F6 have time and raw satellite measurements only; a position needs the satellite ephemerides, which are not in
  these records (they could come from the navigation message elsewhere, or from a public ephemeris for these dates).
- **LYRA / Sun / attitude:** nothing.

Conclusion: APID 18 is decodable GPS receiver telemetry. It gives the real date and an accurate clock for all other APIDs, but no spacecraft position.

## APID 0 (time packets and debris)

Explored in `APID_0.ipynb`. Only 27 packets of the 271 472 in the dumps.

- **Virtual channels:** 0, 1, 2, 3 and 7 are real (97.7-100 % of their frames have the same header `0x1E51` bar the channel id, 7 is the idle
  channel). 4, 5 and 6 have 129 frames in all, almost none with that header: frames with bit errors that got a wrong channel id.
- **15 packets on virtual channel 5 and 1 on channel 2** (7 bytes, one 2 164 bytes, sequence flags 0, zeros or `0x80` every 9th byte): debris of
  those frames, not packets.
- **1 packet of 129 bytes on virtual channel 1** that starts like a housekeeping report: it is an APID 2 report (structure 129 bytes / SID 71)
  with bit errors in its header (APID field 0, garbage marker and tick, CRC fails), the data from byte 32 reads as sensible float32.
- **10 time packets**, 15 bytes, on virtual channel 0, one in each of 10 dumps, sequence flags 3: payload `08 2f`, a 4 byte tick counter, a 2 byte
  fraction (/ 65536), `00`. No secondary header and no CRC. The tick is the counter of the other APIDs and lies inside the time span of
  its dump, so with the GPS offset from APID 18 they are dated 2025-08-08 to 2025-10-10. The sequence count goes up by one per packet, and
  against the tick it gives a period of 367.6 s (about 6 min 7.6 s, between 366.7 and 368.5 s per gap, jitter of the packet time).
- **LYRA / Sun / attitude:** nothing.

Conclusion: APID 0 is a time reference sent every 6 minutes, nothing else in it is real.

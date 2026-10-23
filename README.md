# Experimental environment
Extracting the scientific and calibration data of PROBA-2 from the CADU dumps in `cadus/`.

- **LYRA: found.** It is APID 21 (see below).
- **Other science data to find:** DSLP (Dual segmented Langmuir probe), TPMU (Thermal plasma measurement unit). They should be in the dumps, but they may be degraded and off at times. TPMU (along with DSLP, maybe) should power on at 14:01:03 UTC and off at 16:33:04 during weekdays only. Other reports suggest that it is 210 minutes per weekday, starting at 14:00 UTC. This may be a clue to uncovering the data, since it is very likely to be absent during the weekends.
- **Next big target: the SWAP pixel offset values**, needed to calibrate the images properly. The metadata of the image packets has no field for
  them (see "Where the sun center, attitude and position are not").

TODO: APID 22 (8 channels of int16 at 592 Hz, a pulse every 1.75 s) is not LYRA. Check it, and the groups of float32 columns in APID 2, against DSLP/TPMU, start with the date.
TODO: Look for the SWAP offsets in the APIDs not explored yet, in the later image packets and in the JPEG stream.
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

## APID 21 (LYRA raw data)

Explored in `APID_21.ipynb`. 697 intact packets, all on virtual channel 2, in three dumps only: 10 (261), 8 (221) and 6 (215), plus 40 damaged ones in
the eclipse dump. Offsets are bytes into the whole packet, including the 6 byte primary header, and all numbers are big-endian. The packets are
cut from the frames like the others (`cadu.py`).

### Packet

- **0-5, primary header:** packet id `0x0815` (version 0, secondary header, APID 21), sequence control, length.
- **6-8:** secondary header `10 08 8b`. **9-10:** `00 01`. **11-14:** tick counter (one tick is one second, the same counter as APIDs 2, 18 and 22) and
  **15-16:** its fraction (/ 65536). **17-18:** zeros.
- **19-21 and 23-25:** two 3 byte sample counters. The second is always the first + 1000, and the next packet starts 2 after the last one. Against
  the tick the counter runs at 20 units per second: it is a 20 Hz sample counter, and a packet covers 1002 of its units.
- **26-93:** constant. Zeros, except `4A` at 33, `1F 9A` at 76-77 and `02` at 82 and 91 (not understood). **40-41** is the kind of record: 0 in 693 packets,
  `0x195A` in 4 (see below).
- **94-95:** length of the compressed block (packet length - 98). **96 to the last 2 bytes:** the compressed block. **Last 2 bytes:** CRC-16/CCITT over the
  whole packet (99.9 % pass).
- **Cadence:** one packet every 50 s (ticks step 50, sometimes 51), about 1.6 kB each.

### Compressed block

It starts with `1F 9D 8C`, the magic of the Unix `compress` program (block mode, 12 bit codes), so the data is LZW compressed on board. The codes come in
groups of 8, as `compress` writes them; the decoder is in the notebook and 696 of 697 packets decode (the one that does not has a bad CRC too).

### Decompressed block

8071 bytes (8082 in 6 packets, which have one more time mark). Offsets in the block:

- **0-6011:** 4 channels of 501 signed 24 bit numbers: a first value, then 500 first differences (the lag 1 autocorrelation of the differences is -0.5,
  which is what white noise differenced looks like). One sample every 2 counter units, i.e. 10 per second per channel. Channel *n* is LYRA channel *n* + 1:
  Lyman alpha, Herzberg, aluminium filter, zirconium filter.
- **6012-6015:** the first counter again.
- **6016-8015:** 500 words of 4 bytes, all `2` in normal packets (a flag per sample? not understood).
- **8016-8070:** 5 time marks of 7 bytes (tick, fraction, `00`), then 5 counters of 4 bytes. The marks are 10 s and 200 counts apart, which ties the counter
  to the clock: the time of a sample is the mark time + (its counter - the mark's counter) / 20. The first mark is 40-51 s before the packet's tick.

### Identification as LYRA

Against LYRA level 1 (raw, in kHz) for the dumps 10, 8 and 6 (2025-08-14, 08-17, 08-19). Level 1 is not in sunpy's `Fido`; the files are at
`https://proba2.sidc.be/lyra/data/eng/YYYY/MM/DD/lyra_YYYYMMDD-000000_lev1_std.fits`.

- **Scale:** the APID 21 counts are 50 times the rate in kHz, a count in 50 ms, with no fitted scale: slope 50.0 (Zr, r = 0.996-0.998) and 50.1-50.2
  (Al, r = 0.98-0.99). The UV channels vary by only 1-2 counts (r = 0.86-0.97). The mean difference is -0.09, -0.07, -0.08 and -0.23 counts and the
  residual 1.2, 0.65, 0.4 and 1.6 counts rms, in every dump.
- **Features:** the Herzberg spikes, the Zr dips, a flare in dump 8 (Al +9, Zr +80 counts) and the start of one in dump 6 are in both, at the same
  time and with the same size. So the "flat levels" of 435, 330, 360 and 1100 counts are real LYRA counts, not dark values.
- **Clock:** the APID 21 sample times (with the offset of APID 18) are ahead of UTC by 21.11 s (08-14), 18.51 s (08-17) and 16.38 s (08-19), found as sharp
  minima of the residual against level 1: the 18 s of leap seconds plus a drift of about 0.9 s per day of the on-board tick clock. So times from one
  offset are only good to a few seconds.
- **Not an exact copy** of the level 1 `std` file (residual 0.4-1.6 counts rms): another LYRA unit, or another instant of the 50 ms. Not tested; the
  level 1 directory also has `bst`, `bca`, `bre`, `cal`, `met` and `rej` files.

### The 4 packets of another kind and the eclipse dump

- **Bytes 40-41 are `0x195A`** in 4 packets (two pairs with equal ticks, in dumps 6 and 8), in the same wrapper: the flags are not all 2 and channel 3
  comes out in the millions, so they do not read like the others. Not decoded.
- **Eclipse dump (`proba2_dump_aug_12_eclipse.cadu`, 2026-08-12):** 40 APID 21 packets, none usable as is: no CRC passes, blocks of about 3.5 kB
  (4 frames) instead of 1.6 kB, and the dump has hundreds of frame counter jumps (80-90 % of its APID 18 and 2 packets pass the CRC).
  Reed-Solomon correction does not help (it drops the uncorrectable frames, and no packet is left whole). Decoded up to the first error,
  18 packets give channel 0 (2 658 samples, about a fifth of the 93 minutes), 3 give channel 1, 2 channel 2 and 1 channel 3. They are flat, with
  only the last packet, at 17:17:15 on the GPS clock, in the eclipse: LYRA's zirconium is at -9.5 % and aluminium at -1.1 % at that minute, and
  channel 0 is 2.3 % lower. It is 7 samples, and the correlation over all recovered samples (+0.78 at lag 0) is not significant (40-150 of 2 161
  lags are higher), so the eclipse adds nothing to the identification. The same dump has 375 APID 20 (SWAP images) packets and no APID 22.

Conclusion: **APID 21 is LYRA raw data**: 4 channels, 10 samples per second of counts per 50 ms, compressed with `compress`, in packets of 50 s with a
sample counter and time marks. It reproduces LYRA level 1 (kHz x 50) on three days with no fitted scale, flares included (notebook, sections 9 and 10).
What is left: the 4 packets of another kind, the flags (all 2), the constant header bytes, the small offset against the `std` file, and the clock drift.

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
- **Float columns:** several layouts have groups of 4-7 float32 columns whose changes are correlated (for example offsets 62-90 of 292 / 1). Not identified;
  a candidate to check against DSLP/TPMU.
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
- **Not LYRA** (that is APID 21). What it is stays open: 8 channels that sit near 0 between pulses are not a solar irradiance, so DSLP/TPMU is a candidate.
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
- **Sun / attitude:** nothing.

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
- **Sun / attitude:** nothing.

Conclusion: APID 0 is a time reference sent every 6 minutes, nothing else in it is real.

## APID 14 (telecommand verification reports)

Explored in `APID_14.ipynb`. 1 738 packets, 1 541 on virtual channel 1 (stored) and 197 on channel 2 (real time), in all 13 dumps but mostly
in 12, 1, 6, 8 and 3.

- **Reports:** 1 735 packets are 24 bytes, PUS service 1 (telecommand verification): `10 01 01` (subtype 1, acceptance, 868) and `10 01 07`
  (subtype 7, completion of execution, 866). The rest (61, 75, 721 bytes) is debris, the 61 byte one an APID 2 report with a corrupted APID.
  Layout: marker `00 01` (9-10), tick (11-14) and fraction (15-16) of the report, `00` (17), the request id of the command (18-21) and a
  CRC-16/CCITT (99.9 % pass).
- **Request id:** packet id `0x180E` (version 0, type 1 = telecommand, secondary header, APID 14) in every report, and the packet sequence
  control (flags 3, 14 bit count). So the telecommands are sent to APID 14 and APID 14 in the downlink is the answer to them.
- **Commands:** 852 distinct commands in the 13 dumps, 849 with both reports. Acceptance and completion are 1.2 ms apart (0.7-1.6 ms for
  90 %). The same report is seen on both virtual channels in 30 cases.
- **Counter:** +1 per command within a burst. Over time it goes 2 981 (2025-08-08) to 5 447 (08-23), then only 61 more until 5 508 (10-09) and
  5 738 (10-10), so there was far less commanding in September; one step back between the dumps of August 19 and 20 (4 639, then 4 499).
  The counter climbs by about 2 800 over the two months, the dumps contain only 852 of the commands.
- **Bursts:** 1 to 3 per dump, 20 to 100 commands one to two seconds apart that last a minute or two (plus a few single commands). In 6 of the
  dumps they are late in the span of the telemetry, in 3 early, and in dump 1 the main burst is 5 hours before it: probably uploads in the
  ground contact. The content of the commands is not in the downlink.
- **Sun / attitude:** nothing. A setting change of an instrument (the integration time of LYRA, for instance) has to be commanded, so this is the only
  record of *when* something was uploaded, not of *what*.

Conclusion: APID 14 is a command log with no measurements.

## APID 8 (command channel and test data stream)

Explored in `APID_8.ipynb`. 600 packets in all 13 dumps (569 on virtual channel 1, 31 on 2), 589 pass the CRC-16. All have the PUS secondary header
`10 <service> <subtype>`; offsets are bytes into the whole packet, big-endian.

- **Kinds:** telecommand verification reports 1/1 accepted (242), 1/7 completed (229) and 1/8 failed (15, error code `0x0050`, all in dump 6), connection
  test 17/2 (5), event 5/2 (13, same body every 1 086 s) and data 8/133 (85).
- **Commands:** request packet id `0x1808`, so telecommands are sent to APID 8, on their own 14 bit counter: 1 168-2 702 in August (about 100 a day), 823-940 in
  October, so it was reset in between (APID 14's was not). The sequence count is shared by all packets, and **a data packet sits between the 1/1 and the 1/7
  of the command that asked for it** (85 of 85).
- **Data packet:** 4 zero bytes, function code (`0x6a` or `0x79`, byte 21), 4 byte length (22-25, exactly the number of data bytes), data, CRC. Answers are 1, 4, 5
  or 257 bytes; the 5 byte ones hold multiples of 16.
- **Poll:** one command (839 on 2025-10-10, 2494 on 2025-08-21) is executed every 2 s with the same sequence count, probably an on-board schedule. Each answer is
  256 bytes of a continuous FIFO-like stream of 16 byte records (tag, 24 bit stamp at 7 200 Hz, 12 bytes). Every 0.5 s a cycle of four records (`7d 3d fd 8d`):
  a free-running counter, a 10 step staircase (stride 350 mod 500) with a reading that is the same in August and October, slow counters and noise. That looks like a
  self-test cycle, not a measurement. Extra records between the cycles are a `0x55`/`0xaa` pattern with one wrong bit and a 32 bit address (memory test log; 326 of 362
  are the same bit).
- **Not** the Sun, SWAP offsets, LYRA, or (judging by timing) TPMU/DSLP.

Conclusion: APID 8 is the command/response link of an on-board unit with a self-test stream; open: what the function codes and the unit are.

## APID 11 (telecommand verification reports, stored procedure)

Explored in `APID_11.ipynb`. 132 packets (50 on virtual channel 1, 81 on 2), in dumps 10, 8, 6 and 1 (plus one stray debris packet in dump 5 and in the eclipse dump).

- **Reports:** 130 are 24 bytes, PUS service 1 (`10 01 01` accepted, `10 01 07` completed), the APID 14 layout; 129 pass the CRC-16. Request packet id `0x180b`, so
  telecommands are sent to APID 11, with their own 14 bit counter (949-1 244 in August, 2 565-2 614 on 2025-10-09). No data of any kind.
- **Commands:** 67 in 4 dumps (08-14, 08-17, 08-19, 10-09), 15-18 each. Unlike APID 14 and 8 the completion is *not* 1 ms after the acceptance: each command is accepted when the
  previous one completes, taking 0 to 49 s. The durations (1, 0, 15, 7, 40-49, 13, 2-8 s) are the same in every dump: the steps of a stored procedure, whose second half (the
  counts +8 to +15, same durations) starts 1.5-5 hours later. Two later commands (count +36 to +156) come 900 and 905 s into the dump.
- Nothing about the Sun, SWAP or LYRA.

Conclusion: APID 11 only says when a subsystem ran a scripted procedure, not what it did.

## APIDs 130-133 (memory dump and housekeeping of one unit)

Explored in `APID_13x.ipynb`. All have the PUS secondary header `10 <service> <subtype>` and bytes 9-10 `00 00`, then the tick (11-14), fraction (15-16) and a CRC-16 at the end. They
come in the same few minutes of every ground contact (weekends too), and in the two dumps that have the APID 8 poll, in the same minutes as it.

- **APID 130** (62 packets, PUS 6/6 memory dump report): four packets of 32 800 bytes = 17 byte header, `00 00 00 00`, memory id `01`, start address (22-25: 0x0, 0x8000, 0x10000, 0x18000),
  length (26-29: 0x8000), data, CRC. 128 KB from address 0, a burst 3 s apart (a second one 180 s later in some dumps). The pieces at 0x8000-0x18000 are byte for byte the same in all dumps
  over 7 weeks (static; 0x8000 is dense, the others 63 % zero with a repeated `00 00 02 08`). The piece at 0 changes between dumps and always fails the CRC: about 64 records of 16 bytes (an increasing
  4 byte time tag) in front of mostly `0xff`. Not decoded.
- **APID 131** (14 packets, PUS 3/25): 48 bytes, once per burst of 130; of its 29 payload bytes only about six change. A status/configuration report.
- **APIDs 132 and 133** (about 99 packets each, PUS 3/25): 292 bytes, same tick and sequence count, every 30 s for about 5 minutes per dump: 65 words `80 <channel> <12 bit value>`, an empty word and a footer
  (`00 00 00 03` / `00 00 00 0c`). 12 bit ADC housekeeping scan; 133 reads mid scale (2048) at 25 positions where 132 reads 2 050-3 100.
- Nothing about the Sun, SWAP, LYRA or plasma (DSLP/TPMU).

Conclusion: the housekeeping and memory dump of one on-board unit (APID 8 is probably its command channel), not science data.

## APIDs 17 and 5 (checked by hand, no notebook)

Nine packets each, in dumps 1, 6, 8, 9 and 10 (APID 17) and 1 and 8 (APID 5), all on virtual channel 1 or 2.

- **APID 17** is intelligible: another command target, like APIDs 14, 8 and 11. Five PUS 1/1 acceptance reports (24 bytes, request packet id `0x1811`, so telecommands go to
  APID 17) and five 22 byte PUS 5/1 event packets, one of each per dump (the event alone in dump 9). CRC-16 passes. No completion reports, no data.
- **APID 5** is not telemetry. The packets start with `10 05`, which is a *telecommand* header (type bit set, APID 5), not a downlink packet id. They are back-to-back telecommand
  packets (115 and 201 bytes, a few truncated, 8 in dumps 1 and 8) with rising sequence counts (`cd8e`, `cd90`, `cf9a`, `db60`, `f254`...) and a time tag (`1b f2 86 e0` in dump 1) that,
  halved, is a tick inside the span of the dump. Probably time-tagged commands being uplinked and caught on virtual channel 2. Their CRC fails, so either another checksum or damage. Not
  looked into further; if the payload is a table (pointing, offsets) it would be worth decoding.

## What can be done with the APID 130 memory dump

Probably not much without documentation. Tested: the dense piece at 0x8000 is not zlib, raw deflate, bzip2 or lzma at the first 64 offsets, has no `1f 9d` (LYRA's `compress`)
magic and no SPARC `save`/`retl`/`nop` patterns; its byte statistics are not random (bytes with 0, 1, 7 or 8 bits set are over-represented), so it is structured, but of
what is unknown. The pieces at 0x10000 and 0x18000 are 63 % zeros with a repeated `00 00 02 08`. The one piece worth effort is the live one at address 0: a log with a growing time tag,
which could date events (the 1 086 s event of APID 8) if its units were pinned down. The static pieces could hold a calibration or pointing table, but nothing says so, and 128 KB from address 0 is a
small slice of the memory map. More progress needs another memory range (new dump requests) or the unit's documentation.
Cheap next try: read the 0x8000 piece as packed 12/14 bit samples (12 bit fields are used in APIDs 132/133) and compare the log time tags with the 1 086 s events.

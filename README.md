# Experimental environment
Extracting the scientific and calibration data of PROBA-2 from the CADU dumps in `cadus/`.

- **SWAP: fully decoded.** APID 20, every byte (see below). The sun center (the "pixel offsets") is not in the
  telemetry; the calibration pipeline fits it to the limb, and so does `extract_images.py --process`.
- **LYRA: found.** It is APID 21 (see below).
- **Other science data to find:** DSLP (Dual segmented Langmuir probe), TPMU (Thermal plasma measurement unit). They should be in the dumps, but they may be degraded and off at times. TPMU (along with DSLP, maybe) should power on at 14:01:03 UTC and off at 16:33:04 during weekdays only. Other reports suggest that it is 210 minutes per weekday, starting at 14:00 UTC. This may be a clue to uncovering the data, since it is very likely to be absent during the weekends.

TODO: APID 22 (8 channels of int16 at 592 Hz, a pulse every 1.75 s) is not LYRA. Check it, and the groups of float32 columns in APID 2, against DSLP/TPMU, start with the date.
TODO: Check if PROBA-V GPS log format matches PROBA-2's to get more out of APID 18
TODO: The SWAP detector temperature (`TEMP1DET`, "SW HK T CF", not in APID 20; maybe in APID 2). The level 1 pipeline
(`p2sw_prep.py`, see "Level 1") needs it for the dark current and takes it from the archive or assumes -0.73 C.

## Usage

```
python extract_images.py cadus -o images            # the JPEGs, named after the UTC of the start of acquisition
python extract_images.py cadus -o images --process  # + images/processed/*.png north up and sun centered, centers.csv
python extract_images.py cadus -o images --level1 --archive-headers swap_archive/lv0_headers --gif
                                                    # + images/level1/*.fits and *.png, calibrated like the archive
python p2sw_prep.py swap_lv0_*.fits -o level1 --png # the archive's level 0 FITS files -> level 1
python process_cadu.py cadus/proba2_dump.cadu --gif --mp4  # one dump -> SWAP_pro/*.png calibrated, + animations
python process_cadu.py cadus/proba2_dump.cadu --raw --gif  # no calibration: raw grey levels, only aligned
```

`--process` turns each image by the commanded roll (90 degree steps, from `LANG_ROT`), fits the limb (in the spirit of the
pipeline's `p2sw_imgcor`: a ring correlation, then a circle through the limb maximum on 720 radii), and shifts the sun to
pixel (511.5, 511.5) with bicubic interpolation. `centers.csv` has the center in the turned image and, as `swxcen`/`swycen`,
in the archive's level 0 convention: on the 281 images with a level 1 file it is within 1 pixel of the archive's
`SWXCEN`/`SWYCEN` for 97 % of them and within 2.1 pixels for all, complete or cut. An image whose fit is off the expected
limb radius is saved turned but not shifted (`fit_ok` False).

## APID 20 (SWAP images)

Explored in `APID_20.ipynb`. The PROBA2 Science Center publishes a level 0 FITS file for every SWAP image
(`https://proba2.sidc.be/swap/level0/YYYY/MM/DD/swap_lv0_YYYYMMDD_HHMMSS.fits`), made from these packets; 294 of our images
were matched to theirs (by compressed size and priority) and every field below checked against the FITS keywords. Offsets
are bytes into the packet payload (after the 6 byte CCSDS primary header), all numbers big-endian.

### From CADU to packet

The dumps are not randomized and the frames are not Reed-Solomon corrected. `cadu.py` does what satdump's PROBA module
does with them.

- A CADU is 1279 bytes: 4 byte sync marker, 1115 byte CCSDS TM transfer frame, 160 bytes of Reed-Solomon parity.
- The frame is 6 byte header (the last two bytes hold the 11 bit first header pointer, 2047 = no packet starts here), 1103
  bytes of data, 4 bytes OCF and a 2 byte CRC-16 (CCITT, init `0xFFFF`) over the 1113 before it. Virtual channel 3 carries
  the SWAP images.
- Packets are cut from the data of consecutive frames of a virtual channel. satdump's demuxer (ported in `cadu.py`) ignores
  the frame counter, so a lost frame is not padded and not noticed: the next frames just continue the open packet. Such a
  packet has the wrong length and fails its CRC (see "Lost packets").
- The parity is a CCSDS RS(255,223) with interleaving 5, dual basis symbols, field polynomial `0x187` and first root
  `α^(11*112)`. About 3.5 % of the codewords of `proba2_dump.cadu` have errors, mostly 1-2 symbols. `rs.py` corrects them
  and `iter_packets(path, correct_errors=True)` uses it, but it gives no more intact images (225 either way): it drops the
  frames it can't correct, and the demuxer then glues the packets around the hole. So it is off.

### Packet kinds

All 4 615 packets have packet id `0x0814` (one has a bit error), sequence flags 3, and end in a CRC-16/CCITT (init `0xFFFF`)
over the rest of the packet.

| Kind | Virtual channel | Length | Payload starts | Count (CRC ok) |
|---|---|---|---|---|
| Image packet | 3 | 65 536, the last of an image less | `01` + time stamp | 4 214 (3 760) |
| PUS 5/1 event report | 1, 2 | 22 | `10 05 01` | 367 (367) |
| PUS 1/1 and 1/7 telecommand verification | 1, 2 | 24 | `10 01 01`, `10 01 07` | 27 (27) |
| Debris | | | | 7 (0) |

- **Event reports** (the "housekeeping packets with 2 random bytes"): `10 05 01`, `00 01`, tick (4), fraction (2), `00`, event id
  `01 2c` (300), CRC. **The 2 random bytes are the CRC.** One is sent 0.12-0.18 ms after every image is taken, every 110 s
  (18 s in the eclipse dump), so they log the image times; only about a fifth of the images they report are in our dumps.
- **Telecommand verification**: the APID 14 layout; the request packet id is `0x1814`, i.e. the commands are sent to APID 20
  (SWAP). One or two per dump, at no fixed time against the images.

### Image packet

| Offset | Size | Content |
|---|---|---|
| 0 | 1 | `01` |
| 1-6 | 6 | End of acquisition: 32 bit tick + 16 bit fraction (/ 65 536). The same in every packet of an image, which identifies it |
| 7 | 1 | `00` |
| 8 to the end - 2 | | First packet: the image header (8-91), then the JPEG from 92. The others: JPEG data from 8 |
| last 2 | 2 | CRC-16/CCITT over the packet |

So every packet carries 65 520 bytes of data from offset 8 (the first one 84 bytes of header and 65 436 of JPEG). From offset 88
of the first packet on, the data is `SIZCOMPI` bytes: the JPEG length (4), the JPEG, and 0-15 bytes of padding to a multiple of
16. An image is ceil((80 + `SIZCOMPI`) / 65 520) packets, 12 for all of ours, with consecutive sequence counts.

satdump (and the old `extract_images.py`) took 14 bytes of header from every packet and dropped only the last byte: that loses 6
JPEG bytes at the start of each following packet and keeps one CRC byte at the end of each. Every one of the 225 intact images
then gives libjpeg a "Corrupt JPEG data" warning, and a few percent of the pixels are off (2.9 % in the notebook's example), in
streaks of one block row; cut right, none does.

### Image header (first packet, offsets 0-91)

The names are the level 0 FITS keywords. "Checked" means the field varies and equals the keyword in all 294 matched images;
"value" means it is constant in our data and equals the keyword's (constant) value, so its place is inferred from the order of
the keywords in the FITS header.

| Offset | Size | Keyword | FITS comment | Our data | Status |
|---|---|---|---|---|---|
| 0 | 1 | | | `01` | framing |
| 1-6 | 6 | `EACQTIME` | OBET end acquisition time | tick + fraction | checked |
| 7 | 1 | | | `00` | framing |
| 8-13 | 6 | `SACQTIME` | OBET start acquisition time | 13.9 s before the end | checked |
| 14 | 1 | | | `00` | framing |
| 15-20 | 6 | `TRANTIME` | OBET transferring time | = `EACQTIME` | checked |
| 21 | 1 | | | `00` | framing |
| 22 | 1 | `OBS_MODE` | sun_cen, fix_off, var_off, cme_track | 2 = Variable off-pointing | value |
| 23-30 | 8 | `LANG_ROT` | commanded large angle rotation component | float64: 0, ±0.7071067819, 1 | checked |
| 31-38 | 8 | `PAV_ROT0` | commanded paving rotation component 0 | float64 0 | value |
| 39-46 | 8 | `PAV_ROT1` | commanded paving rotation component 1 | float64 0 | value |
| 47 | 1 | `CAP_MODE` | (DS,CDS) capture mode | 1 = CDS | value |
| 48-51 | 4 | `EXPTIME` | [s] commanded exposure time | 10 | value |
| 52-53 | 2 | `FIRSTROW` - 1 | first read-out detector row | 0 | value |
| 54-55 | 2 | `LAST_ROW` - 1 | last read-out detector row | 1023 | value |
| 56-57 | 2 | `FIRSTCOL` - 1 | first read-out detector column | 0 | value |
| 58-59 | 2 | `LAST_COL` - 1 | last read-out detector column | 1023 | value |
| 60 | 1 | `PGA_OFFS` | PGA offset | 59 | value |
| 61-63 | 3 | `PGA_GAIN`?, `LED_POW`?, `LED_SEL`? | (FITS: 1, off, a) | `00 00 00` | not pinned down |
| 64 | 1 | `HASBLACK` | HAS detector black level | 4 | value |
| 65 | 1 | `HASOFFST` | HAS detector offset level | 11 | value |
| 66 | 1 | `ARTEFX` | (off,on,cosmic rays) on-board artefact removal | 1 = on | value |
| 67 | 1 | `RECODING` | (off,fixed,adaptive) on-board recoding | 1 = fixed | value |
| 68-69 | 2 | `RECNUM` | recoding upper limit | 3600 | value |
| 70-71 | 2 | `RECBIAS` | recoding lower limit | 10 | value |
| 72-73 | 2 | `COMPRESS`? `REBIN`? | (FITS: jpeg, off) | `00 01` | not pinned down |
| 74 | 1 | `LZWDECOR` | LZW decorrelation | 0 = off | value |
| 75 | 1 | `PN` | on-board priority number | 100-130 | checked |
| 76-79 | 4 | `SIZCOMPI` | size of on-board compressed image | 726 256-769 072 | checked |
| 80-83 | 4 | `NPRESCR` | # preserved pixels - cosmic rays | 0 | value |
| 84-87 | 4 | `NPRESLZW` | # preserved pixels - LZW decorrelation | 0 | value |
| 88-91 | 4 | | JPEG length, SOI to EOI (not in the FITS) | `SIZCOMPI` - 4 - padding | checked against the JPEG |

`HASSTDBY` and `READRDIV` (both 0) have no place of their own; they may be among the zeros of 61-63. The rest of the level 0
header (`DATE-OBS`, attitude `P2_X0`/`P2_Y0`/`P2_ROLL`, the CD matrix and `CROTA`, the orbit, `TEMP1DET`/`TEMP2DET`, `DTPLAR1/2`)
comes from other spacecraft telemetry on the ground. Notes:

- **The earlier reading of 0-5 as `0x010d` days + 32 bit milliseconds was wrong.** Byte 0 is the `01`, bytes 1-4 the tick (the counter
  of APIDs 0, 2, 18, 21 and 22) and 5-6 its fraction. The "milliseconds" were tick x 256 + fraction / 256, which advance 0.256 s
  per second: that was the "SWAP clock runs slow by 0.256" (APID 18).
- **Time stamps.** The archive's OBET counts are these 48 bit stamps / 4 + 2^42, exactly.
- **Detector offsets.** `PGA_OFFS`, `HASBLACK` and `HASOFFST` are the electronic offset settings of the detector; they never
  changed over the two months of dumps.
- **`PN`** (100, 101, 102, 103, 110, 120, 130) comes in runs (130 for a dozen images, 110 / 120 alternating): it ranks the images
  for downlink, and has nothing to do with the image itself.
- **`SIZCOMPI`** is the JPEG length + 4, rounded up to a multiple of 16, in all intact images.

### Roll (`LANG_ROT`)

`LANG_ROT` is sin(roll / 2) of the commanded roll about the line of sight (the vector part of its quaternion), matching `P2_ROLL`
of the archive in every matched image. `np.rot90(jpeg, k)` turns an image north up (ecliptic north; checked against level 1 images,
which are rotated to solar north):

| `LANG_ROT` | `P2_ROLL` | `CROTA1` | k for `np.rot90(jpeg, k)` | Sun as seen, relative to roll 270 (by hand) |
|---|---|---|---|---|
| -0.7071 | 270 | 0 | 0 | 0 |
| 1.0 | 180 | 90 | 1 | turned 90 clockwise |
| 0.7071 | 90 | 180 | 2 | 180 |
| 0.0 | 0 | 270 | 3 | turned 270 clockwise |

k = -(roll + 90) / 90 mod 4, with roll = 2 asin(`LANG_ROT`). The pipeline then rotates by `SOLAR_EP` (ecliptic to solar north,
3-7 degrees) and a fixed SWAP mount offset of -0.3 degrees.

### JPEG stream

Standard libjpeg output, the same in every image: JFIF APP0 (1.1, no thumbnail), one quantization table of 64 ones (quality 100),
SOF2 (progressive, 8 bit, 1024 x 1024, one component), DRI of 128 MCUs (one block row), and 6 scans in libjpeg's default
progression for one component, each with its own optimized Huffman table. **No COM or other APP segments.** After the EOI come
0-15 bytes of padding, leftovers of the buffer (often the bytes of the image before). Nothing else is in the stream.

### Pixel values

The level 0 image is our JPEG flipped up-down (FITS row 1 is the last JPEG row), with each grey level replaced by a detector value:
0 -> 10 DN (`RECBIAS`), 255 -> 3588 DN (just under `RECNUM`), in steps of 1, 3, 5 ... 27 DN per grey level (a piecewise linear
square root, the on-board "fixed recoding", reconstructed at the bin centres). 98.7 % of the pixels sit on that table, the rest
one grey level off (IDCT rounding). The table is `RECODE_TABLE` in `extract_images.py`.

### Time

- `DATE-OBS` of the archive is the start of acquisition (8-13): it follows it to 2 ms, the end to 22 ms.
- The tick is not exactly a GPS second: per dump, GPS minus tick (from the F4 records of APID 18) grows by 1.064 s a day, from
  1 209 613 880.9 s on 2025-08-08 to 896.8 on 08-23 and 947.3 on 10-09 (with a step of -1.2 s between 2025-10-09 and 10-10), so the
  single `GPS_MINUS_TICK` of the notebooks is off by up to ±11 s in August and about 56 s in October.
- `extract_images.py` takes the median GPS minus tick of the dump, plus 1.064 s a day from there, minus 18 leap seconds: the start
  times then agree with `DATE-OBS` to a median of 0.08 s in 13 of the 14 dumps, and 266 of 303 file names are the archive's own.
  `proba2_dump.cadu` is 1.78 s off: its images are older than the clock step above.

### Lost packets

Of the 403 images (distinct time stamps), 329 have their first packet and 225 are complete with every packet passing the CRC. A
missing packet can't be skipped (the JPEG data runs on), and a packet glued together across lost frames (wrong length) is garbage
whose stray markers can wreck the whole decode. Cutting the stream at the first missing or wrong-length packet and ending it with
an EOI works best: the JPEG is progressive, so the first packets already hold the whole image at lower quality (the first ~14 kB
are the DC scan, the first 56 kB the coarse AC). Against the archive's complete versions (25 images with missing packets) this is
as good as gluing the packets together or better, by up to 20 % less residual when the gap is early, and the sun center fits
as well as on complete images. Packets that only fail the CRC (bit errors, right length) are kept: over 85 damaged images that is
better than cutting at them in 12, worse in 1, the same in 71, but once it wrecks the decode, and then `extract_images.py` cuts at
the first bad CRC instead. This saves 328 images (226 complete, 102 cut).

### Sun center

Not in APID 20 and not in level 0 either: `CRPIX1/2 = 512.5` and `CRVAL1/2 = 0` ("level < 1: unknown") in every file. The pipeline
(`p2sw_prep.pro`, `p2sw_imgcor.pro` in SolarSoft, `$SSW/proba2/swap/idl/prep/`) finds it by cross-correlating the median filtered
image with a ring of radius `RSUN_ARC / CDELT1` ± 3 pixels and refining with `p2sw_fit_limb.pro` on the limb maximum, and writes
it to level 1 as `SWXCEN`/`SWYCEN`. Those move by about ±40 pixels (461-545), with means differing by up to 20 pixels between roll
states; the spacecraft yaw and pitch (`P2_X0`/`P2_Y0`, attitude telemetry) explain only part of it. A limb fit on our JPEGs
(`APID_20.ipynb`, section 10, and `extract_images.py --process`) reproduces `SWXCEN`/`SWYCEN` to a median of 0.1-0.15 pixel. The
limb at 17.4 nm is about 2 pixels outside the photospheric radius.

## Level 1 (`p2sw_prep.py`)

`p2sw_prep.py` is a Python port of the SolarSoft pipeline that makes the archive's level 1 files (`swap_prep/`, IDL,
`p2sw_prep.pro` v1.5), with the calibration files of `swap_caldb/`. One function per IDL routine, in the same order:

| Step | Routine | What it does (as the archive runs it) |
|---|---|---|
| 1 | `p2sw_hdrchk` | refuses LAR, LED/dark and off-pointed images; detector temperature (`TEMPDARK`) interpolated in time |
| 2 | `p2sw_pmcdiv` | divides by 2 / 4 the pixels multiplied on board (none: the pixel maps hold only 0 and 1) |
| 3 | `p2sw_pixrep` | puts back pixels from the FITS extension (none) and sets pixels >= 0.95 x 4094 to 4400 (none: max 3588) |
| 4 | `p2sw_drksub` | subtracts the dark current, a cubic in the detector temperature per pixel, x `EXPTIME`; clips at the 1 % level |
| 5 | `p2sw_pmcrep` | replaces the 6 466 unread pixels by their 3 x 3 median |
| 6 | `p2sw_ffcorr` | divides by the flat field |
| 7 | `p2sw_despike` | replaces pixels > 6 robust sigma (MAD, 5 x 5) from their 5 x 5 median by it (~32 000 per image) |
| (8) | `p2sw_psfcor` | PSF deconvolution (stray light), optional band removal: not in the archive, `--psf` |
| 9 | `p2sw_imgcor` | limb fit, then sun centered, pixels squared (y x 1.0036) and solar north up (roll, -0.3 deg mount, `SOLAR_EP`) in one cubic interpolation |
| 10 | `p2sw_expnorm` | DN/s |

Written as int16 with `BSCALE` = 1 / (16 x `EXPTIME`), `BZERO` = 2^15 x `BSCALE`, as the archive does.

**Checked against the archive:** the 10 level 0 files of `swap_archive/lv0` with their level 1 files (`swap_archive/lv1`,
`tests/test_p2sw_prep.py`): sun center, `SWAVINT` and the number of despiked pixels are the same in all 10, 98.5-99.7 % of
the pixels are the same integer and the rest are 1 or 2 steps (0.006-0.013 DN/s) off. That took reproducing IDL's single
precision where it changes the result:

- The dark current is the sum of four terms of ~1e5 that cancel to ~15 DN: its float32 rounding (~0.1 DN) decides which
  pixels the despiking takes. It is evaluated left to right in float32, and the 273.15 of `float(TEMPDARK + 273.15)` is a
  float32 literal (272.4067078 K instead of 272.4067383 K turned 241 despiking decisions in one image).
- `POLY_2D` starts each row exactly and adds the x step in float32: the coordinates drift by up to 0.01 pixel along a row.
  Without it only 58 % of the pixels match.
- `POLY_2D` takes points up to x < 1024 (extrapolating the last half pixel), not x <= 1023.
- `INTERPOLATE` (limb profile) has float32 coordinates and `CURVEFIT` stops when chi-square improves by < 1e-3: both move the
  fitted center by ~1e-5 pixel, enough to round it to another 0.01 now and then.

**On our images** (`extract_images.py --level1`) the JPEG is turned back into the level 0 image (flipped, `RECODE_TABLE`)
and header: `CD`/`CROTA` from `LANG_ROT`, `SOLAR_EP` and `RSUN_ARC` computed from the date (0.001 deg and 0.015 arcsec from
the archive's). Not in our telemetry: the detector temperature and the LAR timing. With `--archive-headers` they come
from the archive's level 0 header of the same image (matched by `DATE-OBS`, 1 s); otherwise the temperature is -0.73 C
(the median of the 281 archive files, which range from -1.3 to +1.5 C; 1 C is ~1.8 DN of dark per pixel) and images
blurred by a LAR are not caught. All 14 dumps with `--archive-headers swap_archive/lv0_headers`: 318 level 1 images of
328, the other 10 refused as taken during a LAR. On the 248 that have an archive level 1 file, the sun center is within
0.01 pixel (median) of `SWXCEN`/`SWYCEN`, within 0.5 pixel for 247, and `SWAVINT` within 0.25 % for 90 %; against the 10
archive level 1 images, the median pixel is the same and 99 % are within 4.6 % (the JPEG's loss).

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
  about 4 minutes. A first calibration against the SWAP timestamps gave 0.256 s per tick; that was wrong, the SWAP timestamps were
  misread (see APID 20).
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
  1.0000 s per tick across all files (to 4 digits: per dump, GPS minus tick grows by 1.064 s a day, see APID 20, "Time"). So the tick
  counter of APIDs 2, 22 and 18 is in seconds. The SWAP times the old `extract_images.py` took from satdump (2021-10-31 to
  2021-11-16) advanced only 0.256 s per GPS second between the files: they were a misreading of the SWAP time stamp, which is the
  same tick (see APID 20). Fixed; the image files are now named after the UTC of the start of acquisition.
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

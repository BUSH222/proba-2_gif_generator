# SWAP telemetry and calibration notes

Copied from the README of the `experimental` branch (sections "APID 20" and "Level 1"). Paths such as
`APID_20.ipynb`, `cadus/` and `extract_images.py --process` refer to that branch.

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


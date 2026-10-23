# From a `.cadu` dump to a tinted, sun-centered SWAP image

This document follows one SWAP image through the whole program, byte by byte and pixel by pixel. It starts with the raw
radio frames that SatDump records and ends with the coloured PNG and GIF frame. Every step names the function that does
it. The packet format itself, with how each field was verified against the archive, is in
[swap_pipeline/NOTES.md](swap_pipeline/NOTES.md).

```
.cadu file
  │  1. frames            cadu.iter_frames          1279-byte CADU → 1115-byte transfer frame
  │  2. packets           cadu.Demuxer              frames of one virtual channel → CCSDS source packets
  │  3. sort by APID      extract_images.collect    APID 20 → image packets, APID 18 → GPS clock records
  │  4. clock             extract_images.Clock      on-board tick → UTC
  │  5. image assembly    extract_images.assemble   12 packets → image header + JPEG
  │  6. JPEG decode       extract_images.decode     JPEG → 1024×1024 grey levels 0-255
  │  7. level 0           extract_images.level0     grey levels → detector DN + FITS header
  │  8. calibration       p2sw_prep.prep            dark, flat, bad pixels, despiking
  │  9. sun center        p2sw_prep.find_sun_center ring correlation + limb fit
  │ 10. geometry          p2sw_prep.p2sw_imgcor     center, square pixels, solar north up
  │ 11. DN/s              p2sw_prep.p2sw_expnorm    ÷ exposure time
  │ 12. tone + sharpening p2sw_prep.helioviewer_bytes  DN/s → grey 0-255, Helioviewer look
  │ 13. stretch, rotation render.apply_stretch / rotate (both off by default)
  └ 14. colour + files    render.tint_palette / save_png / save_gif / save_mp4
```

`main.py` drives it: steps 1-6 run in the main process, and steps 7-12 run in a process pool, one image per job
(`Level1Job`, `make_level1`). Steps 13-14 run in the main process again, because the optional stretch needs every frame
at once.

Conventions: all multi-byte numbers in the telemetry are big-endian. In the image arrays, `image[row, column]` follows
the FITS convention, where row 0 is the bottom (south) of the picture. Pixel coordinates `(x, y)` are (column, row),
0-based. A displayed image is the array flipped up-down (`np.flipud`).

---

## 1. CADU → transfer frames (`cadu.iter_frames`)

A `.cadu` file is a plain concatenation of 1279-byte CADUs (channel access data units):

| Bytes | Size | Content |
|---|---|---|
| 0-3 | 4 | attached sync marker |
| 4-1118 | 1115 | CCSDS TM transfer frame |
| 1119-1278 | 160 | Reed-Solomon RS(255,223) parity, interleave 5 |

The dumps are neither randomized nor pre-corrected. The program reads the file in steps of 1279 bytes and keeps bytes
4-1118 of each, the transfer frame.

`iter_frames(path, correct_errors=True)` can also run the Reed-Solomon decoder (`rs.py`: dual-basis symbols, field
polynomial `0x187`, first root α^(11·112), Berlekamp-Massey, Chien search, Forney). It is **off** in the pipeline.
About 3.5 % of the codewords have errors, but correcting them saves no extra image: an uncorrectable frame is dropped,
and the demuxer then glues the packets on either side of the hole together, which breaks them anyway. The per-packet
CRC (step 5) catches the damage instead.

## 2. Transfer frames → source packets (`cadu.Demuxer`, `cadu.iter_packets`)

Each 1115-byte frame has:

| Bytes | Content |
|---|---|
| 0-5 | frame header. The virtual channel ID (VCID) is `(byte1 >> 1) & 7`. The first header pointer (FHP) is the low 11 bits of bytes 4-5 |
| 6-1108 | 1103 bytes of packet data (the M_PDU data zone) |
| 1109-1112 | operational control field |
| 1113-1114 | frame CRC |

Packets span frames, so `iter_packets` keeps one `Demuxer` per VCID. Each frame's 1103 data bytes are fed to the demuxer
of its VCID, together with the FHP:

- **FHP = 2047:** no packet starts in this frame, so all 1103 bytes continue the packet already open.
- **FHP = 2046:** an idle frame, ignored.
- **Otherwise:** the bytes before the FHP finish the open packet, and a new 6-byte CCSDS primary header starts at offset
  FHP. Its length field (bytes 4-5, plus 1) gives the size of the packet's data, and that many bytes are collected,
  across as many frames as it takes. Several packets can start in one frame, and a header can itself be split across two
  frames, which the demuxer handles with `in_header` and `header_buffer`.

This is a line-for-line port of SatDump's CCSDS demuxer, quirks included. It never looks at the frame counter, so a lost
frame isn't noticed: the next frame's bytes just continue the open packet. Such a packet comes out with the **wrong
length** and fails its CRC, which is how step 5 recognizes it.

Each finished packet is yielded as `(vcid, packet bytes)`. For each packet:

- APID = bits 0-10 of bytes 0-1 (`packet_apid`).
- Sequence count = 14 bits of bytes 2-3 (`packet_sequence_count`).

## 3. Sorting packets (`extract_images.collect`)

For every dump, two APIDs are kept:

- **APID 20, the SWAP images.** A packet is an image packet when it is longer than 6 + 92 bytes and its payload (the
  packet after its 6-byte header) starts with `0x01`. Packets with `0x10` there are event and verification reports, and
  are skipped. Payload bytes 1-6 hold the *end of acquisition* time stamp. It is the same in all packets of one image,
  so it is used as the image's key: `images[end_stamp].add(packet)`.

  `Image.add` stores packets by sequence count. If the same count arrives twice, a copy that passes its CRC replaces one
  that doesn't. The CRC is CRC-16/CCITT-FALSE (`binascii.crc_hqx(..., 0xFFFF)`) over the whole packet except its last 2
  bytes, which hold the CRC.
- **APID 18, GPS receiver records.** See step 4.

## 4. On-board time → UTC (`extract_images.gps_record`, `Clock`)

On-board time stamps are 48 bits: a 32-bit *tick* (about one second) plus a 16-bit fraction (1/65536 of a tick). The tick
is not GPS time. It is offset by about 1.2 × 10⁹ s, and the on-board clock gains about 1.064 s per day on GPS. So the
offset is measured separately for each dump, from the GPS receiver's own text records.

An APID 18 packet is a usable GPS record when:

- bytes 6-8 are `10 08 89`
- the CRC is good
- the ASCII text between bytes 24 and −5 starts with `F4` and is 44 characters long

In such a record:

- GPS seconds = week (text 3-6) × 604800 + seconds of week (text 7-14).
- tick = bytes 11-14 + bytes 15-16 / 65536 (the on-board time when the record was stamped).
- The record yields `(tick, GPS seconds − tick)`.

`Clock.from_records` takes the **median** tick and the **median** offset of all records in the dump. Converting a stamp
then works like this:

```
tick   = stamp / 65536
offset = median_offset + 1.064/86400 · (tick − median_tick)     # the clock drift
UTC    = 1980-01-06T00:00:00 + tick + offset − 18 s             # GPS epoch, minus 18 leap seconds
```

A dump without GPS records falls back to the constant `GPS_MINUS_TICK`, which is only right for mid-August 2025, and a
warning is printed. With GPS records, the start times agree with the archive's `DATE-OBS` to a median of 0.08 s.

The file name is the UTC of the **start** of acquisition: image header bytes 8-13, which is the archive's `DATE-OBS`.
It is formatted as `SWAP_YYYYMMDDTHHMMSSZ`. Two images in the same second get `-0`, `-1`, … appended.

## 5. Packets → JPEG (`extract_images.assemble`)

Every image packet has this payload layout:

| Payload offset | Content |
|---|---|
| 0 | `01` |
| 1-6 | end of acquisition (tick + fraction) |
| 7 | `00` |
| 8 … end−2 | data |
| last 2 | CRC-16 |

In the **first packet** of an image, the data begins with an 84-byte image header (payload 8-91), and the JPEG starts at
payload offset 92. `Image.first_packet` finds it as the packet whose bytes at payload offset 92 are the JPEG start
marker `FF D8`. The fields used later are (`ImageHeader.parse`):

| Offset | Field | Used for |
|---|---|---|
| 8-13 | `SACQTIME`, start of acquisition | file name, `DATE-OBS` |
| 22 | `OBS_MODE` | level 0 header |
| 23-30 | `LANG_ROT`, float64 | spacecraft roll, step 7 |
| 31-46 | `PAV_ROT0/1` | off-pointing check, step 8 |
| 47 | `CAP_MODE`: 1 = CDS, 0 = DS | which calibration files to use |
| 48-51 | `EXPTIME` (s) | dark current, DN/s |
| 52-59 | read-out window | calibration area |
| 75 | `PN`, priority | level 0 header |
| 76-79 | `SIZCOMPI`: JPEG length (4) + JPEG + 0-15 padding bytes | packet count |
| 88-91 | JPEG length | trimming the padding |

The number of packets in an image is `ceil((80 + SIZCOMPI) / 65520)`, which is 12 for every image seen so far. Their
sequence counts are consecutive from the first packet's, modulo 2¹⁴. All but the last are exactly 65 536 bytes long.

Assembly walks the expected packets in order:

1. If the next packet is missing, or **its length is not the expected one** (it was glued from frames of other
   packets in step 2), stop there.
2. A packet with the right length but a failed CRC (bit errors) is kept, and counted.
3. Append its data: payload bytes 92 to −2 for the first packet, bytes 8 to −2 for the others.

If all packets came through and the stream, cut at the JPEG length, ends in the end marker `FF D9`, that cut is the
JPEG, with the padding removed. Otherwise the stream is cut where it stopped and `FF D9` is appended.

**Why a truncated image is still a whole image:** SWAP's JPEG is *progressive* (SOF2): the first ~14 kB already hold the
DC values of every 8×8 block of the whole image, and later scans add finer and finer detail. An image cut after a lost
packet is therefore the complete picture at lower quality, not the top part of one.

## 6. JPEG → grey levels (`extract_images.decode`)

The JPEG is decoded with Pillow (`LOAD_TRUNCATED_IMAGES = True`, so a cut stream still decodes as far as it goes) into
a 1024 × 1024 `uint8` array. A result with a standard deviation ≤ 1 (all black, from wrecked data) counts as a failure.

If decoding fails with the bit-error packets kept, `main.make_jobs` assembles again with `keep_bit_errors=False`, cutting
at the first bad CRC, and decodes that. Bit errors are usually harmless (over 85 damaged images, keeping them was better
in 12, worse in 1 and the same in 71), but now and then one wrecks the decode.

The on-board JPEG uses quality 100 (a quantization table of all ones), so the only loss is IDCT rounding:
98.7 % of the pixels decode to exactly the grey level the camera produced.

## 7. Grey levels → level 0 image (`extract_images.level0`)

This rebuilds the archive's level 0 FITS file from the JPEG, so that the calibration can run unchanged.

**Pixels.** Before compressing, the camera maps its 10-3600 DN range onto 8 bits with a piecewise-linear square root
(the "fixed recoding"). `RECODE_TABLE` inverts it: grey 0 → 10 DN, grey 255 → 3588 DN, with steps that widen from 1 DN to
27 DN per grey level. The table was measured against the archive's level 0 images. Then:

```
level0 = flipud(RECODE_TABLE[grey]).astype(int16)      # FITS row 1 is the last JPEG row
```

**Header.**

- *From the image header:* `DATE-OBS`, `EXPTIME`, `CAP_MODE`, `OBS_MODE`, the read-out window, `PN`, `SIZCOMPI`,
  `LANG_ROT`, `PAV_ROT0/1`, and the 48-bit time stamps (converted to the archive's OBET counts: stamp / 4 + 2⁴²).
- *Roll:* roll = 2·asin(`LANG_ROT`), one of 0, 90, 180 or 270°. From it:
  - `CROTA1 = CROTA2 = (270 − roll) mod 360`
  - with c = cos `CROTA` and s = sin `CROTA` (each 0 or ±1), the CD matrix: `CD1_1 = 3.1646941·c`, `CD1_2 = −3.1760467·s`,
    `CD2_1 = 3.1646941·s`, `CD2_2 = 3.1760467·c`

  The pixels are not square: 3.1646941″ along x, 3.1760467″ along y.
- *Computed from the date with sunpy/astropy:*
  - `RSUN_ARC` = asin(696 000 km / Earth-Sun distance), in arcseconds.
  - `SOLAR_EP`, the angle from ecliptic north to solar north as seen from Earth. It is computed by projecting a point
    0.1° ecliptic-north of the Sun and the solar north pole into the helioprojective frame, and taking the difference of
    their position angles. It reproduces the archive's value to 0.001°.
- *Not in the dump:* the detector temperature, which is set to −0.73 °C (the median over the archive), and the LAR
  (large-angle rotation) timing. With `--archive-headers`, these come from the archive's level 0 header of the same image
  instead, matched by `DATE-OBS` within 1 s: `TEMP1DET/2DET`, `TTEMP1/2`, `DTPLAR1/2`, `P2_X0/Y0`, `P2_ROLL`, `SOLAR_EP`,
  `RSUN_ARC`.

## 8. Calibration (`p2sw_prep.prep`, steps 1-7 of `p2sw_prep.pro`)

`p2sw_prep.py` is a function-by-function port of the SolarSoft IDL pipeline that makes the archive's level 1 files. It
reproduces IDL's single-precision arithmetic wherever that changes the result: on the archive's own level 0 files,
98.5-99.7 % of the output pixels are bit-identical. Calibration files come from `swap_pipeline/swap_caldb/`. For each kind,
`calibration_file` picks the newest one older than the image.

1. **`p2sw_hdrchk`.** Refuses images that shouldn't become level 1: taken during a LAR (`DTPLAR1 < 180 s` or
   `DTPLAR2 < 5 s + EXPTIME`, known only with archive headers), with the LED on, or off-pointed (`|PAV_ROT| > 0.02`).
   It also computes `TEMPDARK`, the detector temperature interpolated in time between the two samples within 10 minutes
   of the image.
2. **`p2sw_pmcdiv`.** Divides by 2 or 4 the pixels the pixel map marks as multiplied on board. None are in the current map.
3. **`p2sw_pixrep`.** Sets pixels at or above 0.95 × the saturation map to 4400 DN. None reach it: the maximum is 3588.
4. **`p2sw_drksub`.** In DS mode, subtracts the bias frame first (in CDS mode, the one used here, the bias is removed on
   board). Then it subtracts the dark current, modelled per pixel as a cubic in the detector temperature T (kelvin):
   `dark = (c0 + c1·T + c2·T² + c3·T³) × EXPTIME × PGA_GAIN`, with negatives set to 0. The coefficients come from
   `swap_dark_coefs_20220101_000000.save`. The four terms are about 10⁵ each and cancel to about 15 DN, so they are summed in
   float32 in IDL's order, with 273.15 as a float32 literal. Finally the result is clipped from below at the
   lower edge of the 0.1-DN histogram bin holding the darkest 1 % of pixels (or 0, if that is positive).
5. **`p2sw_pmcrep`.** Replaces the ~6 466 pixels the board doesn't read (0 in the pixel map) by their 3×3 median.
6. **`p2sw_ffcorr`.** Divides by the flat field (`swap_flatcds_…`).
7. **`p2sw_despike`.** Inside the read-out window, takes the 5×5 median M of each pixel and the median, over its 5×5
   neighbourhood, of |neighbour − M| (a robust spread, MAD). Pixels further than 6 × 1.483 × MAD from M are replaced by M.
   That's about 32 000 pixels per image: cosmic-ray hits, hot pixels and the JPEG's dark specks.

The archive doesn't run the optional PSF deconvolution (`p2sw_psfcor`), so it is off here too. Our level 1 of the demo
image matches the archive's level 1 file to a median of 0.1 %.

## 9. Finding the center of the sun (`p2sw_prep.find_sun_center`, `p2sw_fit_limb`)

The sun center isn't in the telemetry. The spacecraft points to within about ±40 px, so it is measured on each image:

**a. Coarse: ring correlation.** The expected photospheric radius in pixels is `R = RSUN_ARC / CDELT1`, about 299 px.
A ring mask of radius R ± 3 px, centered in a 1024 × 1024 frame, is cross-correlated with the 5×5-median-filtered image
by FFT: `|ifft(fft(ring) · fft(image))|`, rolled by half the size. The position of the maximum is where a ring of the
solar radius best lies on the bright limb. That gives the starting center (x₀, y₀) and radius R.

**b. Fine: limb fit.** Repeat up to 30 times:

1. **Polar resampling** (`p2sw_polar_map`). Sample the image on 10 000 angles θ × 40 radii ρ around the current center,
   with ρ from 0.97 R to 1.03 R on the first pass and from 0.99 R to 1.01 R after that. The sample points are
   `x = x_c + ρ sin θ`, `y = y_c + ρ cos θ`, computed in float32 as IDL does, with cubic (Keys, a = −0.5) interpolation.
   Points outside the image are 0.
2. **Limb position per angle** (`p2sw_limb_profile`). Along each radius, take the derivative (a central difference)
   and find where the brightness rises fastest. An angle is usable if its whole radial cut lies inside the image and
   the steepest point is not at either end of the band.
3. **Circle fit** (`idl_curvefit`). For a circle of radius R′ whose center is offset by (a₀, a₁) from the current
   center, the distance from the current center to its edge along angle θ is
   `r(θ) = a₀ sin θ + a₁ cos θ + √(R′² − (a₀ cos θ − a₁ sin θ)²)`. This is fitted to the usable limb points by
   Levenberg-Marquardt least squares, a port of IDL's `CURVEFIT`, which stops once χ² improves by less than 0.1 %.
4. Move the center by (a₀, a₁) and set R = R′. Stop when √(a₀² + a₁² + ΔR²) < 1.5 × 10⁻³ px. If 100 or fewer angles
   are usable, give up and keep the current center.

The result is rounded to 0.01 px and written to the header as `SWXCEN`/`SWYCEN` (1-based). On images with an archive
level 1 file, it matches the archive's own center to a median of 0.01 px.

(With `--raw`, `make_aligned` runs the same search on the uncalibrated level 0 DN image, then moves the JPEG's grey
levels with the geometry of step 10, without calibrating them.)

## 10. Centering, square pixels, solar north up (`p2sw_prep.p2sw_imgcor`)

All geometry is one affine transform of homogeneous pixel coordinates (x, y, 1), applied in a single interpolation. With
the image center c = (511.5, 511.5):

```
M = uncenter · R(SOLAR_EP) · T · R(−0.3°) · ROLL · S · center
```

applied right to left:

| Matrix | Effect |
|---|---|
| `center` | moves the origin to the image center c |
| `S = diag(1, CDELT2/CDELT1)` | stretches y by 1.0036, so the pixels become square at 3.1646941″ |
| `ROLL` | the CD matrix ÷ pixel scales: the 90° step that undoes the spacecraft roll (step 7) |
| `R(−0.3°)` | the fixed mounting offset of SWAP on the spacecraft |
| `T` | the translation that moves the transformed sun center (step 9) to c |
| `R(SOLAR_EP)` | from ecliptic north to solar north, 3-7° depending on the date |
| `uncenter` | moves the origin back |

The output image is computed by inverse mapping. For every output pixel, `M⁻¹` gives the input coordinates, and the value
is interpolated there with the Keys cubic kernel (a = −0.5), as IDL's `POLY_2D` does. Two IDL details are kept: each row's
x coordinate is built by adding the step in float32, so it drifts by up to 0.01 px along a row; and points up to x < 1024
(not ≤ 1023) are interpolated. Points that fall outside the input are 0, which gives the thin black wedges at the edges
of the rotated frame.

After this step the sun's center sits at (511.5, 511.5), the pixels are square, and solar north is up (once the array
is flipped for display).

## 11. DN → DN/s (`p2sw_prep.p2sw_expnorm`)

The image is divided by `EXPTIME` (10 s) and negatives are set to 0. It stays float32. Quiet corona is about 1-20 DN/s,
active regions 100-400 DN/s, and flares can reach thousands.

## 12. DN/s → grey levels: the Helioviewer look (`p2sw_prep.helioviewer_bytes`)

The default `--scaling helioviewer` reproduces how the PROBA2 Science Center renders level 1 files into the JPEG 2000
images on Helioviewer (their `fits2img`). It was fitted on one of their images against its level 1 file, and checked on 9
others: all are within 2.4 grey levels rms. The remainder is their lossy compression. The plain square-root quicklook
scaling is about 6 grey levels off. All three operations work on the float image (x = DN/s), before any rounding:

1. **Tone curve.**
   ```
   grey = 18.0335 · max(x, 0)^0.4485 − 0.8329
   ```

   | DN/s | 0 | 1 | 10 | 100 | ≈370 |
   |---|---|---|---|---|---|
   | grey | 0 (clipped) | 17 | 50 | 141 | 255 |

   It is close to a square root, but with a true black at 0.
2. **Mid-scale sharpening** (unsharp mask): `grey += 0.7724 · (grey − Gauss_σ=2.1566(grey))`. This boosts structure 3-20 px
   across, such as loops and active-region detail, by up to 1.35×, and leaves large-scale brightness unchanged.
3. **Fine-scale smoothing:** `grey = Gauss_σ=0.5793(grey)`, which softens pixel-level noise as their images do.

The result is rounded and clipped to 0-255 `uint8`. Only pixels brighter than about 370 DN/s reach full white, typically
0.02 % of an image, which is the same as in Helioviewer's images. The frame is then flipped up-down for display.

The alternatives are `--scaling quicklook` (`13.42 √x + 9.7`, the P2SC's quicklook PNGs and movies: no sharpening, black
lifted to 9.7) and `--scaling p2sw` (`255.9999 · (x^0.25 − 0.5) / 3.7`, the SWAP team's `p2sw_write_png`: more faint
corona).

## 13. Optional stretch and rotation (`render.stretch_limits`, `apply_stretch`, `rotate`)

- **Stretch, `--stretch P`, default 0 (off).** From the combined histogram of *all* frames, find the grey levels
  `low` and `high` below and above which P % of the pixels lie, then map `low → 0` and `high → 255` linearly with a
  lookup table. Using one pair for the whole set keeps an animation from flickering. It clips, though: 0.3 % (the old
  ImageMagick setting) moves white down to about 200 DN/s and flattens active regions, which is why it is off.
- **Rotation, `-r k`.** `np.rot90(frame, −k)`: k quarter-turns clockwise, for a framing other than solar north up.

## 14. Colour and output files (`render.tint_palette`, `save_png`, `save_gif`, `save_mp4`)

The tint is a fixed palette. It maps each of the 256 grey levels to an RGB colour and is computed once. It reproduces
ImageMagick's `-fill #edb103 -tint 100` (what the old version shelled out to), to within 1 level:

```
fill   = (237, 177, 3)                                   # #edb103
luma   = 0.2126·237 + 0.7152·177 + 0.0722·3 = 177.19     # Rec. 709
vector = percent/100 · fill − luma = (59.8, −0.2, −174.2)
weight = 1 − 4 · (g/255 − 0.5)²                          # 0 at black and white, 1 at mid-grey
rgb(g) = clip(round(g + vector · weight), 0, 255)
```

So black stays black, white stays white, and the mid-tones take the gold hue: grey 128 → (188, 128, 0). `--tint` sets
another fill colour. `--color none` uses no palette and keeps the images grey.

The colour is never baked into the pixels. Each PNG is saved as a **palette (P-mode) image**: the pixel values are the
grey levels and the palette is the tint. That is lossless, a third of the size of RGB, and any viewer shows it in colour.
The GIF (`out.gif`, `--fps` default 10) puts every frame on the same palette, so there is no per-frame colour quantization
and no shimmer. The MP4 (`--make_mp4`) looks every frame up in the palette to get RGB and encodes it with H.264
(libx264).

---

### Where the brightness of one pixel comes from, end to end

```
JPEG grey g (0-255)
  → RECODE_TABLE[g]                       10-3588 DN, the camera's value before on-board recoding
  → − dark(T, pixel) × 10 s, ÷ flat(pixel) calibrated DN (despiked, bad pixels filled)
  → cubic interpolation                   moved to its place: sun centered, square pixels, solar north up
  → ÷ 10 s                                DN/s
  → 18.0335 · x^0.4485 − 0.8329           display grey (float)
  → + 0.7724 · (detail above σ 2.16 px)   mid-scale contrast
  → Gauss σ 0.58 px, round, clip          final grey 0-255
  → (stretch, rotation: off by default)
  → palette[grey]                         gold RGB on screen
```

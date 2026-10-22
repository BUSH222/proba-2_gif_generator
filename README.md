# Experimental environment
Trying to extract LYRA and the calibration data from proba-2 cadus

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

Still to look at: the secondary headers of the later image packets, the housekeeping packets, the other APIDs (2 and 22
carry a lot of data) and the JPEG stream itself.

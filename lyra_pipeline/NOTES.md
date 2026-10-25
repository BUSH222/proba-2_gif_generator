# LYRA telemetry and calibration notes

Worked out on the `experimental` branch: `APID_21.ipynb` decodes the packets, `calibrate_lyra.ipynb` matches them with
the PROBA2 Science Center's archive and finds the calibration. Offsets are bytes into the whole packet (6 byte CCSDS
primary header included), all numbers big-endian.

## APID 21: LYRA's samples

Packets of about 1.6 kB on virtual channel 2, one every 50 s per stream, ending in a CRC-16/CCITT (init `0xFFFF`) over
the rest of the packet.

| Offset | Size | Content |
|---|---|---|
| 6-8 | 3 | `10 08 8b` |
| 11-16 | 6 | packet time: tick (4) and fraction of 1/65536 (2) |
| 19-21 | 3 | sample counter of the first sample |
| 23-25 | 3 | sample counter of the last sample, always first + 1000 |
| 40-41 | 2 | 0 in a data packet; `19 5a` in 4 packets of something else (left out) |
| 94-95 | 2 | length of the block |
| 96 to the CRC | | the block, compressed with Unix `compress` (LZW, `1f 9d 8c`: 12 bit codes, block mode) |

The block decompresses to 8071 bytes (8082 with a sixth time mark):

- 4 channels x 501 samples, 3 byte signed: each channel's first sample, then 500 differences from one sample to the
  next. The channels are LYRA's 1 to 4: Lyman alpha, Herzberg, aluminium filter, zirconium filter.
- the first sample counter again (4 bytes), and 500 flag words of 4 bytes, all `2` (meaning unknown).
- 5 or 6 time marks of 7 bytes (tick, fraction, `00`), then a 4 byte sample counter for each: the clock at that counter
  value. They are 10 s and 200 counter units apart.

**The sample counter runs at 20 per second, and a packet holds every second value**: 501 samples 2 apart, 50 s at 10 Hz.
The other half of LYRA's 20 Hz series is in a second stream of packets: one stream has the even counters, the other the
odd ones, the two take alternate sequence counts, and a dump plays one stream back and then the other. Merged, they are
LYRA's full cadence.

- **Time:** a sample's clock is that of the packet's first time mark plus 1/20 s per counter unit; the clock becomes UTC
  with the GPS records of the dump (APID 18), as for SWAP (`swap_pipeline/extract_images.py`, `Clock`). Against the
  archive our times are 40 to 120 ms early, one or two samples.
- **Level 0** is the counts per sample. **Level 1** is round(counts / 49.9894193 ms, 6), in kHz: the archive's
  `lev1_std` bit for bit, for every sample of the three dumps we have (6, 8 and 10 of August 2025). Its `QFACTOR` is 4
  at the samples with a time mark and 3 elsewhere; level 2's `WARNING` is `40000` and `30000` at the same samples.

## APID 2, structure 70: LYRA's temperatures

The housekeeping reports (PUS 3 / 25, see the README's APID 2 section) of 150 bytes with structure id 70, every 30 s,
have the temperatures of LYRA's three units at bytes 47-48, 55-56 and 63-64 (units 1, 2, 3; unsigned). Unit 2 is the
nominal unit, whose temperature sets the dark current. Against the archive's met files on the days of all 13 dumps:

    T (C) = 5.8766009e-08 raw² - 2.9869839e-03 raw + 66.519503     (unit 2)

good to 0.0006 °C rms over raw 44069 to 45365 = 49.0 to 52.0 °C. The unit's temperature goes from about 44 °C in winter
to 56 °C, so outside that range the conversion is an extrapolation.

## Level 2

The archive's level 2 is level 1 after four steps: rescaled to 1 AU, dark current removed, corrected for the
degradation trend, converted to W/m². The numbers aren't published; they were fitted on the archive's level 1 and
level 2 files of the same days, which hold the same samples:

    level 2 = k (d / 1 AU)² (level 1 - D0 - A exp(B T)) + G

with d the Sun distance, T unit 2's temperature, D0 + A exp(B T) the dark current in kHz, k in W/m² per kHz and G, the
degradation correction, in W/m².

| Channel | k | D0 (kHz) | A (kHz) | B (1/C) |
|---|---|---|---|---|
| 1 Lyman alpha | 1.281138e-05 | 7.0008 | 1.2723e-05 | 0.2409 |
| 2 Herzberg | 1.025837e-03 | 6.4550 | 0 | 0 |
| 3 aluminium | 1.417204e-04 | 6.3246 | 2.8206e-04 | 0.1385 |
| 4 zirconium | 2.164532e-05 | 6.2997 | 4.7993e-04 | 0.1687 |

- **Herzberg** follows this to 1e-13 with constant numbers; it has no temperature term. The others follow it to a few
  1e-6 to 4e-5 of the level within a day. Herzberg and Lyman alpha are mostly G: the nominal unit's UV channels have
  degraded to a very low signal (the archive says so too).
- **The constants hold from 2025-03-25 on.** The archive makes each day's file once with the calibration of the time;
  before 2025-03-25 it used other dark current numbers (off by 1e-4 to 1e-3 with these).
- **G** is updated by the archive as it goes: it grows by 0.06 % (Lyman alpha), 0.16 % (aluminium) and 0.5 % (zirconium) of the
  level a year, with small jumps. `lyra_caldb/degradation.csv` has it for the days sampled from the archive (the 5th, 15th and 25th of
  each month since 2025-03-25); a day in between is interpolated, a day after the last takes the last value.
  `lyra.py --refresh` reads the day's own G from the archive (2 MB with HTTP range requests) and adds it to the file.
- **d** is the Earth's distance from the Sun (sunpy); PROBA2's, in the archive's met files, is within 1000 km of it,
  1e-5 in d².

Level 2 from the three dumps alone, against the archive's at the same samples (relative):

| Channel | G from the table | G of the day (`--refresh`) | One count of one sample |
|---|---|---|---|
| Lyman alpha | 4e-5 to 7e-5 | 5e-6 to 7e-5 | 4e-5 |
| Herzberg | 2e-9 | 2e-9 | (no signal) |
| aluminium | 0.9e-4 to 1.6e-4 | 1e-5 to 1.6e-4 | 1.2e-3 |
| zirconium | 3e-4 to 6e-4 | 5e-5 to 6e-4 | 4.6e-4 |

## Not known

- the dark current numbers before 2025-03-25, and the temperature conversion outside 49-52 °C;
- what the 500 flag words (all 2) and the 4 packets of another kind are;
- which unit sends: the numbers are unit 2's (the nominal unit). A dump from a campaign with a backup unit would need
  that unit's numbers.

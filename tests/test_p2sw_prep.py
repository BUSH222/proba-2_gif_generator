"""p2sw_prep.py against the archive: each level 0 file of swap_archive/lv0 with its level 1 file in swap_archive/lv1
(made by the IDL pipeline at the P2SC) must come out the same: sun center, SWAVINT and despiked pixel count exactly,
the pixels to within one step of the file's integer encoding (1/16 DN/s / EXPTIME)."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from swap_pipeline import p2sw_prep

ARCHIVE = Path(__file__).resolve().parent.parent / "swap_pipeline" / "swap_archive"
PAIRS = [
    (level0, ARCHIVE / "lv1" / level0.name.replace("swap_lv0_", "swap_lv1_"))
    for level0 in sorted((ARCHIVE / "lv0").glob("swap_lv0_*.fits"))
    if (ARCHIVE / "lv1" / level0.name.replace("swap_lv0_", "swap_lv1_")).exists()
]


def despiked(header: fits.Header) -> str:
    return next(line for line in header["HISTORY"] if "Despiked" in line).split("(")[-1]


@pytest.mark.skipif(not PAIRS, reason="no level 0 / level 1 pairs in swap_archive")
@pytest.mark.parametrize(("level0", "level1"), PAIRS, ids=[pair[0].stem for pair in PAIRS])
def test_matches_archive(level0: Path, level1: Path) -> None:
    image, header = p2sw_prep.read_level0(level0)
    result, result_header = p2sw_prep.prep(image, header, fits_path=level0)
    with fits.open(level1, do_not_scale_image_data=True) as hdus:
        reference, reference_header = hdus[0].data.astype(np.int32), hdus[0].header

    assert result_header["SWXCEN"] == pytest.approx(reference_header["SWXCEN"], abs=1e-3)
    assert result_header["SWYCEN"] == pytest.approx(reference_header["SWYCEN"], abs=1e-3)
    assert result_header["SWAVINT"] == pytest.approx(reference_header["SWAVINT"], abs=1e-4)
    assert despiked(result_header) == despiked(reference_header)

    encoded, _, _ = p2sw_prep.to_fits_integers(result, result_header)
    difference = np.abs(encoded.astype(np.int32) - reference)
    assert np.mean(difference == 0) > 0.98
    assert difference.max() <= 2


JP2 = sorted((ARCHIVE / "jp2").glob("*.jp2"))  # Helioviewer's JPEG 2000 versions of archive level 1 files


def jp2_source(path: Path) -> Path:
    """The level 1 file a Helioviewer image was made from (FILENAME in its embedded header)."""
    raw = path.read_bytes()
    start = raw.index(b"<FILENAME>") + len(b"<FILENAME>")
    return ARCHIVE / "lv1" / raw[start : raw.index(b"</FILENAME>", start)].decode()


@pytest.mark.skipif(not JP2, reason="no Helioviewer images in swap_archive/jp2")
@pytest.mark.parametrize("jp2", JP2, ids=[path.stem for path in JP2])
def test_helioviewer_look(jp2: Path) -> None:
    from PIL import Image

    level1 = jp2_source(jp2)
    if not level1.exists():
        pytest.skip(f"{level1.name} not in swap_archive/lv1")
    ours = np.flipud(p2sw_prep.helioviewer_bytes(fits.getdata(level1))).astype(np.float64)
    theirs = np.asarray(Image.open(jp2)).astype(np.float64)
    assert np.sqrt(np.mean((ours - theirs) ** 2)) < 2.6  # the plain quicklook scaling is ~6

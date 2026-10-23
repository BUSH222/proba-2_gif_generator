"""CCSDS Reed-Solomon (255,223) decoding, with the dual basis symbol representation, for interleaved codeblocks."""

import numpy as np

N = 255
NSYM = 32
PRIM_POLY = 0x187
FIRST_ROOT = 112  # first consecutive root, as a power of BETA
ROOT_STEP = 11  # BETA = ALPHA ** 11
MAX_ERRORS = NSYM // 2

_DUAL_BASIS = (0x8D, 0xEF, 0xEC, 0x86, 0xFA, 0x99, 0xAF, 0x7B)


def _build_tables() -> tuple[list[int], list[int]]:
    exp = [0] * (2 * N)
    log = [0] * (N + 1)
    x = 1
    for i in range(N):
        exp[i] = exp[i + N] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIM_POLY
    return exp, log


EXP, LOG = _build_tables()
EXP_ARRAY = np.array(EXP, dtype=np.int32)
LOG_ARRAY = np.array(LOG, dtype=np.int32)


def _build_dual_basis_tables() -> tuple[bytes, bytes]:
    """(dual basis -> conventional, conventional -> dual basis) lookup tables for bytes.translate."""
    to_dual = [0] * 256
    for value in range(256):
        for bit in range(8):
            if value & (1 << bit):
                to_dual[value] ^= _DUAL_BASIS[7 - bit]
    to_conventional = [0] * 256
    for value, dual in enumerate(to_dual):
        to_conventional[dual] = value
    return bytes(to_conventional), bytes(to_dual)


TO_CONVENTIONAL, TO_DUAL = _build_dual_basis_tables()


def _mul(a: int, b: int) -> int:
    return EXP[LOG[a] + LOG[b]] if a and b else 0


def _div(a: int, b: int) -> int:
    return EXP[LOG[a] - LOG[b] + N] if a else 0


def _beta_power(power: int) -> int:
    return EXP[(ROOT_STEP * power) % N]


# exponent (of ALPHA) of the term that coefficient i of a codeword contributes to syndrome j, as an array (NSYM, N)
_SYNDROME_EXPONENTS = (ROOT_STEP * (FIRST_ROOT + np.arange(NSYM)[:, None]) * (N - 1 - np.arange(N)[None, :])) % N


def syndromes(codewords: np.ndarray, count: int = NSYM) -> np.ndarray:
    """Syndromes of codewords of shape (..., N) holding conventional basis symbols; zero for a valid codeword."""
    codewords = codewords.astype(np.int32)
    result = np.zeros((*codewords.shape[:-1], count), dtype=np.int32)
    logs = LOG_ARRAY[codewords]
    for j in range(count):
        terms = EXP_ARRAY[(logs + _SYNDROME_EXPONENTS[j]) % N]
        result[..., j] = np.bitwise_xor.reduce(np.where(codewords != 0, terms, 0), axis=-1)
    return result


def correct(codeword: bytearray) -> bool:
    """Correct a codeword (conventional basis) in place. Returns False if it has more than 16 errors."""
    synd = [int(s) for s in syndromes(np.frombuffer(codeword, dtype=np.uint8))]
    if not any(synd):
        return True

    # Berlekamp-Massey: error locator polynomial
    locator = [1]
    previous = [1]
    degree = 0
    shift = 1
    previous_discrepancy = 1
    for n in range(NSYM):
        discrepancy = synd[n]
        for i in range(1, degree + 1):
            discrepancy ^= _mul(locator[i], synd[n - i])
        if discrepancy == 0:
            shift += 1
            continue
        update = locator + [0] * max(0, len(previous) + shift - len(locator))
        factor = _div(discrepancy, previous_discrepancy)
        for i, coefficient in enumerate(previous):
            update[i + shift] ^= _mul(factor, coefficient)
        if 2 * degree <= n:
            previous, previous_discrepancy = locator, discrepancy
            degree = n + 1 - degree
            shift = 1
        else:
            shift += 1
        locator = update
    if degree > MAX_ERRORS:
        return False

    # Chien search over the powers of BETA that can hold an error
    positions = []
    for power in range(N):
        inverse = _beta_power(-power % N)
        value = 0
        for coefficient in reversed(locator[: degree + 1]):
            value = _mul(value, inverse) ^ coefficient
        if value == 0:
            positions.append(power)
    if len(positions) != degree:
        return False

    # Forney: error values
    evaluator = [0] * NSYM
    for i in range(min(len(locator), NSYM)):
        for j in range(NSYM - i):
            evaluator[i + j] ^= _mul(locator[i], synd[j])
    for power in positions:
        locator_root = _beta_power(power)
        inverse = _beta_power(-power % N)
        numerator = 0
        for coefficient in reversed(evaluator):
            numerator = _mul(numerator, inverse) ^ coefficient
        denominator = 0
        for i in range(1, len(locator), 2):  # formal derivative, only the odd terms survive
            denominator ^= _mul(locator[i], EXP[(LOG[inverse] * (i - 1)) % N]) if locator[i] else 0
        if denominator == 0:
            return False
        magnitude = _mul(_div(numerator, denominator), EXP[(LOG[locator_root] * (1 - FIRST_ROOT)) % N])
        codeword[N - 1 - power] ^= magnitude

    return not syndromes(np.frombuffer(codeword, dtype=np.uint8)).any()

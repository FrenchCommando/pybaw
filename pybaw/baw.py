"""Barone-Adesi-Whaley (1987) American option pricing + implied vol.

The hot functions (`baw_price`, `baw_implied_vol`, `vectorized_baw_implied_vol`,
`find_forward_baw`) come from the compiled `baw_core` extension — a nogil scalar loop
with scipy's Cython brentq inside. The extension is required; build it with
`python setup.py build_ext --inplace` or `pip install .`.

Conventions: `sigma` is the Black *total* vol (total standard deviation,
= sigma_annual * sqrt(T)). The pricer works entirely in (D, F, S, sigma) terms —
rT = -ln(D), bT = ln(F/S) — so no explicit T is needed (T=1 convention).

Dividends never appear: the cost-of-carry b is implied from the spot and forward via
carry = F/S = e^{bT}. Feeding a measured spot is what keeps this dividend-free.

`baw_implied_vol` takes the RAW option premium (not the undiscounted price a European
solver takes): its intrinsic is the American max(S-K, 0) / max(K-S, 0). Sentinels:
0.0 at/below intrinsic, NaN on failure / outside no-arbitrage bounds.
"""

import math

import numpy as np

from .baw_core import (
    baw_implied_vol,
    baw_price,
    vectorized_baw_implied_vol,
)
from .baw_core import find_forward_baw as _core_find_forward_baw

__all__ = [
    "baw_implied_vol",
    "baw_price",
    "bsm_price",
    "find_forward_baw",
    "vectorized_baw_implied_vol",
]

_SQRT2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def bsm_price(strike: float, discount: float, forward: float, sigma: float, flag: str) -> float:
    """European (Black) price, kept as a pure reference for tests and premium bounds.

    sigma = total std dev; flag 'c'/'p'.
    """
    if sigma <= 0:
        if flag == "c":
            return max(discount * (forward - strike), 0.0)
        return max(discount * (strike - forward), 0.0)
    d1 = math.log(forward / strike) / sigma + 0.5 * sigma
    d2 = d1 - sigma
    if flag == "c":
        return discount * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return discount * (strike * _norm_cdf(-d2) - forward * _norm_cdf(-d1))


def find_forward_baw(
    call_prices: np.ndarray, put_prices: np.ndarray, strikes: np.ndarray,
    spot: float, discount: float, strike_pct: float = 0.01,
) -> float:
    """Find the forward F where the BAW call IV ≈ put IV across near-ATM strikes.

    European put-call parity (F = K + (C−P)/D) is biased for American options by the
    early-exercise premium, so instead solve for the F that makes the American call and
    put imply the SAME vol near the money — the BAW-consistent forward, using the
    measured spot. call_prices / put_prices are mid premiums, aligned with strikes.
    """
    # the core names these params S/D positionally; bridge the keyword names here
    return float(_core_find_forward_baw(call_prices, put_prices, strikes, spot, discount, strike_pct))

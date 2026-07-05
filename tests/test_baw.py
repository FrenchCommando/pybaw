"""Offline checks for the compiled BAW pricer.

Golden prices were generated from the reference pure-Python implementation at port time
(parity verified over 480 cases); the hardcoded values guard against silent regressions.

Run: pytest, or python -m tests.test_baw
"""

import numpy as np

from pybaw import baw_implied_vol, baw_price, bsm_price, vectorized_baw_implied_vol

# (spot, strike, discount, forward, sigma_total, flag, expected_price)
GOLDEN = [
    (100.0, 90.0, 0.97, 103.0, 0.20, "c", 15.387707899607525),
    (100.0, 110.0, 0.97, 103.0, 0.20, "c", 5.272892897340892),
    (100.0, 90.0, 0.90, 95.0, 0.30, "p", 7.949639917803874),
    (100.0, 110.0, 0.90, 95.0, 0.30, "p", 19.328099084406244),
    (100.0, 100.0, 0.995, 100.5, 0.15, "p", 5.743230737528639),
    (100.0, 100.0, 0.995, 100.5, 0.15, "c", 6.215547839012925),
]


def test_golden_prices() -> None:
    for spot, strike, discount, forward, sigma, flag, expected in GOLDEN:
        got = baw_price(spot, strike, discount, forward, sigma, flag)
        assert abs(got - expected) < 1e-9, (spot, strike, discount, forward, sigma, flag, got, expected)


def test_iv_roundtrip() -> None:
    """baw_implied_vol inverts baw_price."""
    for spot, strike, discount, forward, sigma, flag, _ in GOLDEN:
        price = baw_price(spot, strike, discount, forward, sigma, flag)
        iv = baw_implied_vol(price, spot, strike, discount, forward, flag)
        assert abs(iv - sigma) < 1e-6, (spot, strike, flag, iv, sigma)


def test_zero_dividend_call_equals_european() -> None:
    """A zero-dividend American call (carry*D == 1, i.e. carry = 1/D) is never exercised
    early, so it must equal the European Black price to machine precision."""
    spot, strike, discount, sigma = 100.0, 105.0, 0.94, 0.25
    forward = spot / discount  # carry = F/S = 1/D  ->  carry*D = 1  (zero dividend)
    american = baw_price(spot, strike, discount, forward, sigma, "c")
    european = bsm_price(strike, discount, forward, sigma, "c")
    assert abs(american - european) < 1e-12, (american, european)


def test_american_geq_european() -> None:
    """American >= European for both flags, with and without dividends."""
    spot, sigma = 100.0, 0.25
    for discount in (0.999, 0.95, 0.90):
        for carry in (0.98, 1.0, 1.0 / discount):  # dividend, flat, zero-dividend
            forward = spot * carry
            for strike in (85.0, 100.0, 115.0):
                for flag in ("c", "p"):
                    american = baw_price(spot, strike, discount, forward, sigma, flag)
                    european = bsm_price(strike, discount, forward, sigma, flag)
                    assert american >= european - 1e-10, (discount, carry, strike, flag)


def test_american_put_premium_nonnegative_and_grows_with_rate() -> None:
    """An American put is >= its European value, and the early-exercise premium grows as
    the discount falls (higher r). Forward held at F = S/D (zero dividend) so only r varies."""
    spot, strike, sigma = 100.0, 100.0, 0.20
    prev_premium = -1.0
    for discount in (0.999, 0.98, 0.95, 0.90):
        forward = spot / discount
        american = baw_price(spot, strike, discount, forward, sigma, "p")
        european = bsm_price(strike, discount, forward, sigma, "p")
        premium = american - european
        assert premium >= -1e-12, (discount, premium)
        assert premium > prev_premium - 1e-9, (discount, premium, prev_premium)
        prev_premium = premium


def test_iv_sentinels() -> None:
    """0.0 at/below the American intrinsic (and for non-finite input); NaN for prices
    above any reachable vol."""
    spot, strike, discount, forward = 100.0, 110.0, 0.95, 101.0
    intrinsic_put = strike - spot  # 10
    assert baw_implied_vol(intrinsic_put - 1.0, spot, strike, discount, forward, "p") == 0.0
    assert np.isnan(baw_implied_vol(1e6, spot, strike, discount, forward, "p"))
    assert baw_implied_vol(float("nan"), spot, strike, discount, forward, "p") == 0.0


def test_vectorized_matches_scalar() -> None:
    spots = np.array([100.0, 100.0, 100.0])
    strikes = np.array([90.0, 100.0, 110.0])
    discounts = np.array([0.95, 0.95, 0.95])
    forwards = np.array([101.0, 101.0, 101.0])
    flags = np.array(["p", "p", "c"])
    prices = np.array([baw_price(s, k, d, f, 0.22, fl)
                       for s, k, d, f, fl in zip(spots, strikes, discounts, forwards, flags)])
    ivs = vectorized_baw_implied_vol(prices, spots, strikes, discounts, forwards, flags)
    assert np.allclose(ivs, 0.22, atol=1e-6), ivs


def test_find_forward_recovers_synthetic_forward() -> None:
    """Price a synthetic chain at a known forward/flat vol; find_forward_baw must recover
    that forward (call/put IVs match exactly there, so the objective's minimum sits on it)."""
    from pybaw import find_forward_baw

    spot, discount, sigma = 250.0, 0.97, 0.20
    true_forward = spot * 1.004  # mild dividend/carry gap
    strikes = np.arange(230.0, 271.0, 2.5)
    calls = np.array([baw_price(spot, k, discount, true_forward, sigma, "c") for k in strikes])
    puts = np.array([baw_price(spot, k, discount, true_forward, sigma, "p") for k in strikes])

    found = find_forward_baw(calls, puts, strikes, spot=spot, discount=discount)
    assert abs(found - true_forward) < 0.05, (found, true_forward)


def _run() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all BAW tests passed")


if __name__ == "__main__":
    _run()

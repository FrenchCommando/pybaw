# pybaw

Barone-Adesi–Whaley (1987) American option pricing and implied volatility as a Cython
extension: `nogil` scalar loops with scipy's Cython `brentq` doing the root solves.

## Conventions

- `sigma` is the Black **total** vol (total standard deviation, `sigma_annual * sqrt(T)`).
- The pricer works in `(D, F, S, sigma)` terms — `rT = -ln(D)`, `bT = ln(F/S)` — so no
  explicit `T` appears (T=1 convention).
- Dividends never appear explicitly: the cost of carry is implied via `carry = F/S`.
- `baw_implied_vol` takes the **raw premium** (not undiscounted); its intrinsic is the
  American `max(S-K, 0)` / `max(K-S, 0)`. Sentinels: `0.0` at/below intrinsic, `NaN` on
  failure.

## Install

```
pip install pybaw
```

Prebuilt wheels cover CPython 3.11+ on Linux (x86_64, aarch64), Windows (x86_64),
and macOS (arm64, x86_64). On anything else pip falls back to the sdist, which
needs a C compiler — the compiled extension is required, there is no pure-Python
fallback. Note scipy is needed at **build** time too, not just runtime: the
extension cimports `scipy.optimize.cython_optimize`.

To work on the package itself, either install from a checkout:

```
pip install .
```

or build in place and put the repo root on `PYTHONPATH`:

```
python setup.py build_ext --inplace
```

## API

| Function | Purpose |
|---|---|
| `baw_price(S, K, D, F, sigma, flag)` | American price; `flag` `'c'`/`'p'` |
| `baw_implied_vol(price, S, K, D, F, flag)` | invert a raw premium to total vol |
| `vectorized_baw_implied_vol(prices, S[], K[], D[], F[], flags[])` | batch inversion |
| `find_forward_baw(calls, puts, strikes, spot, discount)` | forward where call IV ≈ put IV near ATM |
| `bsm_price(K, D, F, sigma, flag)` | pure European reference (used by tests/bounds) |

`find_forward_baw` exists because European put-call parity is biased for American
options by the early-exercise premium; solving for the forward that equates call and
put implied vols near the money removes that bias.

## Tests

```
pip install -e .[test]
pytest
```

## License

MIT — see [LICENSE](LICENSE).

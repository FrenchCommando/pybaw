# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython Barone-Adesi-Whaley (1987) American option pricer and IV solver.

Implements baw_price, baw_implied_vol, vectorized_baw_implied_vol, and
find_forward_baw as C-level code (nogil loops, scipy's Cython brentq inside).
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, log, exp, fabs, erfc, floor, pow
from scipy.optimize.cython_optimize._zeros cimport brentq as _scipy_brentq, zeros_full_output

cnp.import_array()


# ══════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════

cdef double _DBL_MAX = 1.7976931348623157e+308
cdef double _DBL_EPS = 2.2204460492503131e-16

cdef double _ONE_OVER_SQRT_TWO = 0.7071067811865475244008443621048490392848359376887
cdef double _ONE_OVER_SQRT_TWO_PI = 0.3989422804014326779399460599343818684758586311649

cdef double _NCDF_ASYM1 = -10.0
cdef double _NCDF_ASYM2 = -1.0 / 1.4901161193847656e-08  # ~ -6.7e7

cdef double _IV_TOL = 1e-10
cdef double _IV_MAX_SIGMA = 10.0
cdef double _BRENTQ_XTOL = 1e-10


# ══════════════════════════════════════════════════════════════
#  Structs for function pointer args
# ══════════════════════════════════════════════════════════════

cdef struct CriticalParams:
    double K
    double D
    double carry
    double sigma
    double q_val

cdef struct IVParams:
    double S
    double K
    double D
    double F
    double target_price
    int q


# ══════════════════════════════════════════════════════════════
#  erfcx (Cody rational Chebyshev — copied from black_core.pyx)
# ══════════════════════════════════════════════════════════════

cdef inline double _erfcx_fixup(double result, double x) noexcept nogil:
    cdef double ysq, d, yy
    if x < 0.0:
        if x < -26.628:
            return 1.79e308
        ysq = floor(x * 16.0) / 16.0
        d = (x - ysq) * (x + ysq)
        yy = exp(ysq * ysq) * exp(d)
        return (yy + yy) - result
    return result


cdef inline double _erfcx(double x) noexcept nogil:
    cdef double y = fabs(x)
    cdef double result, xnum, xden, ysq

    if y <= 0.46875:
        ysq = y * y if y > 1.11e-16 else 0.0
        xnum = 0.185777706184603153 * ysq
        xden = ysq
        xnum = (xnum + 3.1611237438705656) * ysq
        xden = (xden + 23.6012909523441209) * ysq
        xnum = (xnum + 113.864154151050156) * ysq
        xden = (xden + 244.024637934444173) * ysq
        xnum = (xnum + 377.485237685302021) * ysq
        xden = (xden + 1282.61652607737228) * ysq
        result = x * (xnum + 3209.37758913846947) / (xden + 2844.23683343917062)
        return (1.0 - result) * exp(ysq)

    elif y <= 4.0:
        xnum = 2.15311535474403846e-8 * y
        xden = y
        xnum = (xnum + 0.564188496988670089) * y
        xden = (xden + 15.7449261107098347) * y
        xnum = (xnum + 8.88314979438837594) * y
        xden = (xden + 117.693950891312499) * y
        xnum = (xnum + 66.1191906371416295) * y
        xden = (xden + 537.181101862009858) * y
        xnum = (xnum + 298.635138197400131) * y
        xden = (xden + 1621.38957456669019) * y
        xnum = (xnum + 881.95222124176909) * y
        xden = (xden + 3290.79923573345963) * y
        xnum = (xnum + 1712.04761263407058) * y
        xden = (xden + 4362.61909014324716) * y
        xnum = (xnum + 2051.07837782607147) * y
        xden = (xden + 3439.36767414372164) * y
        result = (xnum + 1230.33935479799725) / (xden + 1230.33935480374942)

    else:
        if y >= 26.543:
            if y >= 2.53e307:
                return _erfcx_fixup(0.0, x)
            if y >= 6.71e7:
                return _erfcx_fixup(0.56418958354775628695 / y, x)
        ysq = 1.0 / (y * y)
        xnum = 0.0163153871373020978 * ysq
        xden = ysq
        xnum = (xnum + 0.305326634961232344) * ysq
        xden = (xden + 2.56852019228982242) * ysq
        xnum = (xnum + 0.360344899949804439) * ysq
        xden = (xden + 1.87295284992346047) * ysq
        xnum = (xnum + 0.125781726111229246) * ysq
        xden = (xden + 0.527905102951428412) * ysq
        xnum = (xnum + 0.0160837851487422766) * ysq
        xden = (xden + 0.0605183413124413191) * ysq
        result = ysq * (xnum + 6.58749161529837803e-4) / (xden + 0.00233520497626869185)
        result = (0.56418958354775628695 - result) / y

    return _erfcx_fixup(result, x)


# ══════════════════════════════════════════════════════════════
#  Normal distribution (copied from black_core.pyx)
# ══════════════════════════════════════════════════════════════

cdef inline double _norm_pdf(double x) noexcept nogil:
    return _ONE_OVER_SQRT_TWO_PI * exp(-0.5 * x * x)


cdef inline double _norm_cdf(double z) noexcept nogil:
    cdef double sum_val, zsqr, g, a, lasta, xx, yy
    cdef int i
    if z <= _NCDF_ASYM1:
        sum_val = 1.0
        if z >= _NCDF_ASYM2:
            zsqr = z * z
            i = 1
            g = 1.0
            a = _DBL_MAX
            lasta = a
            xx = <double>(4 * i - 3) / zsqr
            yy = xx * (<double>(4 * i - 1) / zsqr)
            a = g * (xx - yy)
            sum_val -= a
            g *= yy
            i += 1
            a = fabs(a)
            while lasta > a and a >= fabs(sum_val * _DBL_EPS):
                lasta = a
                xx = <double>(4 * i - 3) / zsqr
                yy = xx * (<double>(4 * i - 1) / zsqr)
                a = g * (xx - yy)
                sum_val -= a
                g *= yy
                i += 1
                a = fabs(a)
        return -_norm_pdf(z) * sum_val / z
    return 0.5 * erfc(-z * _ONE_OVER_SQRT_TWO)


# ══════════════════════════════════════════════════════════════
#  BSM (European Black-style, forward-based)
# ══════════════════════════════════════════════════════════════

cdef inline double _bsm_price(double K, double D, double F, double sigma,
                               int q) noexcept nogil:
    cdef double d1, d2, val
    if sigma <= 0:
        if q == 1:
            val = D * (F - K)
            return val if val > 0.0 else 0.0
        else:
            val = D * (K - F)
            return val if val > 0.0 else 0.0
    d1 = log(F / K) / sigma + 0.5 * sigma
    d2 = d1 - sigma
    if q == 1:
        return D * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    else:
        return D * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))


# ══════════════════════════════════════════════════════════════
#  BAW critical-price solvers
# ══════════════════════════════════════════════════════════════

cdef double _call_critical_func(double Sc, void* params) noexcept nogil:
    cdef CriticalParams* p = <CriticalParams*>params
    cdef double Fc = Sc * p.carry
    cdef double d1c = log(Fc / p.K) / p.sigma + 0.5 * p.sigma
    cdef double bsm_c = _bsm_price(p.K, p.D, Fc, p.sigma, 1)
    cdef double coeff = 1.0 - p.D * p.carry * _norm_cdf(d1c)
    return bsm_c + coeff * Sc / p.q_val - (Sc - p.K)


cdef double _put_critical_func(double Sc, void* params) noexcept nogil:
    cdef CriticalParams* p = <CriticalParams*>params
    cdef double Fc = Sc * p.carry
    cdef double d1c = log(Fc / p.K) / p.sigma + 0.5 * p.sigma
    cdef double bsm_p = _bsm_price(p.K, p.D, Fc, p.sigma, -1)
    cdef double coeff = 1.0 - p.D * p.carry * _norm_cdf(-d1c)
    return bsm_p - coeff * Sc / p.q_val - (p.K - Sc)


cdef double _solve_call_critical(CriticalParams* params) noexcept nogil:
    cdef double lo, hi, g_lo
    cdef int i
    cdef bint found = 0
    cdef zeros_full_output fo

    lo = params.K + 1e-10
    g_lo = _call_critical_func(lo, params)
    if g_lo <= 0:
        return lo

    hi = params.K * 2.0
    for i in range(20):
        if _call_critical_func(hi, params) < 0:
            found = 1
            break
        hi *= 2.0

    if not found:
        return hi

    return _scipy_brentq(
        _call_critical_func, lo, hi, <void*>params,
        _BRENTQ_XTOL, _BRENTQ_XTOL, 100, &fo,
    )


cdef double _solve_put_critical(CriticalParams* params) noexcept nogil:
    cdef double lo, hi, g_hi, g_lo
    cdef zeros_full_output fo

    hi = params.K - 1e-10
    g_hi = _put_critical_func(hi, params)
    if g_hi <= 0:
        return hi

    lo = 1e-10
    g_lo = _put_critical_func(lo, params)
    if g_lo >= 0:
        return lo

    return _scipy_brentq(
        _put_critical_func, lo, hi, <void*>params,
        _BRENTQ_XTOL, _BRENTQ_XTOL, 100, &fo,
    )


# ══════════════════════════════════════════════════════════════
#  BAW American option price
# ══════════════════════════════════════════════════════════════

cdef double _baw_price(double S, double K, double D, double F, double sigma,
                       int q) noexcept nogil:
    cdef double bsm, carry, sig2, N_val, M_val, k2
    cdef double disc, sqrt_disc
    cdef double q2, q1, S_star, F_star, d1_star, A2, A1
    cdef CriticalParams params

    if sigma <= 0:
        if q == 1:
            return (S - K) if S > K else 0.0
        else:
            return (K - S) if K > S else 0.0

    bsm = _bsm_price(K, D, F, sigma, q)

    if D >= 1.0:
        return bsm

    carry = F / S
    sig2 = sigma * sigma
    N_val = 2.0 * log(carry) / sig2
    M_val = -2.0 * log(D) / sig2
    k2 = 1.0 - D

    disc = (N_val - 1.0) * (N_val - 1.0) + 4.0 * M_val / k2
    if disc < 0:
        disc = 0.0
    sqrt_disc = sqrt(disc)

    if q == 1:  # call
        if carry * D >= 1.0:
            return bsm

        q2 = (-(N_val - 1.0) + sqrt_disc) / 2.0
        if q2 <= 0:
            return bsm

        params.K = K
        params.D = D
        params.carry = carry
        params.sigma = sigma
        params.q_val = q2

        S_star = _solve_call_critical(&params)

        if S >= S_star:
            return S - K

        F_star = S_star * carry
        d1_star = log(F_star / K) / sigma + 0.5 * sigma
        A2 = (S_star / q2) * (1.0 - D * carry * _norm_cdf(d1_star))

        return bsm + A2 * pow(S / S_star, q2)

    else:  # put
        q1 = (-(N_val - 1.0) - sqrt_disc) / 2.0
        if q1 >= 0:
            return bsm

        params.K = K
        params.D = D
        params.carry = carry
        params.sigma = sigma
        params.q_val = q1

        S_star = _solve_put_critical(&params)

        if S <= S_star:
            return K - S

        F_star = S_star * carry
        d1_star = log(F_star / K) / sigma + 0.5 * sigma
        A1 = -(S_star / q1) * (1.0 - D * carry * _norm_cdf(-d1_star))

        return bsm + A1 * pow(S / S_star, q1)


# ══════════════════════════════════════════════════════════════
#  IV solver
# ══════════════════════════════════════════════════════════════

cdef double _iv_objective(double sigma, void* params) noexcept nogil:
    cdef IVParams* p = <IVParams*>params
    return _baw_price(p.S, p.K, p.D, p.F, sigma, p.q) - p.target_price


cdef double _baw_implied_vol_c(double price, double S, double K, double D,
                               double F, int q) noexcept nogil:
    """Returns IV, or -1.0 sentinel for NaN."""
    cdef double intrinsic, sigma_hi, sigma_lo, f_lo, f_hi
    cdef IVParams params

    if q == 1:
        intrinsic = (S - K) if S > K else 0.0
    else:
        intrinsic = (K - S) if K > S else 0.0

    if price <= intrinsic + _IV_TOL:
        return 0.0

    # Bracket expansion for upper bound
    sigma_hi = 1.0
    while _baw_price(S, K, D, F, sigma_hi, q) < price and sigma_hi < _IV_MAX_SIGMA:
        sigma_hi *= 2.0

    if sigma_hi >= _IV_MAX_SIGMA:
        return -1.0  # sentinel → NaN

    sigma_lo = 1e-8

    params.S = S
    params.K = K
    params.D = D
    params.F = F
    params.q = q
    params.target_price = price

    cdef zeros_full_output fo

    f_lo = _iv_objective(sigma_lo, &params)
    f_hi = _iv_objective(sigma_hi, &params)

    if f_lo * f_hi > 0:
        return -1.0  # sentinel → NaN

    return _scipy_brentq(
        _iv_objective, sigma_lo, sigma_hi, <void*>&params,
        _IV_TOL, _IV_TOL, 200, &fo,
    )


# ══════════════════════════════════════════════════════════════
#  Public Python-callable wrappers
# ══════════════════════════════════════════════════════════════

def baw_price(double S, double K, double D, double F, double sigma, flag):
    """BAW American option price (Cython).

    S spot, K strike, D discount factor, F forward, sigma total std dev.
    flag: 'c' or 'p'.
    """
    cdef int q = 1 if flag == 'c' else -1
    return _baw_price(S, K, D, F, sigma, q)


def baw_implied_vol(double price, double S, double K, double D, double F, flag):
    """BAW implied total std dev (Cython).

    Returns sigma or NaN on failure.
    """
    cdef int q = 1 if flag == 'c' else -1
    cdef double result = _baw_implied_vol_c(price, S, K, D, F, q)
    if result < 0:
        return float('nan')
    return result


def vectorized_baw_implied_vol(prices, S_arr, K_arr, D_arr, F_arr, flags):
    """Batch BAW IV extraction (Cython).

    All inputs are arrays of equal length. flags is array of 'c'/'p'.
    Returns numpy array of implied vols.
    """
    cdef Py_ssize_t n = len(prices)
    cdef Py_ssize_t i
    cdef double result

    cdef double[:] p_view = np.ascontiguousarray(prices, dtype=np.float64)
    cdef double[:] s_view = np.ascontiguousarray(S_arr, dtype=np.float64)
    cdef double[:] k_view = np.ascontiguousarray(K_arr, dtype=np.float64)
    cdef double[:] d_view = np.ascontiguousarray(D_arr, dtype=np.float64)
    cdef double[:] f_view = np.ascontiguousarray(F_arr, dtype=np.float64)

    # Convert flags to int array (must be done with GIL)
    cdef int[:] q_view = np.array(
        [1 if f == 'c' else -1 for f in flags], dtype=np.intc
    )

    out = np.empty(n, dtype=np.float64)
    cdef double[:] out_view = out

    with nogil:
        for i in range(n):
            out_view[i] = _baw_implied_vol_c(
                p_view[i], s_view[i], k_view[i], d_view[i], f_view[i], q_view[i]
            )

    # Replace -1.0 sentinel with NaN
    result_arr = np.asarray(out)
    result_arr[result_arr < 0] = np.nan
    return result_arr


# ══════════════════════════════════════════════════════════════
#  Bounded scalar minimizer (Brent's method, replaces minimize_scalar)
# ══════════════════════════════════════════════════════════════

ctypedef double (*minimize_func_t)(double, void*) noexcept nogil

cdef double _GOLDEN = 0.3819660112501051  # (3 - sqrt(5)) / 2
cdef double _SQRT_EPS = 1.4901161193847656e-08  # sqrt(2.2e-16)


cdef double _fminbound(minimize_func_t f, double xa, double xb, void* args,
                       double xatol, int maxiter) noexcept nogil:
    """Brent's method for bounded scalar minimization."""
    cdef double x, w, v, fx, fw, fv
    cdef double a, b, midpoint, d, e, tol1, tol2
    cdef double r, q, p, u, fu
    cdef int i

    a = xa
    b = xb
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = f(x, args)
    d = 0.0
    e = 0.0

    for i in range(maxiter):
        midpoint = 0.5 * (a + b)
        tol1 = _SQRT_EPS * fabs(x) + xatol / 3.0
        tol2 = 2.0 * tol1

        if fabs(x - midpoint) <= (tol2 - 0.5 * (b - a)):
            return x

        # Try parabolic interpolation
        if fabs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            else:
                q = -q
            r = e
            e = d

            if fabs(p) < fabs(0.5 * q * r) and p > q * (a - x) and p < q * (b - x):
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if x < midpoint else -tol1
            else:
                e = (a if x >= midpoint else b) - x
                d = _GOLDEN * e
        else:
            e = (a if x >= midpoint else b) - x
            d = _GOLDEN * e

        if fabs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d > 0 else -tol1)

        fu = f(u, args)

        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v = w; fv = fw
            w = x; fw = fx
            x = u; fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w; fv = fw
                w = u; fw = fu
            elif fu <= fv or v == x or v == w:
                v = u; fv = fu

    return x


# ══════════════════════════════════════════════════════════════
#  Forward search
# ══════════════════════════════════════════════════════════════

cdef struct ForwardParams:
    double S
    double D
    double* sel_calls
    double* sel_puts
    double* sel_strikes
    Py_ssize_t n


cdef double _fwd_iv_abs_diff(double F_trial, void* params) noexcept nogil:
    """Mean |call_iv - put_iv| at F_trial."""
    cdef ForwardParams* p = <ForwardParams*>params
    cdef Py_ssize_t j
    cdef Py_ssize_t count = 0
    cdef double iv_c, iv_p, total, diff
    total = 0.0
    for j in range(p.n):
        iv_c = _baw_implied_vol_c(p.sel_calls[j], p.S, p.sel_strikes[j],
                                  p.D, F_trial, 1)
        iv_p = _baw_implied_vol_c(p.sel_puts[j], p.S, p.sel_strikes[j],
                                  p.D, F_trial, -1)
        if iv_c <= 0 or iv_p <= 0:
            continue
        diff = iv_c - iv_p
        if diff < 0:
            diff = -diff
        total += diff
        count += 1
    if count == 0:
        return 1e10
    return total / <double>count


def find_forward_baw(call_prices, put_prices, strikes, double S, double D,
                     double strike_pct=0.01):
    """Find forward price F where BAW call IV ≈ put IV across near-ATM strikes."""
    call_prices = np.asarray(call_prices, dtype=np.float64)
    put_prices = np.asarray(put_prices, dtype=np.float64)
    strikes = np.asarray(strikes, dtype=np.float64)

    # Widen window until we have strikes
    for pct in [strike_pct, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]:
        lo = S * (1.0 - pct)
        hi = S * (1.0 + pct)
        mask = (strikes >= lo) & (strikes <= hi)
        if mask.any():
            break
    else:
        mask = np.ones(len(strikes), dtype=bool)

    cdef double[:] c_calls = np.ascontiguousarray(call_prices[mask], dtype=np.float64)
    cdef double[:] c_puts = np.ascontiguousarray(put_prices[mask], dtype=np.float64)
    cdef double[:] c_strikes = np.ascontiguousarray(strikes[mask], dtype=np.float64)

    # Initial F estimate from put-call parity: C - P ≈ D*(F - K), so F ≈ K + (C - P)/D
    cdef Py_ssize_t n_sel = c_calls.shape[0]
    cdef Py_ssize_t j
    cdef double f_est_sum = 0.0
    cdef Py_ssize_t f_est_count = 0
    for j in range(n_sel):
        f_est_sum += c_strikes[j] + (c_calls[j] - c_puts[j]) / D
        f_est_count += 1
    cdef double F_init = f_est_sum / f_est_count if f_est_count > 0 else S

    cdef ForwardParams params
    params.S = S
    params.D = D
    params.sel_calls = &c_calls[0]
    params.sel_puts = &c_puts[0]
    params.sel_strikes = &c_strikes[0]
    params.n = n_sel

    # Search ±3% around put-call parity estimate
    cdef double search_lo = F_init * 0.97
    cdef double search_hi = F_init * 1.03

    cdef double result
    with nogil:
        result = _fminbound(
            _fwd_iv_abs_diff, search_lo, search_hi, <void*>&params,
            1e-6, 100,
        )
    return result

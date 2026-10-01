"""
Tiny scipy-free replacement for scipy.optimize.brentq.

Plain bisection: slower to converge than Brent's method (linear vs.
superlinear), but trivial to verify correct, and for this project we only
ever call it a handful of times (finding a return-map fixed point, or a
critical slope), so the extra iterations cost nothing in practice.
"""


def bisect(f, a, b, tol=1e-10, max_iter=200):
    """
    Find x in [a, b] with f(x) ~ 0, given f(a) and f(b) have opposite
    signs (raises ValueError otherwise, matching brentq's behavior).
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("f(a) and f(b) must have different signs")

    lo, hi, f_lo = a, b, fa
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_mid == 0.0 or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)
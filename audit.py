import numpy as np
from scipy.stats import beta

def cp_upper(k, n, a):
    return 1.0 if k >= n else beta.ppf(1 - a, k + 1, n - k)

def eps_lb(fp, fn, n, delta=1e-5, gamma=0.05):
    a, b = cp_upper(fp, n, gamma / 2), cp_upper(fn, n, gamma / 2)
    best = 0.0
    for num, den in [(1 - b - delta, a), (1 - a - delta, b)]:
        if num > 0:
            best = max(best, np.log(num / den))
    return best

def audit(s_in, s_out, frac_cal=0.2, **kw):
    s_in, s_out = np.asarray(s_in), np.asarray(s_out)
    m = int(frac_cal * len(s_in))
    grid = np.sort(np.concatenate([s_in[:m], s_out[:m]]))
    t = max(grid, key=lambda t: eps_lb((s_out[:m] > t).sum(),
                                       (s_in[:m] <= t).sum(), m, **kw))
    ne = len(s_in) - m
    return eps_lb((s_out[m:] > t).sum(), (s_in[m:] <= t).sum(), ne, **kw)
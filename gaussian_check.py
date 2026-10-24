import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from audit import audit

def delta_of(eps, sig):
    return norm.cdf(1/(2*sig) - eps*sig) - np.exp(eps)*norm.cdf(-1/(2*sig) - eps*sig)

sig = brentq(lambda s: delta_of(2.0, s) - 1e-5, 0.3, 10)   # sigma giving true eps = 2
rng = np.random.default_rng(0)
res = np.array([audit(rng.normal(1, sig, 1000), rng.normal(0, sig, 1000))
                for _ in range(200)])
print("median eps_lb:", np.median(res))
print("fraction above true eps (want < 0.05):", np.mean(res > 2.0))
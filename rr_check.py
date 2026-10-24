import numpy as np
from audit import audit

eps = 2.0
p = np.exp(eps) / (1 + np.exp(eps))
rng = np.random.default_rng(0)
res = np.array([audit(rng.binomial(1, p, 1000).astype(float),
                      rng.binomial(1, 1 - p, 1000).astype(float), delta=0.0)
                for _ in range(200)])
print("median eps_lb:", np.median(res))
print("fraction above true eps (want < 0.05):", np.mean(res > eps))
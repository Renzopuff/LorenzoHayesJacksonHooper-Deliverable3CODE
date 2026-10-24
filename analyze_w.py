import sys, os, numpy as np
from audit import audit

HERE = os.path.dirname(os.path.abspath(__file__))
eps = sys.argv[1]
s_in = np.load(os.path.join(HERE, "results", f"W_eps{eps}_in.npy"))
s_out = np.load(os.path.join(HERE, "results", f"W_eps{eps}_out.npy"))
for d in [1e-2, 1e-3, 1e-5]:
    print(f"W, eps_target={eps}, delta={d:g}: eps_lb =", audit(s_in, s_out, delta=d))
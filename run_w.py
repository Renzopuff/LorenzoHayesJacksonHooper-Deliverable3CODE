import sys, os, numpy as np
from train import train_once

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

eps = sys.argv[1]            # label only, e.g. "2"
sigma = float(sys.argv[2])   # sigma from Step 2 for that epsilon
n = int(sys.argv[3])         # runs per world, e.g. 20 for a pilot, 1000 for real

f_in = os.path.join(RES, f"W_eps{eps}_in.npy")
f_out = os.path.join(RES, f"W_eps{eps}_out.npy")
s_in  = list(np.load(f_in))  if os.path.exists(f_in)  else []
s_out = list(np.load(f_out)) if os.path.exists(f_out) else []

for r in range(len(s_in), n):
    s_in.append(train_once(True,  sigma, seed=2 * r))
    s_out.append(train_once(False, sigma, seed=2 * r + 1))
    if r % 10 == 0:
        np.save(f_in, s_in); np.save(f_out, s_out)
        print("done", r + 1, "of", n, flush=True)
np.save(f_in, s_in); np.save(f_out, s_out)
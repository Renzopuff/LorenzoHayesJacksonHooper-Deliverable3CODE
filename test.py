"""
dp_audit_tools.py

Helpers for the DP-SGD auditing paper:
  Step 1: accountant calibration (sigma, eps_RDP, eps_PRV)         -> needs opacus
  Step 2: Clopper-Pearson audit (eps_lb, eps_max, convergence)     -> numpy, scipy
  Step 5: gap decomposition table                                  -> numpy, pandas
  Extra : analytic eps for the Gaussian control (Table III row 1)  -> scipy

Install:  pip install numpy scipy pandas opacus
Run the self-test (no opacus needed for it):  python dp_audit_tools.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import beta, norm

# ----------------------------------------------------------------------
# Fixed configuration from Section III-C
# ----------------------------------------------------------------------
Q = 0.02          # Poisson sampling rate (L=100, n=5000)
T = 300           # training steps
DELTA = 1e-5
GAMMA = 0.05      # audit confidence parameter
TARGETS = (1, 2, 4)


# ======================================================================
# STEP 1: accountant calibration
# ======================================================================
def calibrate_and_account(target_eps: float, q: float = Q, steps: int = T,
                          delta: float = DELTA, tol: float = 0.01) -> dict:
    """Calibrate sigma with the RDP accountant, then report the eps that the
    RDP and PRV accountants return for that SAME sigma, q, and steps.
    (Section IV-E: report actual accountant outputs, not the target.)
    """
    from opacus.accountants import RDPAccountant, PRVAccountant
    from opacus.accountants.utils import get_noise_multiplier

    sigma = get_noise_multiplier(
        target_epsilon=target_eps, target_delta=delta, sample_rate=q,
        steps=steps, accountant="rdp", epsilon_tolerance=tol,
    )

    rdp = RDPAccountant()
    rdp.history = [(sigma, q, steps)]
    eps_rdp = rdp.get_epsilon(delta=delta)

    prv = PRVAccountant()
    prv.history = [(sigma, q, steps)]
    eps_prv = prv.get_epsilon(delta=delta)

    return {"target_eps": target_eps, "sigma": sigma,
            "eps_RDP": eps_rdp, "eps_PRV": eps_prv}


def accountant_table(targets=TARGETS) -> pd.DataFrame:
    """Fills the eps_RDP and eps_PRV columns of Table IV."""
    return pd.DataFrame([calibrate_and_account(t) for t in targets])


# ======================================================================
# Gaussian control: analytic eps (fills "Analytic ε to be set")
# ======================================================================
def gaussian_analytic_epsilon(mu: float, delta: float = DELTA) -> float:
    """Exact eps for distinguishing N(0,1) from N(mu,1) (Gaussian mechanism
    with sensitivity/sigma = mu), from Balle & Wang 2018:
        delta(eps) = Phi(-eps/mu + mu/2) - e^eps * Phi(-eps/mu - mu/2)
    """
    def d(eps):
        return norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)
    if d(0.0) <= delta:
        return 0.0
    hi = 1.0
    while d(hi) > delta:
        hi *= 2
    return brentq(lambda e: d(e) - delta, 0.0, hi)


# ======================================================================
# STEP 2: audit routine
# ======================================================================
def cp_upper(k: int, n: int, level: float) -> float:
    """One-sided Clopper-Pearson upper limit for a binomial rate, where
    `level` is the one-sided confidence (e.g. 1 - gamma/2)."""
    if k >= n:
        return 1.0
    return float(beta.ppf(level, k + 1, n - k))


def eps_from_counts(k_alpha: int, k_beta: int, n_alpha: int, n_beta: int,
                    delta: float = DELTA, gamma: float = GAMMA):
    """eps_lb = max{ ln[(1-beta+ - delta)/alpha+], ln[(1-alpha+ - delta)/beta+] }
    with one-sided CP upper limits, each at level gamma/2. Floored at 0."""
    a_up = cp_upper(k_alpha, n_alpha, 1 - gamma / 2)
    b_up = cp_upper(k_beta, n_beta, 1 - gamma / 2)
    cands = []
    for num, den in ((1 - b_up - delta, a_up), (1 - a_up - delta, b_up)):
        if num > 0 and den > 0:
            cands.append(math.log(num / den))
    return max(0.0, max(cands) if cands else 0.0), a_up, b_up


def eps_max(n_eval: int, gamma: float = GAMMA) -> float:
    """Finite-sample ceiling ln[N_eval / ln(2/gamma)] (5.38 at N=800)."""
    return math.log(n_eval / math.log(2 / gamma))


def choose_threshold(cal_scores_Dprime, cal_scores_D,
                     delta: float = DELTA, gamma: float = GAMMA) -> float:
    """Pick threshold t on the CALIBRATION split only. Guess D when s > t.
    Maximises the CP-based eps on calibration data, so t is not chosen by a
    noisy point estimate that happens to sit on a tiny tail count."""
    s0 = np.asarray(cal_scores_Dprime)   # world D' (target absent)
    s1 = np.asarray(cal_scores_D)        # world D  (target present)
    cands = np.unique(np.concatenate([s0, s1]))
    best_t, best_eps = float(np.median(cands)), -1.0
    for t in cands:
        k_alpha = int((s0 > t).sum())     # false positives
        k_beta = int((s1 <= t).sum())     # false negatives
        e, _, _ = eps_from_counts(k_alpha, k_beta, len(s0), len(s1), delta, gamma)
        if e > best_eps:
            best_eps, best_t = e, float(t)
    return best_t


@dataclass
class AuditRecord:
    """Everything Section III-B says the audit record should contain."""
    k_alpha: int
    k_beta: int
    alpha_upper: float
    beta_upper: float
    gamma: float
    delta: float
    n_eval: int
    threshold: float
    eps_lb: float
    eps_max: float

    @property
    def near_ceiling(self) -> bool:
        return self.eps_lb >= 0.9 * self.eps_max


def run_audit(scores_Dprime, scores_D, cal_frac: float = 0.2,
              delta: float = DELTA, gamma: float = GAMMA,
              seed: int = 0, n_eval: int | None = None) -> AuditRecord:
    """Full audit for ONE adversary at ONE epsilon target.

    scores_Dprime / scores_D: one score per run in each world, oriented so
    that HIGHER score means "target present" (e.g. W uses -loss, S uses the
    projection <u, theta_t - theta_{t+1}> summed over t).
    With 1000 runs per world and cal_frac=0.2 -> 200 calibration, 800 eval.
    If n_eval is given, only the first n_eval evaluation runs are used
    (threshold still comes from the calibration split) for the convergence curve.
    """
    rng = np.random.default_rng(seed)
    s0 = np.asarray(scores_Dprime, dtype=float)
    s1 = np.asarray(scores_D, dtype=float)
    p0, p1 = rng.permutation(len(s0)), rng.permutation(len(s1))
    n_cal0, n_cal1 = int(round(cal_frac * len(s0))), int(round(cal_frac * len(s1)))
    cal0, ev0 = s0[p0[:n_cal0]], s0[p0[n_cal0:]]
    cal1, ev1 = s1[p1[:n_cal1]], s1[p1[n_cal1:]]
    if n_eval is not None:
        ev0, ev1 = ev0[:n_eval], ev1[:n_eval]

    t = choose_threshold(cal0, cal1, delta, gamma)
    k_alpha = int((ev0 > t).sum())
    k_beta = int((ev1 <= t).sum())
    e, a_up, b_up = eps_from_counts(k_alpha, k_beta, len(ev0), len(ev1), delta, gamma)
    return AuditRecord(k_alpha, k_beta, a_up, b_up, gamma, delta,
                       len(ev0), t, e, eps_max(len(ev0), gamma))


def convergence_curve(scores_Dprime, scores_D, n_evals=(50, 100, 200, 400, 800),
                      **kw) -> pd.DataFrame:
    """eps_lb vs N_eval, with eps_max alongside (required diagnostic).
    Add eps_PRV as a reference line when plotting."""
    rows = []
    for n in n_evals:
        r = run_audit(scores_Dprime, scores_D, n_eval=n, **kw)
        rows.append({"N_eval": n, "eps_lb": r.eps_lb, "eps_max": r.eps_max,
                     "k_alpha": r.k_alpha, "k_beta": r.k_beta})
    return pd.DataFrame(rows)


def records_to_frame(records: dict[str, AuditRecord]) -> pd.DataFrame:
    """records = {'W': rec, 'O': rec, ...} -> tidy table for the appendix."""
    return pd.DataFrame({k: asdict(v) for k, v in records.items()}).T


# ======================================================================
# STEP 5: gap decomposition (Table V)
# ======================================================================
def gap_row(eps_rdp, eps_prv, lb: dict) -> dict:
    """lb = {'W':..., 'O':..., 'C':..., 'S_f':..., 'S':...} (eps_lb values).
    Negative components are kept, not clipped (Section IV-C)."""
    d_acct = eps_rdp - eps_prv
    d_res = eps_prv - lb["S"]
    d_data = lb["C"] - lb["W"]
    d_access = lb["O"] - lb["W"]
    d_int = lb["S"] - lb["W"] - d_data - d_access
    total = eps_rdp - lb["W"]
    d_hid = lb["S"] - lb["S_f"]
    check = d_acct + d_res + d_data + d_access + d_int
    assert abs(check - total) < 1e-9, "components must sum to eps_RDP - eps_lb(W)"
    return {"Δ_acct": d_acct, "Δ_data": d_data, "Δ_access": d_access,
            "Δ_int": d_int, "Δ_res": d_res, "Total": total, "Δ_hid": d_hid}


def gap_table(acct: pd.DataFrame, lb_by_target: dict[float, dict]) -> pd.DataFrame:
    """acct: output of accountant_table(); lb_by_target: {1: {...}, 2: {...}, 4: {...}}."""
    rows = []
    for _, r in acct.iterrows():
        t = r["target_eps"]
        row = {"ε target": t}
        row.update(gap_row(r["eps_RDP"], r["eps_PRV"], lb_by_target[t]))
        rows.append(row)
    return pd.DataFrame(rows).round(3)


# ======================================================================
# Self-test with synthetic data (no training or opacus required)
# ======================================================================
if __name__ == "__main__":
    print(f"eps_max(800) = {eps_max(800):.2f}  (paper: 5.38)")
    print("ceilings:", {n: round(eps_max(n), 2) for n in (50, 100, 200, 400, 800)})

    # Gaussian control: truth must stay above eps_lb, and eps_lb should rise with N
    rng = np.random.default_rng(1)
    mu = 3.0
    truth = gaussian_analytic_epsilon(mu)
    print(f"\nGaussian control, mu={mu}: analytic eps = {truth:.3f}")
    s0 = rng.normal(0, 1, 1000)
    s1 = rng.normal(mu, 1, 1000)
    print(convergence_curve(s0, s1).round(3).to_string(index=False))
    rec = run_audit(s0, s1)
    print("full audit eps_lb:", round(rec.eps_lb, 3), "| below truth:", rec.eps_lb <= truth)

    # Decomposition with made-up numbers, ONLY to test the arithmetic
    fake_acct = pd.DataFrame([{"target_eps": 2, "sigma": 0.0, "eps_RDP": 2.0, "eps_PRV": 1.6}])
    fake_lb = {2: {"W": 0.2, "O": 0.5, "C": 0.4, "S_f": 1.0, "S": 1.4}}
    print("\nDecomposition arithmetic check (fake numbers, NOT results):")
    print(gap_table(fake_acct, fake_lb).to_string(index=False))

    # Real accountant step (needs opacus):
    # print(accountant_table())
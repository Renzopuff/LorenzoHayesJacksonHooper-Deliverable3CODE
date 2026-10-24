Empirical Privacy Auditing of DP-SGD

Python scripts for estimating the privacy loss that a differentially private training run actually incurs, rather than the bound its accountant reports. The pipeline trains many small models with and without a target example, scores each model with an adversary, and converts the resulting error rates into a statistically valid lower bound on ε.

The target implementation is DP-SGD as provided by Opacus (PyTorch), trained on a small MNIST subset so that thousands of independent runs are practical.

Contents
Overview
Repository layout
Installation
Usage
Configuration
How the audit works
Adversaries
Extending the audit
Interpreting results
Reproducibility
Troubleshooting
References
Overview

A mechanism is (ε, δ)-differentially private if no test can distinguish two neighboring datasets with more advantage than ε and δ allow. An audit turns this around: fix a neighboring pair, run the mechanism many times on each, apply a test, and measure how well it distinguishes them. The measured advantage gives a lower bound ε_lb on the true privacy loss of the implementation. Comparing ε_lb with the accountant's bound shows how much of the guarantee an attack can realize, and an ε_lb above the claimed bound would indicate a violation.

This repository provides:

Privacy accounting for the configured DP-SGD setup, using the RDP and PRV accountants.
A reusable audit function that selects a threshold on a calibration split and reports ε_lb with Clopper–Pearson confidence limits.
Validity checks on mechanisms with known privacy loss (a Gaussian mechanism and randomized response).
A training and collection pipeline for the weak adversary W: black-box access to the final model, with an ordinary training example as the target.
Repository layout
.
├── accounting.py        # σ, ε_RDP and ε_PRV for each target ε
├── audit.py             # core audit: Clopper–Pearson limits and ε_lb
├── gaussian_check.py    # validity check on a Gaussian mechanism
├── rr_check.py          # tightness check on randomized response
├── train.py             # one private training run; returns the adversary W score
├── check_model.py       # confirms the model learns when noise is near zero
├── try_one.py           # times a single training run
├── run_w.py             # collects scores for adversary W (resumable)
├── analyze_w.py         # computes ε_lb from saved scores
├── requirements.txt
├── data/                # MNIST download (created on first run, not tracked)
└── results/             # saved scores (created automatically, not tracked)
Installation

Python 3.10 or newer is required.

bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

For an NVIDIA GPU, install PyTorch from pytorch.org first, then run the command above.

Results depend on library versions. Record the environment you used:

bash
pip freeze > requirements-lock.txt

Do not name a script after a library it imports (opacus.py, torch.py, numpy.py, scipy.py). Python would import the script instead of the library.

Usage

Run all commands from the repository root.

1. Accounting
bash
python accounting.py

Prints the noise multiplier σ and the ε values from the RDP and PRV accountants for each target ε, at δ ∈ {10⁻², 10⁻³, 10⁻⁵}. The PRV value is tighter than the RDP value for the same σ, q and number of steps.

2. Audit validity checks
bash
python gaussian_check.py
python rr_check.py
gaussian_check.py audits a single Gaussian mechanism with analytically known ε = 2 at δ = 10⁻⁵ (1,000 samples per world, 200 repeated audits). It prints the median ε_lb and the fraction of audits whose ε_lb exceeds the true ε; that fraction should stay below 0.05. A typical median is about 0.39. This check demonstrates validity, not tightness: at small δ the privacy loss of a Gaussian mechanism is driven by rare tail events that a few thousand samples cannot resolve.
rr_check.py audits randomized response, a mechanism with pure ε-DP (δ = 0) and known ε = 2. Here the bound can be tight, so the median ε_lb should approach 2 (roughly 1.8 with 1,000 samples per world), again with a fraction above the true ε under 0.05.
3. Model check and timing
bash
python check_model.py
python try_one.py

check_model.py trains with almost no noise. The printed scores (negative loss on the target example) should be well above the chance level of about −2.3. If they are not, tune the learning rate in train.py. try_one.py prints the seconds taken by one training run; multiply by twice the number of runs per world to estimate the cost of one campaign.

4. Collect scores for adversary W
bash
python run_w.py <label> <sigma> <runs_per_world>

python run_w.py 2 1.1206 20      # short test
python run_w.py 2 1.1206 1000    # full campaign

Writes results/W_eps<label>_in.npy (target example in the training set) and results/W_eps<label>_out.npy (target example removed). Every run uses a distinct seed. The script is resumable: rerun the same command after an interruption and it continues from the saved scores. Progress is printed every ten runs.

5. Compute ε_lb
bash
python analyze_w.py 2

Prints ε_lb at δ = 10⁻², 10⁻³ and 10⁻⁵ from the same saved scores.

Configuration
Parameter	Value
Dataset	MNIST, fixed random subset of 5,000 examples; example 0 is the audit target
Model	Small CNN: two convolution layers, one linear layer
Sampling	Poisson, rate q = 0.02 (expected batch size 100)
Steps T	300 (6 epochs of 50 steps)
Clip norm C	1.0
Optimizer	Plain SGD; the learning rate is set in train.py
Target ε	1, 2, 4 at δ = 10⁻⁵
Noise multiplier σ	1.6797, 1.1206, 0.8350 respectively
Neighboring relation	Add/remove one example

σ depends on q, T and the target. After changing the dataset size, batch size, number of epochs or targets, recompute σ (for example with opacus.accountants.utils.get_noise_multiplier) and update the values in accounting.py.

How the audit works

Each trained model yields one score, oriented so that a higher score means "the target was in the training set". For each world (target present, target absent):

The first 20% of runs choose the threshold that maximizes ε_lb.
The remaining 80% count false positives (a guess of "present" when the target was absent) and false negatives at that threshold.
One-sided Clopper–Pearson upper limits α⁺ (false positives) and β⁺ (false negatives), each at level γ/2 with γ = 0.05, give ε_lb = max{ ln((1 − β⁺ − δ) / α⁺), ln((1 − α⁺ − δ) / β⁺) } which holds with probability at least 1 − γ for the tested pair of datasets and the tested adversary.

Selecting the threshold on separate runs avoids correcting for multiple comparisons. With N_eval evaluation runs per world, a mechanism with bounded privacy loss cannot be certified above about ln(N_eval / ln(2/γ)); this is roughly 5.4 for N_eval = 800. For Gaussian-type mechanisms such as DP-SGD the practical limit is lower and depends on δ and on the strength of the signal.

Adversaries
ID	Target	Access	Score
W	Ordinary training example	Black-box, final model	Negative loss on the target

run_w.py implements W. The analysis functions in audit.py apply to any adversary that produces one scalar score per run in each world.

Extending the audit

The audit core is adversary-independent, so stronger adversaries and faulty implementations can be added by producing the same pair of score arrays.

Stronger adversaries
ID	Target	Access	Score
O	Ordinary example z	White-box, all iterates	Σₜ ⟨clip_C(∇ℓ(θₜ; z)), θₜ − θₜ₊₁⟩
C	Crafted input canary x_c	Black-box, final model	−ℓ(θ_T; x_c)
S	Gradient canary C·u	White-box, all iterates	Σₜ of the per-step log-likelihood ratio of the observed projection ⟨u, θₜ − θₜ₊₁⟩
S_f	Gradient canary C·u	White-box, final model only	⟨u, θ₀ − θ_T⟩

Notes:

A gradient canary is added to the clipped gradient sum with probability q at each step, matching the accountant's sampling model. Its direction u is a fixed random unit vector.
Scoring S as the plain sum Σₜ ⟨u, θₜ − θₜ₊₁⟩ gives the same statistic as S_f, because the sum telescopes. S must use a nonlinear per-step statistic; the difference between S and S_f then measures the benefit of observing intermediate iterates.
Gradient canaries are added after clipping, so they cannot detect clipping faults. Use an input-space canary (C) for those.
Fault-injected variants

Variants with a known departure from the stated mechanism serve as positive controls for the audit.

ID	Departure	Suggested adversary
M1	Clip the batch-summed gradient to norm C instead of each per-sample gradient, then add noise σC	C
M2(k)	Divide the noise multiplier by k, for example k ∈ {1.25, 1.5, 2}	S
M3	Per-layer clipping at C per layer while the accountant assumes a total norm of C	C
M4	Fixed-size shuffled batches while the accountant assumes Poisson sampling	S

Implementation hints:

M2 changes only the noise_multiplier passed to make_private.
M3: make_private(..., clipping="per_layer", max_grad_norm=[C, ...]), with one entry per parameter tensor. Check how your Opacus version defines a layer.
M4: make_private(..., poisson_sampling=False).
M1 requires replacing DPOptimizer.clip_and_accumulate with a version that sums per-sample gradients before clipping. This relies on Opacus internals that differ between versions, so check it against the installed opacus/optimizers/optimizer.py. Raise the learning rate roughly in proportion to the batch size, since clipping the whole sum shrinks honest steps.
Flag a variant when ε_lb exceeds the tighter valid bound for the correct mechanism (ε_PRV at the audit δ). A non-detection means only that this adversary and sample size did not detect the fault.
Interpreting results
ε_lb is a lower bound for the tested adversary, dataset pair and sample size. A small value means the audit could not show more than that, not that the mechanism is that private.
ε_lb above the stated bound at the stated confidence level indicates a violation, but only after the validity checks behave as expected and the unmodified implementation is not flagged.
A gap between ε_lb and the accountant's bound can come from accountant slack (RDP versus PRV), a weak adversary, too few runs, or a real implementation difference. Collect scores at several values of N_eval and plot ε_lb against N_eval to see whether an estimate is still rising.
Always compare ε_lb with theory at the same δ.
Report the stated bound (RDP and PRV), ε_lb, the confidence level, N_eval, δ, and the Opacus and PyTorch versions together.
Reproducibility
Seeds are fixed per run (2r for the world with the target, 2r + 1 for the world without) and are independent across runs. Use distinct seed ranges for different campaigns that share an output folder.
The data subset is chosen with a fixed seed.
The learning rate and architecture must be identical in both worlds and across all campaigns.
Pin and record library versions (pip freeze).
Troubleshooting
Symptom	Cause and fix
No module named 'opacus.accountants'; 'opacus' is not a package	A file named opacus.py exists in the folder. Rename or delete it, and delete __pycache__.
FileNotFoundError for results/...	Use the provided run_w.py and analyze_w.py, which create results/ next to the script.
PowerShell refuses to activate the environment	Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned, then activate again.
Training is slow	Time one run with try_one.py. Use a GPU, run overnight (campaigns resume), or reduce the epochs and recompute σ.
rr_check.py prints a median far below 2	Check that scores are oriented so a higher value means "present", and that delta=0.0 reaches eps_lb.
"Full backward hook is firing..."	Harmless warning from Opacus.
"Optimal order is the largest alpha"	Harmless. The reported ε is still a valid upper bound; Opacus's default Rényi orders were used.
References
M. Abadi et al., "Deep learning with differential privacy," ACM CCS, 2016, pp. 308–318.
A. Yousefpour et al., "Opacus: User-friendly differential privacy library in PyTorch," arXiv:2109.12298, 2021.
I. Mironov, "Rényi differential privacy," IEEE CSF, 2017, pp. 263–275.
S. Gopi, Y. T. Lee, L. Wutschitz, "Numerical composition of differential privacy," NeurIPS, 2021.
P. Kairouz, S. Oh, P. Viswanath, "The composition theorem for differential privacy," ICML, 2015, pp. 1376–1385.
C. J. Clopper and E. S. Pearson, "The use of confidence or fiducial limits illustrated in the case of the binomial," Biometrika 26(4), 1934, pp. 404–413.
B. Balle and Y.-X. Wang, "Improving the Gaussian mechanism for differential privacy: Analytical calibration and optimal denoising," ICML, 2018, pp. 394–403 (analytic Gaussian δ(ε) formula used in gaussian_check.py).
M. Jagielski, J. Ullman, A. Oprea, "Auditing differentially private machine learning: How private is private SGD?" NeurIPS, 2020.
M. Nasr, S. Song, A. Thakurta, N. Papernot, N. Carlini, "Adversary instantiation: Lower bounds for differentially private machine learning," IEEE S&P, 2021, pp. 866–882.
M. Nasr et al., "Tight auditing of differentially private machine learning," USENIX Security, 2023, pp. 1631–1648.
T. Steinke, M. Nasr, M. Jagielski, "Privacy auditing with one (1) training run," NeurIPS, 2023.
L. Chua et al., "How private are DP-SGD implementations?" ICML, 2024, pp. 8904–8918.

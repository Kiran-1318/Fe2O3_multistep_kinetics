# Fe₂O₃ Multi-Step Kinetics — Coupled Inverse PINN + SINDy with Learnable n

Autonomous step-wise kinetic discovery for the three-step Fe₂O₃ reduction
reaction network (Fe₂O₃→Fe₃O₄→FeO→Fe) using a coupled Inverse PINN and
SINDy pipeline with a learnable Avrami exponent.

**No prior kinetic model assumed at any step.**
Both the Arrhenius parameters (A_i, Eₐ_i), the reaction model f_i(X_i),
and the Avrami exponent n_i are discovered simultaneously from deconvoluted
TGA data.

**Related preprints:**
- [Preprint 1](https://doi.org/10.26434/chemrxiv.15003636/v1) — Inverse PINN
  + PI-DeepONet for lumped Fe₂O₃ kinetics (ChemRxiv, May 2026)
- [Preprint 2](https://doi.org/10.26434/chemrxiv.15003637/v1) — Coupled PINN
  + SINDy for lumped model discovery (ChemRxiv, May 2026)
- **Preprint 3:** *In preparation — DOI will be added upon publication*

---

## Overview

Fe₂O₃ reduction by H₂ proceeds through three sequential steps in chemical
looping hydrogen generation:

Fe₂O₃ → Fe₃O₄ → FeO → Fe

Each step has its own kinetic parameters and reaction mechanism. This
repository implements a coupled Inverse PINN + SINDy pipeline applied
separately to each step using Wang et al. (2023)'s deconvoluted DTG data,
then aggregates the results to reproduce total conversion X(t).

**Governing equations:**

dX₁/dt = A₁·exp(−Eₐ₁/RT)·f₁(X₁)     Step 1: Fe₂O₃→Fe₃O₄
dX₂/dt = A₂·exp(−Eₐ₂/RT)·f₂(X₂)     Step 2: Fe₃O₄→FeO
dX₃/dt = A₃·exp(−Eₐ₃/RT)·f₃(X₃)     Step 3: FeO→Fe
X_total = w₁·X₁ + w₂·X₂ + w₃·X₃
w₁=0.111, w₂=0.334, w₃=0.555  (stoichiometric weights)

---

## Notebooks

### `4_multistep_kinetics_invPINN_SINDY.ipynb` — Step detection diagnostic

Investigates step-number discovery from raw total X(t) data using:
- Signal processing (PINN autograd derivatives, peak detection)
- Bayesian deconvolution (Gaussian peak fitting with BIC selection)

**Key finding:** Step number cannot be reliably resolved from lumped X(t)
data due to severe peak overlap at lower temperatures — motivating the use
of Wang et al.'s deconvoluted step data.

---

### `4_multi_kinetics_discovery.ipynb` — Main coupled pipeline

**The core contribution.** Applies the coupled Inverse PINN + SINDy
algorithm (from Preprint 2) to each reduction step independently using
deconvoluted DTG data.

**Key innovation — learnable Avrami exponent n:**

f_JMA(X, n) = (1−X) · (−ln(1−X))^((n−1)/n)

- n is jointly optimised with A and Eₐ in the Inverse PINN (Step A)
- SINDy library is parameterised with discovered n after iteration 1
- Eliminates discretisation error from fixed n library terms
- Results in guaranteed single-term sparse f_i(X_i)

**Coupled iterative algorithm:**

INITIALISE: A_i⁰ (Wang 2023/60 s⁻¹), Eₐ_i⁰, n_i⁰=2.0
Iteration k:
Step A — Inverse PINN:
Fix f_i^k(X_i) → optimise PINN weights + log_A_i + log_Eₐ_i + n_i
Physics residual: dX̂_i/dt − A_i·exp(−Eₐ_i/RT)·f_i^k(X̂_i) = 0
Output: A_i^(k+1), Eₐ_i^(k+1), n_i^(k+1)
Step B — SINDy:
Fix A_i^(k+1), Eₐ_i^(k+1), n_i^(k+1)
Compute f̂_i(X_i) = dX̂_i/dt|PINN / k_i(T)  (PINN autograd — no noise)
Library: parameterised (1−X)·(−ln(1−X))^((n−1)/n)
Modified BIC threshold sweep (penalty ×200)
Output: f_i^(k+1)(X_i) — symbolic, 1 active term
Until ΔA < 1%, ΔEₐ < 1%, Δn < 1%

**SINDy libraries per step:**

| Step | Library type | Rationale |
|---|---|---|
| Step 1 (Fe₂O₃→Fe₃O₄) | JMA + parameterised n | Bell-shaped f(X) in conversion space |
| Step 2 (Fe₃O₄→FeO) | JMA + parameterised n | Bell-shaped f(X) in conversion space |
| Step 3 (FeO→Fe) | Diffusion/zero-order | Flat-plateau f(X) — trapezoid DTG shape |

**Final results:**

| Step | A (s⁻¹) | Eₐ (kJ/mol) | n | f(X) | Weight |
|---|---|---|---|---|---|
| Fe₂O₃→Fe₃O₄ | 4.52 | 32.60 | **2.008** | (1−X)(−ln(1−X))^0.502 | 0.111 |
| Fe₃O₄→FeO | 0.469 | 23.49 | **2.008** | (1−X)(−ln(1−X))^0.502 | 0.334 |
| FeO→Fe | 0.0143 | 12.01 | N/A | **(1−X)^(2/3)** | 0.555 |

**Aggregated validation — X_total = w₁X₁ + w₂X₂ + w₃X₃:**

| Temperature | R² | Status |
|---|---|---|
| 750°C | 0.9599 | Training |
| 850°C | 0.9824 | Training |
| 900°C | 0.9684 | Training |
| 800°C | 0.9796 | Validation (interpolation) |
| 950°C | **0.9803** | Validation (extrapolation) |

**Key scientific findings:**
1. **n₁=2.008, n₂=2.008** — independently confirms Wang 2023's assumed JMA n=2
   without that assumption being made anywhere in the algorithm
2. **Step 3 follows shrinking core (1−X)^(2/3)** — not JMA as Wang 2023 assumed
3. **Step 2 Eₐ=23.49 kJ/mol** agrees with Wang 2023 (26.7, Δ=12%)
4. **Step 1 Eₐ exhibits kinetic compensation** — k(T) is reliably identified;
   individual A and Eₐ are degenerate over the 750–900°C training window

**Framework:** Pure PyTorch + pysindy. No DeepXDE or TensorFlow.

---

## Data

### `converted_data.csv`

Preprocessed deconvoluted DTG data from Wang et al. (2023) Figures 5a–5c.

**Columns:**
- `Step`: Step_1_rxn / Step_2_rxn / Step_3_rxn
- `Temperature_K`: temperature in Kelvin
- `Time_s`: time in seconds
- `Rate`: raw DTG rate (min⁻¹)
- `X_active`: step conversion X_i ∈ [0.03, 0.97] (active reaction region)
- `f_norm`: normalised reaction model f_i(X_i) = Rate / k_i(T)

**Preprocessing applied:**
- Raw (time_min, dX_i/dt) digitised from Wang 2023 Figures 5a–5c
- X_i(t) obtained by cumulative trapezoidal integration
- Normalised to X_i ∈ [0,1] per step per temperature
- Savitzky-Golay smoothing (window=7, polyorder=3)
- Baseline correction: negative rates clipped to zero

**Training split:** 750, 850, 900°C
**Validation:** 800°C (interpolation), 950°C (extrapolation)
**Step 3 training:** all 5 temperatures (wider coverage for Eₐ identification)

**Original source:**
Wang, H. et al. (2023). Multistep kinetic study of Fe₂O₃ reduction by H₂
based on isothermal thermogravimetric analysis data deconvolution.
International Journal of Hydrogen Energy, 48, 16601–16613.

---

## Requirements

torch>=2.0.0
pysindy>=1.7.0
numpy
pandas
matplotlib
scikit-learn
scipy

---

## Related repositories

[1D-Heat-Equation-PINN](https://github.com/Kiran-1318/1D-Heat-Equation-PINN)
— Heat equation PINN. Rel L2: 0.99%.

[DeepXDE-Chemical-Looping-Problems](https://github.com/Kiran-1318/DeepXDE-Chemical-Looping-Problems)
— 10 original PINN problems: forward ODEs → inverse Arrhenius → hybrid.

[NeuralOperator-Chemical-Looping-Problems](https://github.com/Kiran-1318/NeuralOperator-Chemical-Looping-Problems)
— 5 neural operator problems: DeepONet, FNO from scratch, PI-DeepONet.

[Fe2O3_redox_PINN](https://github.com/Kiran-1318/Fe2O3_redox_PINN)
— Preprint 1: inverse PINN + PI-DeepONet for lumped kinetics.
[DOI: 10.26434/chemrxiv.15003636/v1](https://doi.org/10.26434/chemrxiv.15003636/v1)

[Fe2O3_stepwise_kinetics](https://github.com/Kiran-1318/Fe2O3_stepwise_kinetics)
— Preprint 2: coupled PINN + SINDy — discovers grain model without prior assumption.

---

## Citation

If you use this code or data, please cite:


---

## Author

**Kiran Thammina**
M.Tech Energy Systems Engineering, IIT Bombay (CPI 9.84, Best Thesis Award)
GitHub: [github.com/Kiran-1318](https://github.com/Kiran-1318)

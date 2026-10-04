# HOPS_NLayer — n-layer 2D HOPS/AWE grating solver (no PINNs)

This package extends the 2-layer HOPS/AWE solver in `HOPS_Python` (the Python port of the MATLAB code that
goes with *hopsawecomput.jsc*) to stacks of **L ≥ 2 layers** of arbitrary materials. Every one of the L−1
interfaces can be corrugated with its own profile and its own amplitude. The solver computes the reflectivity
map R(ε, ω), the energy defect D (lossless stacks) or the absorptance A (lossy stacks), the fields in every
layer, and the analysis tools built on them: manufactured solutions, a single (ε, δ) test, maps, a layer-count
study, movies and a material survey.

`HOPS_Python` must be a **sibling folder**, for example `.../2d/HOPS_Python` and `.../2d/HOPS_NLayer`. You can
also point the environment variable `HOPS_PYTHON` to it.

```
pip install -r requirements.txt
python -m pytest tests -q                 # 10 tests, about 20 s
python test_mms_error_nlayer.py           # same checks, PASS/FAIL table (analogue of test_mms_error.py)
python refl_map_nlayer.py --scenario bragg_L8
```

## 1. Formulation

**Geometry.**

- Layers l = 0 (top, incidence medium) … L−1 (substrate) are separated by the interfaces
  z = h_j + g_j(x), with g_j = ε·amp_j·f_j(x), h_1 = 0 and h_{j+1} = h_j − d_j.
- Artificial boundaries sit at z = a above the top interface and z = −b below the bottom interface.

**Transformed field expansion (TFE) in each layer.** Each layer is flattened by

    z = z_b + g_b + s (d + g_t − g_b),   s ∈ [0, 1],

where λ = 1 + εμ, μ = (f_t − f_b)/d and q = f_b' + s(f_t' − f_b'). Multiplying the Helmholtz operator by λ²
makes the equation **exactly quadratic in ε**:

    Δu + 2iᾱ ∂x u + γ̄² u  =  ε P1 + ε² P2  (+ 2iᾱ B_k, γ̄² C_k terms),

with

    P1 = ∂x(2μ ux − q uz) − μ' ux − ∂z(q ux)
    P2 = ∂x(μ² ux − μq uz) − μ'(μ ux − q uz) + ∂z(−μq ux + q² uz)

For L = 2 these are exactly the coefficients of the `hops` 2-layer code.

**Interface conditions.**

- Continuity: u_above − u_below = ζ_j.
- Flux: λ_b Ñ_above − τ_j² λ_a Ñ_below = λ_a λ_b ψ_j, where
  Ñ = (1 + g_x²) u_z − λ g_x u_x − iαλ g_x u is cubic in ε.
- τ_j² = (n_{j−1}/n_j)² for TM and 1 for TE.
- The incident plane wave enters only through interface 1, using `hops.setup_zeta_psi_n_m`.

**Transparent boundary conditions.** At z = a and z = −b the condition is u_z = ±λ T(ω) u, with the Dirichlet–Neumann
operator `T_dno` of `hops`.

**Joint (ε, δ) expansion.** Writing ω = ω̄(1 + δ), the solution is expanded as

    u = Σ_{n ≤ N, m ≤ M} u_{n,m} εⁿ δᵐ

Each order (n, m) is **one** solve with the *same* flat multi-domain operator. Its right-hand side collects
the lower orders (n−1, n−2, n−3 and m−1, m−2).

**Energy.** R, T and D = 1 − R − T are evaluated in the Fourier domain with `hops.energy_defect`
(τ² = (n_0/n_{L−1})²). The sums use Taylor or Padé (`fcn_sum_fast`), as in the 2-layer code.

## 2. Speed-ups

The 2-layer speed-ups were all kept, and some new ones were added:

| speed-up | effect |
|---|---|
| **one block-Chebyshev operator per Fourier mode**, size Σ(Nz_l + 1). It is inverted **once per window** (batched `np.linalg.inv` over all modes) and cached by (stack, ω̄, ᾱ) | each of the (N+1)(M+1) orders costs one batched mat-vec instead of a factorisation |
| batched solve `np.matmul(Ainv, fft(rhs)[:, :, None])` for all Fourier modes at once | no Python loop over p |
| caches of the x/z derivatives of every stored order (VX, VZ) | ≈ 35 % faster than recomputing them |
| only 4 n-levels kept in memory (the recursion reaches back to n − 3) | memory O(4 M L Nx Nz) |
| dense spectral x-differentiation matrices for Nx ≤ 64, FFT above that | faster for small Nx |
| interface sums done in the Fourier domain; energies summed with `fcn_sum_fast` | as in the 2D code |
| windows run in parallel (`--workers`, process pool) | maps scale with the number of cores |
| conformal or "top-only" corrugation | removes the TFE singularity ε* (see §5) and allows larger ε |

**Cost per window** (clean single-process run, `bench_nlayer.py`, N = M = 12, Nz = 24 per layer):

| L | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| matrix size | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 300 | 400 | 500 |
| Nx = 32 (s) | 0.26 | 0.34 | 0.45 | 0.56 | 0.70 | 0.82 | 0.93 | 1.8 | 3.0 | 4.2 |
| Nx = 64 (s) | 0.35 | 0.53 | 0.68 | 0.87 | 1.07 | 1.30 | 1.47 | | | |

The cost is **linear in L**, about 0.11 s per extra layer at Nx = 32. The O(L³) inverse is below 10 % of the
total even at L = 8, so the recursion dominates. The 2-layer case costs about the same as the 2D coupled
solver (0.19 vs 0.17 s). For L = 12 and L = 20, the flat-stack R agrees with the transfer-matrix result to
2e-14 and 5e-14, so going beyond 8 layers is both feasible and accurate.

Further speed-ups considered but not needed:

- Block-tridiagonal (Thomas) elimination instead of a dense inverse. This only matters once L ≳ 30.
- float32 for the high orders. Not done, because it harms the Padé summation.
- Numba for the RHS assembly. The NumPy code is already vectorised over x and z.

## 3. Validation

| check | result |
|---|---|
| L = 2 reproduces the 2D `hops` coupled solver (all coefficients) | relative < 1e-10 (test threshold) |
| `refl_map_nlayer` with L = 2 equals `refl_map.py` (silver) | R and D agree to < 1e-9 |
| flat stacks agree with the transfer-matrix method (`hops_nl/reference.tmm`), TE and TM, lossy layers | < 1e-11 (typically 1e-14) |
| fictitious-interface invariance (splitting a layer into two identical ones) | < 1e-11 |
| energy conservation, lossless stacks (pointwise) | \|D\| < 1e-11 (typically 1e-14) |
| **MMS**, exact per-layer Helmholtz solutions (`mms_error_nlayer.py`), N = M = 12, ε ≤ 0.2, \|δ\| ≤ 0.1 | Padé 1e-9 – 2.5e-8, Taylor 1e-7 – 2.6e-6 for L = 3 … 8 |
| MMS, L = 8 oblique incidence (α = 0.15) | Taylor 1e-3, Padé 8.6e-5 |
| **single (ε, δ)** (`test_single_eps_delta_nlayer.py`): V_j, ubar, wbar **and the volume field of every layer** | 1e-11 – 1e-13 (Padé), L = 3, 5, 8 |
| *true* error of every map: AWE map vs pointwise HOPS (no δ expansion, Padé in ε, N = 24) on 3 columns per window | see §4 |
| **per-layer absorption** from the volume series vs transfer matrix layer by layer (flat, TE/TM); Σ A_l = 1 − R − T on corrugated stacks | < 1e-8 / < 1e-6 (test N9) |
| **exact dispersion + fixed angle** (Kretschmann BK7/Au/water, 68°): flat row vs TMM with n(λ); corrugated rows vs pointwise HOPS | < 1e-7 / < 1e-6 (test N10) |

MMS convergence with N = M (figures/mms_error/mms_convergence.png): the Padé error falls from about 1e-2 to
2.5e-11 (L = 2–3) and to 2e-8 (L = 8). The Taylor error falls geometrically to about 2e-8 at L ≤ 3. For
L ≥ 5 it bottoms out at 1e-6 – 1e-5 around N = M ≈ 10: round-off grows in the highest-order coefficients as
layers are added. So **N = M = 10 – 12 with Padé** is the recommended setting.

## 4. Reflectivity maps (`refl_map_nlayer.py`, `run_all_maps.py`)

Scenarios are listed by `python refl_map_nlayer.py --list-scenarios`. Figures go to `figures/refl_map/L<L>/`
(R and D/A maps, each with a stack sketch) and the table to `results/maps_summary.md`. The settings are a
100 × 100 grid per window, joint windows, |δ| ≤ 0.1 and Padé summation, unless noted otherwise.

| scenario | L | stack | R range | log10\|D\| max / median, or A | true error max / median |
|---|---|---|---|---|---|
| film_dielectric | 3 | 1 \| 1.3 \| 1.1 | 0.002–0.046 | −0.9 / −9.4 | 6.4e-5 / 2e-10 |
| film_dielectric_paper (Taylor, paper windows) | 3 | same | 0.002–0.069 | −1.2 / −4.9 | 1.1e-2 / 1.4e-5 |
| film_dielectric_top / _anti | 3 | same, other corrugations | 0.002–0.045 | −0.8 / −9.6 | 5e-6 / 2e-3 |
| silver_film, silver_film_thick (IMI) | 3 | 1 \| Ag \| 1 | 0.48–0.97 | A 0.03–0.10 | 3e-4 |
| gold_film_glass | 3 | air \| Au 40 nm \| BK7 | 0.17–0.97 | A 0.02–0.55 | 2.4e-5 |
| **gmr_filter** (guided-mode resonance) | 3 | air \| TiO2 120 nm \| silica | 0.03–**0.93** | −0.9 / −13.0 | 2.4e-6 |
| ar_coating | 3 | air \| MgF2 100 nm \| BK7 | 0.011–0.044 | — | 1.5e-7 |
| **mim_absorber** (gap plasmon) | 4 | air \| Ag 25 nm \| SiO2 60 nm \| Ag | 0.00–1.00 | **A up to 1.00** | 4.1e-4 |
| bragg_L4 … bragg_L8 | 4–8 | (H L)… on glass, n_H = 2.3, n_L = 1.45 | max R 0.34 → 0.85 | −1.6 … −0.5 / −8.6 … −7.0 | 3e-4 … 2e-2 / 2e-9 … 1e-7 |
| bragg_L8_conformal | 8 | all interfaces corrugated | 0.02–0.85 | 0.1 / −7.5 | 1.4e-2 / 3e-8 |
| graded_L5, graded_L8 | 5, 8 | graded index 1.1 → 1.5 | ≤ 0.043 | −3 / −10 | 5e-6, 1.7e-5 |
| hmm_L8, hmm_L8_top (hyperbolic metamaterial) | 8 | (Ag 0.15 \| 2.3 0.3)… | 0.02–0.98 | A 0.02–0.16 | 2e-3 / 5e-9 |
| silver_L2_ref (paper map, L = 2) | 2 | 1 \| Ag | 0–1.04 | A −0.04–1.00 | 4.3e-2 (same as the 2D paper map) |

The Bragg L = 8 peak R = 0.848 matches the quarter-wave formula (0.847).

**About the max error.** The maximum error always sits at a window edge next to a Rayleigh anomaly
(e.g. ω ≈ 1.97–2.0), exactly as in the 2-layer paper. The median is 1e-7 – 1e-13.

- Shrinking the windows fixes it. For Bragg L = 8, max_delta = 0.1 / 0.025 / 0.0125 gives a max error of
  3.3e-3 / 1.3e-3 / 8.7e-4, while the median falls from 1.3e-7 to 4.7e-11.
- The pointwise reference is converged at those points (N = 16 … 40 give identical values).

## 5. What changes as layers are added (`layer_study.py` → `results/layer_study.md`, `figures/layer_study/`)

The study covers four families for L = 2 … 8: Bragg (top grating), Bragg (conformal), graded index, and a
metal/dielectric HMM. All use band q = 1, ε ≤ 0.2, joint windows and Padé.

* **The reflectivity map.**
  - The Bragg stop band builds up as theory predicts: max R_flat = 0.04, 0.31, 0.34, 0.63, 0.66, 0.83, 0.85
    for L = 2 … 8.
  - The grating lowers R at ε = 0.15 by 2–4 % and adds resonance lines (guided modes of the mirror) inside
    the band.
  - The graded stack drives R towards 0 (≤ 0.003 at L ≥ 6).
  - The HMM shows coupled plasmon bands.
* **Radius of convergence in δ** (median, root test on the specular coefficients):

  | family | L = 2 | L = 3 | L = 4 | L = 5 | L = 6 | L = 7 | L = 8 |
  |---|---|---|---|---|---|---|---|
  | Bragg | 5.5 | 1.3 | 1.0 | 0.40 | 0.37 | 0.20 | 0.19 |
  | graded | 5.8 | 2.3 | 1.2 | 1.0 | 0.71 | 0.59 | 0.53 |
  | HMM | 5.8 | 4.1 | 1.2 | 0.72 | 0.70 | 0.53 | 0.41 |

  Every extra layer brings Fabry–Pérot, guided or coupled-plasmon poles closer to the real frequency axis.
  This is the main new effect: **the frequency windows must shrink with L**. |δ| ≤ 0.1 is safe up to L = 8,
  and Padé recovers what Taylor loses.
* **Radius of convergence in ε.** It falls from about 1.2 (L = 2) to about 0.5 (L = 8) for top-only
  corrugation of thin layers. The hard limit is the TFE singularity **ε\* = d / max(f_b − f_t)**: the
  interfaces of a layer must not cross. `refl_map_nlayer` refuses ε_max ≥ ε\* with an error rather than a
  warning. **Conformal corrugation** (f_b = f_t) removes the bound: the Bragg ε-radius rises from 0.47 to 0.61.
* **The energy defect** of the lossless maps: the median log10|D| goes from −11.6 (L = 2) to about −8
  (L = 8 Bragg), and stays at −11 for the graded stack. Its maximum tracks the Rayleigh-edge points.
* **Accuracy.** The true maximum error grows from 2e-6 (L = 2) to 1e-3 – 2e-2 (Bragg L ≥ 7, at the Rayleigh
  edges). The graded family stays at 1e-7 – 1e-8 and the HMM at about 1e-5.

## 6. Movies (`refl_movie_nlayer.py` → `figures/movies/`)

Each frame shows:

- the R map with a cursor;
- the log10|D| or absorptance map;
- the R/T/A spectrum along the sweep;
- **Re u_tot through the whole stack** over two periods: the Padé sum of the volume series in every layer
  plus the incident wave, with the interfaces drawn;
- the intensity profile ⟨|u|²⟩_x(z).

All coefficients are computed once per window, so each frame is only a Padé sum.

| preset | stack | what to see |
|---|---|---|
| gmr | air \| TiO2 120 nm \| silica | R jumps to about 0.9 at the guided-mode resonance; the field is trapped in the film |
| silver_film | vacuum \| Ag \| vacuum (IMI) | coupled surface plasmons on both faces |
| mim | air \| Ag \| SiO2 \| Ag | perfect absorption (A → 1) near 0.53 µm with the field concentrated in the 60 nm gap |
| bragg | (H L)³ on glass, L = 8 | stop band (R → 0.85), field decays into the mirror |
| hmm | Ag/dielectric ×3, L = 8 | hyperbolic bands, field decay through the metal layers |
| film_eps | dielectric film | ε sweeps 0 → 0.2 at a fixed wavelength |

## 7. Material survey (`material_survey_nlayer.py` → `results/material_survey_*.md`, `figures/material_survey/`)

All parts now use **exact material dispersion** (§10.1). The first version of Parts A and B, which froze the indices
per window, is kept in `results/material_survey_frozen/`. The median change in the survey metrics is 0.01. The
largest changes are 0.15, in Si R_max and in the Ag/K/Na resonance depths.

**Part A: 54 materials as a 50 nm film on fused silica (L = 3) versus the same material in bulk (L = 2).**
All runs use a conformal cos x grating with period 0.4 µm, λ = 0.4 – 1.0 µm and ε ≤ 0.2.

* High-index dielectrics and semiconductors **reflect much more as thin films** because of thin-film
  Fabry–Pérot fringes: Si 0.06–0.88 vs 0.31–0.49 in bulk, and GaP 0.10–0.66 vs 0.27–0.38.
* Good metals (Au, Cu, Pt, W) behave almost like bulk at 50 nm. The alkali metals and Ag do not:
  - K: A 0.04 vs 0.07 mean, 0.15 vs 0.94 max.
  - Ag_IR: max A 0.36 vs 0.86.
* Grating resonances (min R/R_flat) are **deeper as a film** for Ag, Al, Mg, Ag_IR, TiO2, diamond, GaN, ZnS
  and SiC. They are shallower for Na, K and GaP.
* AZO absorbs much less as a 50 nm film (0.27 vs 0.71).

**Part B: 90 Bragg mirrors, air | (H L)^P | BK7, with P = 1, 2, 3 (L = 4, 6, 8).**

* R_flat(ω₀) now agrees with the quarter-wave formula to **≤ 1.3e-4** for all 90 stacks. The frozen-index
  version was off by 1.4e-2.
* The stop-band FWHM narrows with P (about 0.9 → 0.55 → 0.4) towards (4/π) arcsin(contrast), with Spearman
  0.78 at P = 3.
* The grating lowers R inside the band by 2–7 % (median min R/R_flat 0.96).

**Parts C – E (new):**

* **10 nm absorbing film on a mirror** (14 films × Au/Ag/Al/Cu, L = 3; `absorbers_on_mirrors.png`, which
  includes the reflected colours). Near-perfect absorbers:
  - Ge/Cu and Ge/Au: A = 1.000 / 0.999 at 626 nm;
  - GaAs/Au: 0.999 at 512 nm;
  - InP/Ag: 0.997 at 439 nm;
  - Si/Cu: 0.999 at 489 nm.

  On Ag nearly all of the absorption is in the film (97 %), while on Au/Cu most of it is in the metal. Lossy
  metals (Cr, Ti, W, Pt, Ni) give broadband grey absorbers, which the ε = 0.2 grating raises from about 0.3
  to about 0.7.
* **Tamm plasmons for 15 metals** (30 nm on the TiO2/SiO2 mirror, L = 8; `tamm_metals.png`):
  - Na (R = 0.03 at 752 nm), Au (0.07), Cu (0.12) and Ag (0.18) give the deepest Tamm dips.
  - Transmission dominates for Na/Ag. Absorption dominates for TiN/Cr/Ti/Ni/Pd.
  - A dip reported at 0.778 µm, the edge of the range, means that no Tamm state was found.
* **Metal–insulator–metal absorbers** (4 metals × 6 spacers, L = 4; `mim_metal_spacer.png`):
  - Ag/SiO2 is a **grating-enabled perfect absorber**: A = 0.154 flat → **0.993** at ε = 0.12, at 530 nm
    (gap plasmon).
  - Au and Cu stacks already absorb about 0.98 flat (Fabry–Pérot absorber).

## 8. Scripts and layout

```
hops_nl/solver.py            Stack, multilayer_solve (coupled multi-layer HOPS/AWE recursion), energies
hops_nl/mms.py               manufactured exact solutions for any L
hops_nl/reference.py         TMM, pointwise HOPS reference, check_map (true error of a map)
refl_map_nlayer.py           maps (scenarios or CLI: --layers 1 1.5 Ag 1.0 --thick 0.6 0.25 --profiles cosx cosx flat)
mms_error_nlayer.py          MMS presets L3_fig2, L3, L4, L5, L6, L8, L8_oblique, convergence
test_single_eps_delta_nlayer.py  --run 1 2 3  (L = 3, 5, 8)
test_mms_error_nlayer.py     PASS/FAIL driver;  tests/test_nlayer.py (pytest)
run_all_maps.py              all scenarios + results/maps_summary.md (true error column)
layer_study.py               L = 2 … 8 study (radii, errors, cost)
refl_movie_nlayer.py         --preset gmr silver_film mim bragg hmm film_eps  tamm lrspp kretschmann ge_au vo2 perovskite cavity berreman
material_survey_nlayer.py    --part films bragg absorbers tamm mim
explore_materials_nlayer.py  materials gallery L = 3 ... 9: maps, true error, frozen-vs-exact dispersion, per-layer absorption
deep_dives_nlayer.py         --study tamm lrspp sensor vo2 ge_au cavity solar contrast
bench_nlayer.py              timing vs L -> results/bench.md
hops_nl/absorption.py        absorption in every layer from the volume series (TE/TM, curved layers, substrate tail)
hops_nl/summation_ext.py     eps-only (M = 0) summation and energies for the exact-dispersion mode
hops_nl/materials_ext.py     VO2 transition (Bruggeman VO2_f10 ... VO2_f90) and hBN (in-plane) models
figures/{mms_error, test_single_eps_delta, refl_map/L2..L8, layer_study, movies, material_survey, explore/L3..L9, deep_dives}
results/{maps_summary.md, layer_study.md, material_survey_*.md|csv, bench.md, explore_summary.md, deep_dives.md,
         maps/, layer_study/, material_survey/, explore/, deep_dives/, material_survey_frozen/, logs/}
```

In PyCharm, open the parent folder (containing both `HOPS_Python` and `HOPS_NLayer`). Every script also runs
directly with the green ▶ button; edit the `SCENARIO` / `PRESET` constants at the top of each file.

## 9. Limitations

* The TFE needs non-crossing interfaces: ε·max(amp_b f_b − amp_t f_t) < d for every layer. Use conformal
  profiles for thin layers.
* In the default HOPS/AWE mode (`dispersion='window'`), dispersive indices are frozen at each window's centre,
  as in the 2-layer code. For real metals this can be off by O(0.1–1) in R near plasmon resonances (§10.1).
  Use `dispersion='exact'`, which costs about 5–10× more.
* Mid-IR polar crystals (SiC film on Au at 10 µm) were tried and dropped. The ε-series converges poorly next
  to the Rayleigh anomaly that the surface phonon polariton sits on (true error about 4e-2 even at ε ≤ 0.05).
* Only the refractiveindex.info pages bundled with HOPS_Python (62 pages) were available. This session could
  not download more of the database, so VO2 in transition and hBN use literature models (`hops_nl/materials_ext.py`),
  and AlAs uses n = 2.95.
* For L ≥ 7 Bragg-type stacks with strong resonances, use max_delta ≤ 0.05 near Rayleigh anomalies, or check
  with `reference.check_map`.
* The Taylor sum stalls at about 1e-6 for L ≥ 5 and N = M ≥ 12 (round-off), so use Padé.
* Only 2D (1D-periodic) gratings are covered. The 3D extension is `HOPS_AWE_PINN_3D` / `HOPS_Python` 3D.

## 10. Materials gallery L = 3 … 9: multilayer analogues of the 2D material scenarios

`explore_materials_nlayer.py` → `figures/explore/L<L>/` and `results/explore_summary.md`;
`deep_dives_nlayer.py` → `figures/deep_dives/` and `results/deep_dives.md`.

The 2-layer study found several material regimes: SPP lines on Ag/Na/K, SPR sensing with water over gold,
ENZ ITO, phase-change VO2 and high-index Wood anomalies. Their multilayer analogues are:

| 2-layer regime | multilayer scenario (L) | what the extra layers add |
|---|---|---|
| SPP lines on Ag | `lrspp_ag10/20/40`, `imi_asym_ag20` (3) | the two faces' SPPs couple into **long-range and short-range** plasmons |
| water over gold (SPR) | `kretschmann_au` (3), `kretschmann_cr/ti` (4), `biosensor_L4` (4) | prism coupling at a fixed 68°, adhesion layers, a protein layer |
| ENZ (ITO) | `berreman_ito` (3) | a thin ENZ film gives the **Berreman** absorption peak at λ_ENZ = 1.26 µm |
| phase change (VO2) | `vo2_cold/f30/hot_sapphire` (3) | VO2 film on sapphire: **perfect absorption mid-transition** |
| lossy metals / semiconductors | `ge_on_au` (3), `salisbury` (4) | ultrathin-film interference absorbers and colours |
| high-index dielectrics | `ir_mirror_L6/L8` (Ge/ZnS), `vcsel_dbr_L8`, `microcavity_L9` | high-contrast mirrors, a cavity |
| plasmonic + dielectric | `tamm_ag/au` (8), `mim_filter` (5), `hmm_au_tio2_L6/L8` | **Tamm plasmons**, Fabry–Pérot filters, metal/dielectric metamaterials |
| solar | `perovskite_cell`, `si_cell` (4) | absorption in the active layer vs the parasitic layers |

### 10.1 New numerical capabilities (needed for real materials)

* **Exact material dispersion** (`dispersion='exact'`, the default for every gallery scenario).
  - Each frequency column is solved with HOPS in ε only (M = 0, Padé in ε) using n_l(λ). This costs 0.05 s
    (L = 3) to 0.2 s (L = 9) per frequency.
  - The frozen-index HOPS/AWE map (`dispersion='window'`) differs from it by a **median of 0.001 – 0.035 and a
    maximum of 0.01 – 1.8** in R (column "AWE with frozen indices" of `results/explore_summary.md`).
  - Near plasmon resonances the frozen indices produce visible steps at the window joints. For dispersive
    metals the AWE frequency expansion therefore needs either very short windows or exact dispersion.
  - A *dispersive* AWE, which would expand n_l(ω) in δ inside the solver, is the natural next step.
* **Fixed incidence angle** (`theta=`).
  - The paper's convention keeps α fixed inside a window, so the angle jumps from window to window.
  - The new mode uses α = n_top(λ) ω sin θ at every frequency, which matches the TMM Kretschmann curve to 1e-9.
  - Windows are cut at the Rayleigh frequencies p / (n_top(1 ± sin θ)) of the top layer and at those of the
    substrate.
* **Oblique incidence fix.** With α ≠ 0 the top layer's Rayleigh points |α − p| / n_top fall *inside* the
  paper's bands. They are now window cuts.
* **Absorption in every layer** (`hops_nl/absorption.py`).
  - Computed as ∫ Im(ε)|E|² over the curved TFE layers, with Clenshaw–Curtis quadrature in s and the exact
    Jacobian.
  - The analytic tail of an absorbing substrate is included.
  - It equals TMM layer by layer and sums to 1 − R − T to 1e-9.
* Mid-transition VO2 (Bruggeman) and in-plane hBN models.

### 10.2 Gallery highlights (exact dispersion, true error = AWE/HOPS vs converged pointwise HOPS)

| scenario | L | result | true error max |
|---|---|---|---|
| ge_on_au | 3 | 10 nm Ge on Au: **R = 0.000 at 625 nm** (85 % absorbed in the Ge) | 5e-11 |
| vo2_f30_sapphire | 3 | 30 % metallic VO2: **R = 0.005 at 12.17 µm** (A_VO2 = 0.91) | 3e-12 |
| kretschmann_au | 3 | SPR at 0.717 µm with R = 0.01; the grating out-couples the SPP into water orders | 6e-9 |
| berreman_ito | 3 | ITO absorption peak 0.37 at **1.255 µm = λ_ENZ** | 1e-12 |
| salisbury | 4 | Cr 6 nm / SiO2 / Al: R = 0.01 at 0.655 µm, 94 % absorbed in 6 nm of Cr | 3e-11 |
| tamm_au | 8 | Tamm dip at 0.705 µm (R 0.07); at ε = 0.2 the hybrid Tamm–SPP mode reaches **R = 0.01 at 0.686 µm** | 4e-5 |
| tamm_ag | 8 | Tamm dip at 0.684 µm (TMM 0.684) | 1e-4 |
| mim_filter | 5 | Ag/SiO2/Ag transmission filter at 0.506 µm | 3e-5 |
| microcavity_L9 | 9 | cavity mode at 0.700 µm, T = 0.958 (the air/BK7 mismatch limit 4n/(1+n)²) | 6e-8 |
| ir_mirror_L8 | 8 | Ge/ZnS mirror, R → 1.000 in the 3–5.5 µm stop band; lossless, so D measures accuracy (median 10^-12.5) | 7e-4 |

### 10.3 Deep dives (`results/deep_dives.md`)

* **Tamm plasmon × surface plasmon** (`tamm_spp_anticrossing.png`, L = 8).
  - The flat Tamm state is at 0.684 µm.
  - Scanning the period moves the grating-coupled Ag/air SPP through it. The two resonances **anticross**,
    and the anticrossing widens with ε.
  - At the crossing (P = 0.67 µm) the hybrid dip reaches **critical coupling: R = 0.017 at ε = 0.1**, against
    0.17 for the flat stack. It over-couples beyond that (R = 0.24 at ε = 0.2).
* **Long- and short-range SPPs** (silica | Ag t | silica, `lrspp_thickness.png`).
  - The HOPS dips agree with a flat-film IMI mode solver to ≤ 2 nm for t = 8–60 nm. Examples: t = 20 nm gives
    LR 0.7351 vs 0.7331 µm and SR 0.8447 vs 0.8451 µm; t = 60 nm gives 0.7510/0.7644 vs 0.7504/0.7645 µm.
  - The SR branch red-shifts strongly as the film thins: 0.764 → 0.943 µm going from 60 to 12 nm.
  - The LR branch narrows to a 2.8 nm FWHM at 8 nm.
* **Grating-assisted Kretschmann sensor** (`sensor_kretschmann.png`; analyte 1.333 → 1.343, 68°, TM):

  | stack | sensitivity S (nm/RIU) | FOM |
  |---|---|---|
  | Au 50 nm | 5150 | 90 |
  | Ag 50 nm | 5350 | **147**, best |
  | Cu 50 nm | 4970 | 79 |

  - 2 nm Cr or Ti adhesion layers broaden the dip (57 → 63–64 nm FWHM) and lower the FOM by about 10 %.
  - A 10 nm protein layer shifts the resonance by +35 nm.
  - A grating with ε = 0.1 shifts the dip by +3 nm and costs 2–13 % of FOM, except for Cu. It radiates part of
    the SPP into the water, so the grating does not improve a Kretschmann sensor.
* **VO2 through the insulator–metal transition** (`vo2_transition.png`). Peak absorptance runs
  0.93 (cold) → 0.83 (20 %) → **0.995 at 12.14 µm (30 % metallic)** → 0.79 (50 %) → 0.36 (hot). This is the
  near-perfect absorption at an intermediate state that Kats et al. (2012) measured. The 8 µm grating matters
  only below 8 µm, once diffraction opens.
* **Ge on Au** (`ge_on_au_colours.png`).
  - The absorption peak red-shifts linearly with thickness: 543 → 825 nm for 5 → 25 nm.
  - It is perfect (A = 1.000) at 10 nm.
  - Reflected colours run from gold (bare) through magenta and violet to teal (10 nm) and grey. The grating
    adds an SPP dip at λ ≈ P = 0.5 µm.
* **Microcavity** (L = 5, 9, 13: up to 13 layers).
  - Q = 4 → 17 → 61, about (n_H/n_L)² per added pair.
  - The grating shifts the mode by only −0.4 nm and raises T by 1.5 %.
* **Solar cells** (per-layer absorption).
  - Perovskite (300 nm): 77 % mean absorption in MAPbI3. The grating changes it by < 1 %, and parasitic
    absorption in the ITO is 2.6 %.
  - Thin c-Si (300 nm on Ag): the grating raises Si absorption from 10.1 % to **15.2 % (+50 %) at ε = 0.2**,
    through guided-mode resonances near the band edge. Parasitic absorption in the Ag goes from 1.5 to 2.4 %.
* **Accuracy vs index contrast** (lossless (H L)³ mirrors, HOPS/AWE with |δ| ≤ 0.1):

  | n_H | median true error | max true error |
  |---|---|---|
  | 1.6 | 6e-11 | 6e-6 |
  | 2.4 | 4e-8 | 2e-2 |
  | 3.0 – 4.0 | 4e-7 – 4e-6 | 0.3 – 0.9 |

  The max errors sit at window edges next to Rayleigh points, where guided-mode poles crowd in. **High-contrast
  stacks (Si/Ge-like) need short windows or the exact pointwise mode.**

### 10.4 New movies (`figures/movies/`, spectrum panel = per-layer absorption stacked)

`tamm` (Tamm–SPP hybrid), `lrspp`, `kretschmann`, `ge_au`, `vo2` (30 % metallic VO2), `perovskite`, `cavity`
(L = 9) and `berreman`.

# Deep dives with the full refractiveindex.info database (deep_dives_db_nlayer.py)

### WS2 monolayer exciton-polaritons (L = 15)
* exciton 2.031 eV (0.6105 um); flat cavity polaritons at normal incidence: 1.994 eV, 2.038 eV; splitting at the smallest cavity-exciton detuning: **2g = 44 meV** (the monolayer's background index also red-shifts the cavity mode, so the two branches are not symmetric about the exciton).
* HOPS with the 1 um grating: normal-incidence dips at eps = 0: 2.037 eV (R 0.50), 1.993 eV (R 0.19); at eps = 0.2: 2.037 eV (R 0.38), 1.994 eV (R 0.09).  True error of the HOPS map 2.6e-06.
(277 s)

### SQIB dye in a silver cavity (L = 6): ultrastrong coupling
* exciton 1.931 eV (0.642 um); at zero detuning (PMMA spacers 75 nm) the polaritons are at 1.663 eV, 2.218 eV: **splitting 555 meV = 29 % of the exciton energy** (ultrastrong coupling regime > 10 %).
* HOPS at the design (175 nm cavity): absorbed in SQIB (spectral max) 0.51 flat -> 0.64 at eps = 0.2; in Ag 0.04 -> 0.29.
(41 s)

### Ge2Sb2Te5 phase-change stacks (L = 5 colour pixel, L = 3 mid-IR absorber)
* colour pixels: the reflected colour changes with crystallisation for every ITO spacer (figures/deep_dives/db_gst_switching.png); the grating (eps 0.2) adds a small hue shift.
* mid-IR: best switching contrast for GST 1000 nm on Au: R 0.99 (amorphous) -> 0.001 (crystalline) at 2.55 um.  HOPS (0.6 um, P 1.5 um): R_min crystalline flat 0.015, eps 0.2 0.025; amorphous 0.87 / 0.39.
| GST (nm) | lambda (um) | R amorphous | R crystalline |
|---|---|---|---|
| 200 | 4.70 | 0.960 | 0.307 |
| 300 | 2.43 | 0.990 | 0.201 |
| 400 | 2.04 | 0.945 | 0.023 |
| 500 | 3.74 | 0.990 | 0.074 |
| 600 | 2.11 | 0.976 | 0.017 |
| 700 | 5.10 | 0.990 | 0.017 |
| 800 | 5.79 | 0.990 | 0.005 |
| 900 | 2.90 | 0.921 | 0.001 |
| 1000 | 2.55 | 0.990 | 0.001 |

(150 s)

### VO2 thermochromic window through the measured temperature series (L = 5)
* flat: T_lum 0.532 -> 0.559, T_sol 0.505 -> 0.494 (**dT_sol = 1.1 %**), NIR 0.598 -> 0.358 from 20 to 80 C.
* grating eps = 0.2: T_lum 0.528 -> 0.555, dT_sol = 1.4 %.
| T (C) | T_lum flat | T_sol flat | T_lum eps 0.2 | T_sol eps 0.2 |
|---|---|---|---|---|
| 20 | 0.532 | 0.505 | 0.528 | 0.504 |
| 30 | 0.535 | 0.507 | 0.531 | 0.505 |
| 40 | 0.539 | 0.509 | 0.536 | 0.507 |
| 50 | 0.547 | 0.511 | 0.543 | 0.510 |
| 55 | 0.702 | 0.600 | 0.699 | 0.598 |
| 60 | 0.635 | 0.560 | 0.631 | 0.557 |
| 70 | 0.562 | 0.494 | 0.557 | 0.490 |
| 80 | 0.559 | 0.494 | 0.555 | 0.490 |

(solar weighting: 5778 K black body; luminous: CIE y-bar)  (412 s)

### Liquid-crystal (5CB) tunable silver cavity, TE (L = 4)
* cavity dips with the director along z (n_o): 0.642 um (R 0.35), 0.797 um (R 0.80), 1.057 um (R 0.39), 1.220 um (R 0.99); along y (n_e): 0.709 um (R 0.69), 0.879 um (R 0.81), 1.104 um (R 1.00), 1.163 um (R 0.12).
* grating eps = 0.2 splits the modes and adds grating-coupled guided-mode dips (TE: no surface plasmons): n_o 0.639, 0.788, 0.809, 1.057; n_e 0.705, 0.869, 0.890, 1.163 um.
(66 s)

### Thermoplasmonics: Kretschmann sensor with hot metals (L = 3, measured temperature pages)
| metal | lambda_res (um) | R_min | FWHM (nm) | S (nm/RIU) | FOM |
|---|---|---|---|---|---|
| Au 25 C | 0.7155 | 0.010 | 56.4 | 4677 | 82.9 |
| Au 225 C | 0.7131 | 0.000 | 60.1 | 4676 | 77.8 |
| Au 350 C | 0.7087 | 0.018 | 62.1 | 4726 | 76.1 |
| Ag 298 K | 0.6499 | 0.035 | 62.4 | 5739 | 91.9 |
| Ag 404 K | 0.6402 | 0.040 | 59.4 | 5529 | 93.1 |
| Ag 501 K | 0.6352 | 0.055 | 59.9 | 5740 | 95.8 |
| Ag 600 K | 0.6361 | 0.053 | 59.2 | 5501 | 92.9 |

(251 s)

### Very many layers: polymer Bragg mirrors, L = 12 ... 52 (HOPS/AWE, lossless)
| pairs | L | matrix | s / window | log10\|D\| max / median | true error max / median | flat R vs TMM | max R_flat |
|---|---|---|---|---|---|---|---|
| 5 | 12 | 156 | 15.3 | -3.5 / -5.1 | 6.6e-06 / 1.0e-10 | 1.5e-13 | 0.246 |
| 10 | 22 | 286 | 7.0 | -2.5 / -4.8 | 1.2e-04 / 8.0e-08 | 1.3e-09 | 0.499 |
| 15 | 32 | 416 | 11.8 | -1.4 / -4.6 | 1.9e-03 / 5.0e-06 | 2.8e-07 | 0.703 |
| 20 | 42 | 546 | 18.9 | -0.8 / -4.4 | 2.7e-03 / 1.2e-04 | 3.1e-05 | 0.835 |
| 25 | 52 | 676 | 19.2 | -0.7 / -4.0 | 1.2e-02 / 1.3e-03 | 1.5e-04 | 0.912 |
| 30 | 62 | 806 | 36.5 | 0.5 / -3.3 | 1.6e-02 / 2.5e-03 | 1.4e-03 | 0.954 |

(399 s)

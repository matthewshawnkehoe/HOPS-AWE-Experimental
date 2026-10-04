# Deep dives (deep_dives_nlayer.py)

### Tamm plasmon x surface plasmon (tamm_ag, L = 8)
* flat Tamm state at **0.6840 um** (TMM, R = 0.172); the Ag/air SPP (first grating order) crosses it at period **0.670 um**.
* the period scans (figures/deep_dives/tamm_spp_anticrossing.png) show the grating-coupled SPP line running into the Tamm line and the two **anticrossing**; the repulsion grows with eps (at eps = 0.2 the Tamm branch is pushed to 0.70-0.72 um where the SPP crosses it). (The automatic dip-pair separation in tamm.json is only indicative: near the crossing the two dips merge into one broad hybrid dip.)
* at the crossing the hybrid dip reaches **critical coupling** (R = 0.017 at eps = 0.1); R at the dip: eps 0.00: 0.174, eps 0.05: 0.121, eps 0.10: 0.017, eps 0.15: 0.056, eps 0.20: 0.242
* (1329 s)

### Long- and short-range surface plasmons (silica | Ag | silica, L = 3)
| Ag (nm) | HOPS dips at eps = 0.05 | flat-film mode solver: lambda = P Re n_eff |
|---|---|---|
| 8 | 0.7295 um (R 0.25, FWHM 2.8 nm) | LR 0.7284, SR 1.0631 |
| 12 | 0.7313 um (R 0.46, FWHM 24.5 nm); 0.9428 um (R 0.54, FWHM 9.3 nm) | LR 0.7296, SR 0.9436 |
| 16 | 0.7333 um (R 0.60, FWHM 153.1 nm); 0.8804 um (R 0.60, FWHM 8.1 nm) | LR 0.7312, SR 0.8809 |
| 20 | 0.7351 um (R 0.69, FWHM 115.4 nm); 0.8447 um (R 0.62, FWHM 6.0 nm) | LR 0.7331, SR 0.8451 |
| 30 | 0.7385 um (R 0.77, FWHM 3.7 nm); 0.7990 um (R 0.61, FWHM 3.6 nm) | LR 0.7383, SR 0.7993 |
| 40 | 0.7436 um (R 0.36, FWHM 2.0 nm); 0.7797 um (R 0.70, FWHM 4.6 nm) | LR 0.7433, SR 0.7795 |
| 60 | 0.7510 um (R 0.66, FWHM 2.2 nm); 0.7644 um (R 0.45, FWHM 2.3 nm) | LR 0.7504, SR 0.7645 |

(194 s)

### Grating-assisted Kretschmann sensor (BK7 prism, 68 deg, TM; analyte 1.333 -> 1.343)
| stack | flat: lambda_res (um) / R_min / FWHM (nm) / S (nm/RIU) / FOM | grating eps = 0.1: same |
|---|---|---|
| Au 50 nm | 0.7295 / 0.009 / 57.0 / 5147 / 90.2 | 0.7327 / 0.017 / 59.0 / 5194 / 88.1 |
| Ag 50 nm | 0.6256 / 0.046 / 36.4 / 5347 / 146.9 | 0.6297 / 0.020 / 42.3 / 5392 / 127.5 |
| Cu 50 nm | 0.7163 / 0.028 / 62.6 / 4970 / 79.4 | 0.7194 / 0.068 / 63.8 / 5084 / 79.7 |
| Al 20 nm | 0.8851 / 0.245 / 112.6 / 77 / 0.7 | 0.8851 / 0.248 / 111.2 / 80 / 0.7 |
| Cr 2 nm + Au 48 nm | 0.7317 / 0.012 / 63.4 / 5173 / 81.6 | 0.7349 / 0.015 / 64.2 / 5223 / 81.4 |
| Ti 2 nm + Au 48 nm | 0.7318 / 0.009 / 64.3 / 5176 / 80.5 | 0.7350 / 0.015 / 65.1 / 5227 / 80.3 |
| Au 50 nm + protein 10 nm | 0.7641 / 0.005 / 59.7 / 5327 / 89.2 | 0.7674 / 0.018 / 63.0 / 5381 / 85.5 |

(577 s)

### VO2 on sapphire through the phase transition (L = 3, Bruggeman mix)
| metallic fraction | peak A flat (lambda um) | peak A at eps = 0.2 (lambda um) |
|---|---|---|
| 0 % | 0.926 (9.98) | 0.931 (9.98) |
| 10 % | 0.873 (10.03) | 0.881 (10.03) |
| 20 % | 0.831 (11.69) | 0.832 (11.69) |
| 30 % | 0.995 (12.14) | 0.995 (12.14) |
| 40 % | 0.941 (12.45) | 0.945 (12.39) |
| 50 % | 0.787 (12.20) | 0.797 (12.20) |
| 60 % | 0.647 (11.75) | 0.681 (8.01) |
| 70 % | 0.541 (11.63) | 0.689 (8.01) |
| 80 % | 0.462 (11.50) | 0.705 (8.01) |
| 90 % | 0.402 (11.38) | 0.724 (8.01) |
| 100 % | 0.356 (11.38) | 0.745 (8.01) |

Maximum absorption 0.995 at 12.14 um for 30 % metallic domains. (283 s)

### Ultrathin Ge on Au: absorption and colour (L = 3)
| Ge (nm) | peak absorptance flat (lambda um) | peak A at eps = 0.2 |
|---|---|---|
| 0 | 0.598 (0.470) | 0.727 |
| 5 | 0.817 (0.543) | 0.842 |
| 7 | 0.938 (0.597) | 0.939 |
| 10 | 1.000 (0.624) | 1.000 |
| 15 | 0.946 (0.685) | 0.953 |
| 20 | 0.879 (0.754) | 0.892 |
| 25 | 0.827 (0.825) | 0.846 |

(186 s)

### Microcavity: Q vs mirror pairs, and the grating (L = 5, 9, 13)
| p | L | lambda_c (um) | FWHM (nm) | Q | cavity peak at eps = 0 / 0.1 / 0.2 (lambda um, T) |
|---|---|---|---|---|---|
| 0 | 5 | 0.7000 | 184.30 | 4 | 0.7002, 0.958 / 0.6998, 0.960 / 0.6995, 0.965 |
| 1 | 9 | 0.7000 | 40.00 | 17 | 0.7002, 0.958 / 0.6998, 0.962 / 0.6995, 0.972 |
| 2 | 13 | 0.7000 | 11.55 | 61 | 0.7002, 0.958 / 0.6998, 0.962 / 0.6998, 0.972 |

(1108 s)

### Thin-film solar cells: where the light is absorbed (per-layer absorption from the volume series)
* **perovskite_cell**: active-layer absorptance (spectral mean) eps 0.00: 0.771, eps 0.05: 0.772, eps 0.10: 0.773, eps 0.15: 0.776, eps 0.20: 0.779; parasitic ITO: 0.026 -> 0.026; Ag: 0.003 -> 0.003
* **si_cell**: active-layer absorptance (spectral mean) eps 0.00: 0.101, eps 0.05: 0.106, eps 0.10: 0.118, eps 0.15: 0.135, eps 0.20: 0.152; parasitic Ag: 0.015 -> 0.024
(117 s)

### HOPS/AWE accuracy vs index contrast (lossless (H L)^3 mirrors, L = 8)
| n_H | contrast | max R_flat | log10\|D\| max / median | true error max / median |
|---|---|---|---|---|
| 1.6 | 0.049 | 0.217 | -3.4 / -9.7 | 5.6e-06 / 6.3e-11 |
| 2.0 | 0.159 | 0.682 | -2.1 / -8.3 | 2.1e-04 / 4.6e-09 |
| 2.4 | 0.247 | 0.880 | -0.6 / -7.5 | 2.3e-02 / 4.1e-08 |
| 3.0 | 0.348 | 0.967 | 0.3 / -6.6 | 2.8e-01 / 3.7e-07 |
| 3.5 | 0.414 | 0.987 | 0.4 / -6.2 | 2.7e-01 / 1.1e-06 |
| 4.0 | 0.468 | 0.994 | 1.0 / -5.7 | 8.7e-01 / 3.8e-06 |

(247 s)

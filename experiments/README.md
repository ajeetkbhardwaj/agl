# Real-World Experiments

One self-contained script per problem. Each script downloads nothing by
itself — place the data under `data/` using the links below (commands
included per script docstring). Run from the repository root:

```
/opt/miniconda3/envs/aisystem/bin/python experiments/<script>.py
```

| # | Script | Data (source) | Modules exercised | Headline result |
|---|--------|---------------|-------------------|-----------------|
| 1 | `kepler_law.py` | NASA Exoplanet Archive `pscomppars` (TAP CSV) → `data/exoplanets.csv` | `PolynomialNeuralNetwork` (`activation='none'`), symbolic distillation | Discovers Kepler's third law from 5586 planets: R² = 0.9903, effective exponent 1.43–1.54 (theory 1.5); exact equation matches network to 1e-06 |
| 2 | `co2_trend.py` | NOAA GML Mauna Loa monthly CO₂ → `data/co2_mlo.csv` | `OrthoPolyNetwork` (Chebyshev-first basis, rank 3) | In-window R² = 0.999162 (RMSE 0.63 ppm); out-of-window predictions freeze at a clamp plateau — quantified extrapolation limit; seasonal amplitude 7.03 vs 7.80 ppm raw |
| 3 | `sunspots.py` | SILSO monthly sunspots → `data/sunspots.txt` | `OrthoPolyNetwork` (3 bases) vs `RationalPolyLayer`, persistence baseline | At horizon 6 months RationalPolyLayer wins (RMSE 0.5013, corr 0.893) vs persistence 0.5946 — localized poles capture cycle shape; at horizon 1 persistence wins (honesty baseline) |
| 4 | `concrete_strength.py` | UCI #165 Concrete Compressive Strength → `data/concrete/Concrete_Data.xls` | `PolynomialNeuralNetwork` distillation, `OrthoPolyNetwork`, `get_polynomial_weights` spectral importance | Additive OrthoPoly R² = 0.8916 (5.06 MPa); exact-equation distillation error 5.2e-07; importance ranking `age ≫ superpl > flyash > …` matches curing physics |
| 5 | `power_plant.py` | UCI #294 Combined Cycle Power Plant → `data/ccpp/CCPP/Folds5x2_pp.xlsx` | `OrthoPolyNetwork` additive vs multiplicative (Segre) modes, PyTorch MLP baseline | Additive OrthoPoly R² = 0.9432 / RMSE 4.055 MW — beats a 64×64 tanh MLP (4.301 MW) at published benchmark level; multiplicative mode does not help on this smooth response |
| 6 | `airfoil_noise.py` | UCI #291 Airfoil Self-Noise (NASA) → `data/airfoil/airfoil_self_noise.dat` | `PolynomialNeuralNetwork` + `create_groebner_pnn` hard constraints | Distilled law exact to 5.2e-07; ideal `<x0 - x1>` enforced exactly (`w_freq = 0` after every step) costs ΔR² = 0.123 — algebra rejects "frequency has no linear effect" |

## Data download commands

```bash
# 1. Exoplanets (Kepler)
curl -L "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_orbper,pl_orbsmax+from+pscomppars+where+pl_orbper+is+not+null+and+pl_orbsmax+is+not+null&format=csv" -o data/exoplanets.csv

# 2. Mauna Loa CO2
curl -L "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv" -o data/co2_mlo.csv

# 3. Sunspots
curl -L "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt" -o data/sunspots.txt

# 4. Concrete
curl -L "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip" -o data/concrete.zip && unzip -o data/concrete.zip -d data/concrete

# 5. Power plant
curl -L "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip" -o data/ccpp.zip && unzip -o data/ccpp.zip -d data/ccpp

# 6. Airfoil
curl -L "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip" -o data/airfoil.zip && unzip -o data/airfoil.zip -d data/airfoil
```

All results above were produced by actually running the scripts against
the downloaded data (seeds fixed: `torch.manual_seed(7)`,
`np.random.RandomState(42)` splits).

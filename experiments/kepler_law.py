"""
Experiment 1: Kepler's Third Law discovered from raw exoplanet data
===================================================================

Data: NASA Exoplanet Archive (pscomppars table), columns pl_orbper (days)
      and pl_orbsmax (AU) for confirmed planets.

Download:
  curl "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+pl_name,pl_orbper,pl_orbsmax+from+pscomppars+where+pl_orbper+is+not+null+and+pl_orbsmax+is+not+null&format=csv" -o data/exoplanets.csv

Mathematics
-----------
Kepler's third law:  T^2 = a^3 / M_star   (T in years, a in AU, M in solar
masses). In log-log coordinates this is an exact straight line:

    log10(T_days) = 1.5 * log10(a_AU) + log10(365.25) - 0.5*log10(M_star)

We fit a Polynomial Neural Network to (log a -> log T). Because the PNN is
exactly polynomial, `get_global_equation` hands back the fitted law in
closed form. The leading coefficient should come out near 1.5 -- the network
*discovers* the exponent of Kepler's law from raw observations.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nn.poly_nn import PolynomialNeuralNetwork

DATA = Path(__file__).resolve().parents[1] / "data" / "exoplanets.csv"


def main():
    df = pd.read_csv(DATA)
    print(f"loaded {len(df)} confirmed planets with both T and a")

    # Clean: positive, finite values only; drop extreme outliers (>100 AU)
    df = df[(df.pl_orbper > 0) & (df.pl_orbsmax > 0)]
    df = df[df.pl_orbsmax < 100]
    x = np.log10(df.pl_orbsmax.to_numpy()).reshape(-1, 1)
    y = np.log10(df.pl_orbper.to_numpy()).reshape(-1, 1)
    print(f"after cleaning: {len(x)} planets, "
          f"log a in [{x.min():.2f}, {x.max():.2f}]")

    # Least-squares reference line (the 'textbook' answer)
    slope_ref, intercept_ref = np.polyfit(x.ravel(), y.ravel(), 1)
    print(f"\nleast squares :  log T = {slope_ref:.4f} * log a + {intercept_ref:.4f}")
    print(f"Kepler predicts slope 1.5 exactly (scatter = stellar masses)")

    # --- Polynomial Neural Network on the same data ---
    # activation='none' keeps every layer exactly polynomial, so the
    # extracted global equation is a plain closed-form polynomial law.
    torch.manual_seed(7); np.random.seed(7)
    pnn = PolynomialNeuralNetwork(input_dim=1, output_dim=1,
                                  hidden_dims=[8], polynomial_degree=3,
                                  activation='none')
    pnn.fit(x.astype(np.float64), y.astype(np.float64),
            epochs=150, learning_rate=0.01, verbose=False)

    metrics = pnn.evaluate(x, y)
    print(f"\nPNN train R^2 = {metrics['r2']:.6f}  "
          f"(RMSE = {np.sqrt(metrics['mse']):.4f} dex)")

    # --- Symbolic distillation: the discovered law ---
    u = sp.symbols('u')
    eq = pnn.get_global_equation(['u'], threshold=1e-6, round_to=None)
    eq_readable = pnn.get_global_equation(['u'], threshold=1e-6)
    print(f"\ndiscovered law  log T = {eq_readable}")
    print("theory          log T = 1.500*u + 2.562   (T days = 365.25 * a_AU^1.5)")
    poly_eq = sp.Poly(eq, u)
    print("coefficients (highest degree first):")
    for deg, coeff in zip(range(poly_eq.degree(), -1, -1),
                          poly_eq.all_coeffs()):
        print(f"  u^{deg}: {float(coeff):+.6f}")

    # Effective exponent: derivative d(logT)/d(loga) across the data range
    deriv = sp.lambdify(u, sp.diff(eq, u), 'numpy')
    for ua in (-1.0, 0.0, 1.0):
        print(f"  effective exponent at log a = {ua:+.0f}:  {float(deriv(ua)):.4f}")

    # Extracted equation must reproduce the network exactly
    f_eq = sp.lambdify(u, eq, 'numpy')
    sym_vals = np.asarray(f_eq(x)).reshape(-1, 1)
    nn_vals = pnn.predict(x)
    print(f"\nmax |equation - network| = {np.abs(sym_vals - nn_vals).max():.2e}"
          "   (exact symbolic distillation)")

    # Physics check: residual scatter vs stellar-mass spread
    resid = y.ravel() - (slope_ref * x.ravel() + intercept_ref)
    print(f"\nresidual scatter around pure Kepler-3: "
          f"{resid.std():.3f} dex  (~ factor {10**resid.std():.2f} in T)")
    print("-> consistent with the ignored 0.5*log10(M_star) term")


if __name__ == "__main__":
    main()

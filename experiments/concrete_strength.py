"""
Experiment 4: Concrete compressive strength -- an interpretable mix-design law
=============================================================================

Data: UCI Machine Learning Repository #165 (Yeh 1998), 1030 concrete mixes,
      8 ingredients/age features -> compressive strength (MPa).

Download:
  curl -L "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip" \
       -o data/concrete.zip && unzip -o data/concrete.zip -d data/concrete

Mathematics
-----------
Two questions:
  1. Can a polynomial network match black-box accuracy AND hand back a
     closed-form engineering formula?  -> PolynomialNeuralNetwork +
     get_global_equation, verified numerically against forward().
  2. Which inputs matter?  -> OrthoPolyNetwork in additive mode: the
     spectral coefficient norms per input channel are an exact,
     architecture-native importance ranking.

Inputs are standardized; the extracted equation lives in standardized
coordinates u_i = (x_i - mean)/std (table printed for engineering use).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nn.poly_nn import PolynomialNeuralNetwork
from src.nn.orthopoly_nn import OrthoPolyNetwork

DATA = Path(__file__).resolve().parents[1] / "data" / "concrete" / "Concrete_Data.xls"
SHORT = ["cement", "slag", "flyash", "water", "superpl", "coarse",
         "fine", "age"]


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def main():
    df = pd.read_excel(DATA)
    df.columns = SHORT + ["strength"]
    X = df[SHORT].to_numpy().astype(np.float64)
    y = df["strength"].to_numpy().astype(np.float64).reshape(-1, 1)
    print(f"loaded {X.shape[0]} mixes, {X.shape[1]} features")

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]

    mu_x, sd_x = X[tr].mean(0), X[tr].std(0)
    mu_y, sd_y = y[tr].mean(), y[tr].std()
    Xs = (X - mu_x) / sd_x
    ys = (y - mu_y) / sd_y

    # ---------- 1. PNN: accuracy + closed-form extraction ----------
    torch.manual_seed(7); np.random.seed(7)
    pnn = PolynomialNeuralNetwork(input_dim=8, output_dim=1,
                                  hidden_dims=[16], polynomial_degree=2,
                                  activation='tanh')
    pnn.fit(Xs[tr], ys[tr], epochs=250, learning_rate=0.01, verbose=False)

    pred_te = pnn.predict(Xs[te]) * sd_y + mu_y
    print(f"\nPNN (tanh) test R^2 = {r2(y[te], pred_te):.4f}   "
          f"RMSE = {np.sqrt(np.mean((y[te] - pred_te)**2)):.2f} MPa")

    names = [f"u{i}_{n}" for i, n in enumerate(SHORT)]
    eq_pruned = pnn.get_global_equation(names, threshold=5e-3)
    print(f"\ndiscovered mix-design law (standardized coords, pruned):\n"
          f"  strength_z = {str(eq_pruned)[:300]}")

    eq_exact = pnn.get_global_equation(names, threshold=1e-9, round_to=None)
    f_eq = sp.lambdify(sp.symbols(" ".join(names)), eq_exact, "numpy")
    sym_vals = np.asarray(f_eq(*[Xs[te][:, i] for i in range(8)]),
                          dtype=float).reshape(-1, 1)
    nn_vals = pnn.predict(Xs[te])
    print(f"max |exact equation - network| on test set = "
          f"{np.abs(sym_vals - nn_vals).max():.2e}  (exact distillation)")

    # ---------- 2. Additive OrthoPoly net: spectral importance ----------
    torch.manual_seed(7); np.random.seed(7)
    opn = OrthoPolyNetwork(input_dim=8, output_dim=1, hidden_dims=[24],
                           max_degree=4, rank=4, basis_type="chebyshev_first")
    opn.fit(Xs[tr], ys[tr], epochs=250, learning_rate=0.01, verbose=False)
    pred2 = opn.predict(Xs[te]) * sd_y + mu_y
    print(f"\nOrthoPoly (additive) test R^2 = {r2(y[te], pred2):.4f}   "
          f"RMSE = {np.sqrt(np.mean((y[te] - pred2)**2)):.2f} MPa")

    # In additive mode the first layer IS a sum of univariate contributions:
    #   y = sum_i p_i(u_i) + ...
    # Extract each channel polynomial exactly and measure its spread over
    # the actual test data -- an architecture-native importance ranking.
    opn.eval()
    polys = opn.layers[0].get_polynomial_weights()[0]     # output 0 -> [p_i]
    print("\nper-channel contribution spread on test data "
          "(std of p_i(u_i)):")
    from src.alggeom.polynomial import Variable
    contribs = []
    for i, poly in enumerate(polys):
        var = Variable(f"x{i}")               # channel polynomials use x_i
        vals = np.array([float(poly.evaluate({var: v}))
                         for v in Xs[te][:, i]])
        contribs.append(vals.std())
        print(f"  {SHORT[i]:<9} std={vals.std():.3f}  "
              f"mean={vals.mean():+.3f}")
    order = np.argsort(-np.asarray(contribs))
    print("ranking:", " > ".join(SHORT[i] for i in order))


if __name__ == "__main__":
    main()

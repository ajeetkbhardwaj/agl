"""
Experiment 6: Airfoil self-noise -- scaling laws and algebraic constraints
=========================================================================

Data: UCI Machine Learning Repository #291 (NASA, Brooks & Marcolini).
      1503 wind-tunnel measurements: frequency (Hz), angle of attack (deg),
      chord length (m), free-stream velocity (m/s), suction-side
      displacement thickness (m)  ->  scaled sound pressure (dB).

Download:
  curl -L "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip" \
       -o data/airfoil.zip && unzip -o data/airfoil.zip -d data/airfoil

Mathematics
-----------
Part A -- symbolic scaling law: a PNN is trained on standardized features
and distilled into one closed-form equation, verified against forward().

Part B -- algebraic constraints as testable hypotheses: a GroebnerLayer
network trained under the ideal <x0 - x1>. After every optimizer step the
first-layer weights are replaced by their normal form modulo the ideal,
(w0, w1) -> (0, w0 + w1): frequency's linear sensitivity is forced to be
absorbed into the angle-of-attack channel -- the hypothesis "frequency
carries no independent linear effect" is enforced EXACTLY by algebra, not
by a soft penalty. If test accuracy barely drops, the hypothesis is
consistent with the data; if it collapses, the physics rejects it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nn.poly_nn import PolynomialNeuralNetwork, create_groebner_pnn

DATA = Path(__file__).resolve().parents[1] / "data" / "airfoil" / "airfoil_self_noise.dat"
SHORT = ["freq", "alpha", "chord", "vel", "thickness"]


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def main():
    df = pd.read_csv(DATA, sep=r"\s+", header=None, names=SHORT + ["ssp"])
    X = df[SHORT].to_numpy().astype(np.float64)
    y = df["ssp"].to_numpy().astype(np.float64).reshape(-1, 1)
    print(f"loaded {len(X)} wind-tunnel measurements")

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0)
    mu_y, sd_y = y[tr].mean(), y[tr].std()
    Xs, ys = (X - mu_x) / sd_x, (y - mu_y) / sd_y

    # ---------- Part A: symbolic scaling law ----------
    torch.manual_seed(7); np.random.seed(7)
    pnn = PolynomialNeuralNetwork(input_dim=5, output_dim=1,
                                  hidden_dims=[12], polynomial_degree=2,
                                  activation='tanh')
    pnn.fit(Xs[tr], ys[tr], epochs=250, learning_rate=0.01, verbose=False)
    pred = pnn.predict(Xs[te]) * sd_y + mu_y
    print(f"\n[A] PNN test R^2 = {r2(y[te], pred):.4f}   "
          f"RMSE = {np.sqrt(np.mean((y[te] - pred)**2)):.3f} dB")

    names = [f"u{i}_{n}" for i, n in enumerate(SHORT)]
    eq = pnn.get_global_equation(names, threshold=5e-3)
    print(f"distilled law (standardized coords):\n  ssp_z = {str(eq)[:260]}")
    eq_exact = pnn.get_global_equation(names, threshold=1e-9, round_to=None)
    f_eq = sp.lambdify(sp.symbols(" ".join(names)), eq_exact, "numpy")
    sym = np.asarray(f_eq(*[Xs[te][:, i] for i in range(5)]),
                     dtype=float).reshape(-1, 1)
    print(f"max |exact equation - network| = "
          f"{np.abs(sym - pnn.predict(Xs[te])).max():.2e}")

    # ---------- Part B: Groebner hard constraint ----------
    # ideal <x0 - x1>: after every step, (w_freq, w_alpha) -> (0, w_freq+w_alpha)
    torch.manual_seed(7); np.random.seed(7)
    gp = create_groebner_pnn(
        5, 1, constraint_ideals=[["x0 - x1"], []],
        hidden_dims=[12], polynomial_degree=2, activation='tanh')
    gp.fit(Xs[tr], ys[tr], epochs=250, learning_rate=0.01,
           verbose=False)

    pred_c = gp.predict(Xs[te]) * sd_y + mu_y
    print(f"\n[B] Groebner-constrained network test R^2 = "
          f"{r2(y[te], pred_c):.4f}   "
          f"RMSE = {np.sqrt(np.mean((y[te] - pred_c)**2)):.3f} dB")

    W = gp.layers[0].weight_matrix.detach().numpy()
    print(f"constraint check |w_freq| = {abs(W[0,0]):.2e}   "
          f"(normal form: frequency weight exactly zero)")
    drop = r2(y[te], pred) - r2(y[te], pred_c)
    verdict = ("hypothesis consistent with data"
               if abs(drop) < 0.02 else "physics rejects the hypothesis")
    print(f"R^2 drop from hard constraint: {drop:+.4f}  -> {verdict}")


if __name__ == "__main__":
    main()

"""
Experiment 5: Combined Cycle Power Plant -- additive vs multiplicative modes
============================================================================

Data: UCI Machine Learning Repository #294 (Tufekci 2014; Kaya et al. 2019).
      9568 sensor readings from a combined-cycle power plant over 6 years:
      ambient temperature AT, exhaust vacuum V, ambient pressure AP,
      relative humidity RH  ->  net electrical output PE (MW).

Download:
  curl -L "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip" \
       -o data/ccpp.zip && unzip -o data/ccpp.zip -d data/ccpp

Mathematics
-----------
The physical law is smooth and dominated by temperature, with interaction
effects (humidity matters more when it is hot). We compare:

  1. OrthoPolyNetwork additive mode       y = sum_i f_i(x_i)   per layer
  2. OrthoPolyNetwork multiplicative mode rank-1 cross terms across inputs
     (Segre embedding) -- interactions at exponential expressive gain
  3. a plain PyTorch MLP of comparable parameter count

All models see standardized features; test RMSE in MW is the score.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nn.orthopoly_nn import OrthoPolyNetwork

DATA = Path(__file__).resolve().parents[1] / "data" / "ccpp" / "CCPP" / "Folds5x2_pp.xlsx"


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main():
    df = pd.read_excel(DATA)
    X = df[["AT", "V", "AP", "RH"]].to_numpy().astype(np.float64)
    y = df["PE"].to_numpy().astype(np.float64).reshape(-1, 1)
    print(f"loaded {len(X)} plant readings")

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    n_tr = int(0.8 * len(X))
    tr, te = idx[:n_tr], idx[n_tr:]
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0)
    mu_y, sd_y = y[tr].mean(), y[tr].std()
    Xs = ((X - mu_x) / sd_x)
    ys = ((y - mu_y) / sd_y)

    results = {}

    torch.manual_seed(7); np.random.seed(7)
    add = OrthoPolyNetwork(input_dim=4, output_dim=1, hidden_dims=[16],
                           max_degree=4, rank=4, basis_type="chebyshev_first",
                           interaction_mode="additive")
    add.fit(Xs[tr], ys[tr], epochs=200, learning_rate=0.01, verbose=False)
    results["OrthoPoly additive       "] = add.predict(Xs[te]) * sd_y + mu_y

    torch.manual_seed(7); np.random.seed(7)
    mul = OrthoPolyNetwork(input_dim=4, output_dim=1, hidden_dims=[16],
                           max_degree=3, rank=4, basis_type="chebyshev_first",
                           interaction_mode="multiplicative")
    mul.fit(Xs[tr], ys[tr], epochs=200, learning_rate=0.01, verbose=False)
    results["OrthoPoly multiplicative "] = mul.predict(Xs[te]) * sd_y + mu_y

    # Plain MLP baseline: ~ same spirit as hidden [64, 64]
    torch.manual_seed(7); np.random.seed(7)
    mlp = nn.Sequential(nn.Linear(4, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(),
                        nn.Linear(64, 1))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    Xtr_t = torch.tensor(Xs[tr], dtype=torch.float32)
    ytr_t = torch.tensor(ys[tr], dtype=torch.float32)
    Xte_t = torch.tensor(Xs[te], dtype=torch.float32)
    for epoch in range(300):
        opt.zero_grad()
        loss = nn.functional.mse_loss(mlp(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        results["MLP 64x64 baseline     "] = \
            mlp(Xte_t).numpy() * sd_y + mu_y

    print(f"\n{'model':<31} {'test RMSE (MW)':>14} {'test R^2':>10}")
    for name, pred in results.items():
        print(f"{name:<31} {rmse(y[te], pred):14.3f} "
              f"{r2(y[te], pred):10.4f}")

    # Interaction evidence: does humidity matter more when hot?
    hot = X[:, 0] > np.percentile(X[:, 0], 90)
    cold = X[:, 0] < np.percentile(X[:, 0], 10)
    rh_med = float(np.median(X[:, 3]))
    dPE_dRH_hot = (y[hot & (X[:, 3] > rh_med)].mean() -
                   y[hot & (X[:, 3] <= rh_med)].mean())
    dPE_dRH_cold = (y[cold & (X[:, 3] > rh_med)].mean() -
                    y[cold & (X[:, 3] <= rh_med)].mean())
    print(f"\nhumidity effect on output: {dPE_dRH_hot:+.2f} MW (hot days) vs "
          f"{dPE_dRH_cold:+.2f} MW (cold days)  -> interaction present"
          if abs(dPE_dRH_hot - dPE_dRH_cold) > 1 else
          "\nno strong temperature-humidity interaction in raw data")


if __name__ == "__main__":
    main()

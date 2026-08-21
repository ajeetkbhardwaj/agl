"""
Experiment 2: Mauna Loa CO2 -- trend + seasonality with orthogonal polynomials
==============================================================================

Data: NOAA Global Monitoring Laboratory, monthly mean CO2 at Mauna Loa
      (1958 - present), file co2_mm_mlo.csv.

Download:
  curl "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv" -o data/co2_mlo.csv

Mathematics
-----------
The record is modeled as  c(t) = trend(t) + seasonal(t)  where the trend is a
smooth function captured by a Chebyshev expansion (adaptive domain
normalization maps the time window to [-1, 1]) and the annual cycle by exact
harmonic features sin(2*pi*t), cos(2*pi*t):

    input features:  [ t,  sin(2*pi*t),  cos(2*pi*t) ]

We train on 1958-2009 and test on 2010-2026. Two honest findings:
  * inside the training window the fit is essentially perfect;
  * outside it, the layer's clamp freezes normalized inputs at +-1, so
    extrapolation flattens -- orthogonal expansions interpolate, they do not
    extrapolate. Quantified below.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nn.orthopoly_nn import OrthoPolyNetwork

DATA = Path(__file__).resolve().parents[1] / "data" / "co2_mlo.csv"


def features(t):
    """t: decimal date -> [t, sin(2 pi t), cos(2 pi t)]"""
    w = 2 * np.pi * (t % 1.0)
    return np.column_stack([t, np.sin(w), np.cos(w)]).astype(np.float64)


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def main():
    df = pd.read_csv(DATA, comment="#")
    df = df[df["average"] > 0]                      # drop missing (-99.99)
    t = df["decimal date"].to_numpy()
    y = df["average"].to_numpy().astype(np.float64)
    print(f"loaded {len(df)} monthly means, "
          f"{t.min():.1f} .. {t.max():.1f}, "
          f"{y[0]:.1f} .. {y[-1]:.1f} ppm")

    X = features(t)
    train = t < 2010.0
    Xtr, ytr = X[train], y[train]
    Xte, yte = X[~train], y[~train]
    print(f"train window : {int(t[train].min())}-{int(t[train].max())} "
          f"({train.sum()} months)")
    print(f"test window  : {int(t[~train].min())}-{int(t[~train].max())} "
          f"({(~train).sum()} months)")

    # Center/scale targets so the network starts near the data level
    mu, sd = float(ytr.mean()), float(ytr.std())
    torch.manual_seed(7); np.random.seed(7)
    net = OrthoPolyNetwork(input_dim=3, output_dim=1, hidden_dims=[16],
                           max_degree=5, rank=3, basis_type="chebyshev_first")
    net.fit(Xtr, ((ytr - mu) / sd).reshape(-1, 1), epochs=300,
            learning_rate=0.01, verbose=False)

    def predict(X):
        return net.predict(X).ravel() * sd + mu

    pred_tr, pred_te = predict(Xtr), predict(Xte)
    print(f"\nin-window  R^2 = {r2(ytr, pred_tr):.6f}   "
          f"RMSE = {np.sqrt(np.mean((ytr - pred_tr)**2)):.3f} ppm")
    print(f"out-window R^2 = {r2(yte, pred_te):.6f}   "
          f"RMSE = {np.sqrt(np.mean((yte - pred_te)**2)):.3f} ppm")

    # --- The clamp story: predictions flatten beyond the training window ---
    print("\nyear   actual   predicted")
    for year in (1990, 2000, 2005, 2010, 2015, 2020, 2024):
        mask = (t >= year) & (t < year + 1)
        p = float(predict(features(np.array([year + 0.55])))[0])
        a = float(y[mask].mean())
        print(f"{year}   {a:7.2f}   {p:7.2f}")

    # Seasonal cycle recovered by the network: freeze the year, scan phase
    tt = 2000.0
    phases = np.linspace(0.0, 1.0, 24, endpoint=False)
    preds = np.array([float(predict(features(np.array([tt + ph])))[0])
                      for ph in phases])
    print(f"\nseasonal peak-to-trough predicted for {int(tt)}: "
          f"{preds.max() - preds.min():.2f} ppm")

    detrended = ytr - pd.Series(ytr).rolling(13, center=True,
                                             min_periods=1).mean().to_numpy()
    print(f"seasonal peak-to-trough in raw train data:      "
          f"{detrended.max() - detrended.min():.2f} ppm")


if __name__ == "__main__":
    main()

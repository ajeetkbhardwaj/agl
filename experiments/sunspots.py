"""
Experiment 3: Sunspot cycle forecasting -- Hermite vs Chebyshev vs Rational
==========================================================================

Data: SILSO monthly mean total sunspot number, 1749 - present
      (file SN_m_tot_V2.0.txt, whitespace-separated).

Download:
  curl "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.txt" -o data/sunspots.txt

Mathematics
-----------
Autoregressive forecasting: predict SN(t+1) from the last L = 24 monthly
values. The sunspot series is bounded (0 .. ~500) and spiky -- cycles rise
fast and decay slowly. Three function approximators compete:

  1. OrthoPolyNetwork with chebyshev_first basis  (bounded domain [-1,1])
  2. OrthoPolyNetwork with hermite basis          (unbounded domain R)
  3. a single RationalPolyLayer                   (Pade-style P/(1+|Q|),
                                                   localized poles for spikes)

Train on the first 70% of history, test on the most recent 30%.
Report test RMSE and lag-1 forecast correlation.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nn.orthopoly_nn import OrthoPolyNetwork
from src.layers.rational_layer import RationalPolyLayer

DATA = Path(__file__).resolve().parents[1] / "data" / "sunspots.txt"
LAGS = 24


def make_dataset(series, lags=LAGS, horizon=1):
    """Predict the value `horizon` months after the last input month."""
    X, y = [], []
    for i in range(len(series) - lags - horizon + 1):
        X.append(series[i:i + lags])
        y.append(series[i + lags + horizon - 1])
    return np.asarray(X), np.asarray(y).reshape(-1, 1)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def main():
    rows = [line.split() for line in
            DATA.read_text().strip().splitlines()]
    dates = np.array([float(r[2]) for r in rows])
    sn = np.array([float(r[3]) for r in rows])
    keep = sn >= 0                                   # -1 marks missing
    dates, sn = dates[keep], sn[keep]
    print(f"loaded {len(sn)} monthly values, {dates.min():.1f}..{dates.max():.1f}")

    # Scale to zero-mean/unit-variance for training
    mu, sd = sn.mean(), sn.std()
    z = (sn - mu) / sd

    split = int(0.7 * len(z))

    def run_benchmark(horizon):
        X, y = make_dataset(z, horizon=horizon)
        Xtr, ytr = X[:split], y[:split]
        Xte, yte = X[split:], y[split:]
        print(f"\n=== horizon: {horizon} month(s) ahead | "
              f"train {len(ytr)} | test {len(yte)} ===\n")

        results = {}

        torch.manual_seed(7); np.random.seed(7)
        net_cheb = OrthoPolyNetwork(input_dim=LAGS, output_dim=1,
                                    hidden_dims=[24], max_degree=4, rank=4,
                                    basis_type="chebyshev_first")
        net_cheb.fit(Xtr, ytr, epochs=200, learning_rate=0.005, verbose=False)
        results["OrthoPoly chebyshev_first"] = net_cheb.predict(Xte)

        torch.manual_seed(7); np.random.seed(7)
        net_herm = OrthoPolyNetwork(input_dim=LAGS, output_dim=1,
                                    hidden_dims=[24], max_degree=4, rank=4,
                                    basis_type="hermite")
        net_herm.fit(Xtr, ytr, epochs=200, learning_rate=0.005, verbose=False)
        results["OrthoPoly hermite        "] = net_herm.predict(Xte)

        torch.manual_seed(7); np.random.seed(7)
        rational = RationalPolyLayer(input_dim=LAGS, output_dim=1,
                                     max_degree=3, rank=6)
        opt = torch.optim.Adam(rational.parameters(), lr=0.005)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32)
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.mean((rational(Xtr_t) - ytr_t) ** 2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            results["RationalPolyLayer         "] = \
                rational(Xte_t).numpy()

        naive = Xte[:, -1:]                           # persistence forecast

        print("model                          test RMSE (scaled)  corr")
        print(f"{'persistence (last value)':<31} {rmse(yte, naive):.4f}          "
              f"{np.corrcoef(yte.ravel(), naive.ravel())[0,1]:.4f}")
        for name, pred in results.items():
            r = rmse(yte, pred)
            c = np.corrcoef(yte.ravel(), pred.ravel())[0, 1]
            print(f"{name:<31} {r:.4f}          {c:.4f}")
        return yte, results

    run_benchmark(1)
    yte6, results6 = run_benchmark(6)

    # Cycle-amplitude check on the strongest recent maximum (6-month horizon)
    i_max = int(np.argmax(yte6))
    best = min(results6.items(), key=lambda kv: rmse(yte6, kv[1]))
    print(f"\nstrongest test-month peak (6-mo horizon): actual "
          f"{yte6[i_max,0]*sd+mu:.0f}, best-model ({best[0].strip()}) "
          f"prediction {best[1][i_max,0]*sd+mu:.0f}")


if __name__ == "__main__":
    main()

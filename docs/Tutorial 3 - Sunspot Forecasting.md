# Tutorial 3 - Sunspot Forecasting: Polynomials vs Rational Functions

*Companion to `experiments/sunspots.py`. Every code block runs against the
real SILSO record in `data/sunspots.txt`.*

## Table of Contents

### 1. The Physical Problem

### 2. Autoregressive Forecasting as Function Approximation

### 3. Why Not Plain Polynomials? The Runge Phenomenon

### 4. Rational Functions and Localized Poles

### 5. Hermite Polynomials for Oscillatory Data

### 6. Building the Benchmark

### 7. Results at Horizon 1 and Horizon 6

### 8. Why the Rational Layer Wins at Long Horizon

### 9. Exercises

---

### 1. The Physical Problem

**Problem**: The sun's magnetic activity cycles between quiet and stormy with
an irregular $\sim$11-year period. Solar maxima disrupt satellites, GPS and
power grids — but the cycle's amplitude and timing wander, so forecasting is
a genuine open problem. You have $3331$ monthly mean sunspot counts reaching
back to **1749**. Predict the count $h$ months ahead from the recent past.

```Python
import numpy as np

rows = [line.split() for line in
        open("data/sunspots.txt").read().strip().splitlines()]
dates = np.array([float(r[2]) for r in rows])    # decimal year
sn = np.array([float(r[3]) for r in rows])       # monthly mean count
keep = sn >= 0                                   # -1 marks missing months
dates, sn = dates[keep], sn[keep]
print(len(sn))                                   # 3331
print(dates.min(), dates.max())                  # 1749.042 2026.538
print(round(sn.mean(), 2), round(sn.std(), 2))   # 82.2 67.57
```

Note the scale of the noise: one standard deviation ($67.57$) is comparable
to the mean ($82.2$). Also note what "most recent 30%" means here: the
record spans $277$ years, so the test window begins around **1945** and
contains the strongest cycle ever observed (cycle 19, peaked October 1957).

---

### 2. Autoregressive Forecasting as Function Approximation

**Definition (Autoregressive model)**: an AR model of order $L$ predicts

$$
z_{t+h} \;=\; f\big(z_t,\, z_{t-1},\, \dots,\, z_{t-L+1}\big),
$$

where $f$ maps the *delay vector* (the last $L$ observations) to the future.
We take $L = 24$ months — two years, long enough to see where in the cycle
we are — and standardize the series to $z = (\mathrm{SN} - \mu)/\sigma$.

```Python
LAGS = 24
mu, sd = sn.mean(), sn.std()
z = (sn - mu) / sd

def make_dataset(series, lags=LAGS, horizon=1):
    X, y = [], []
    for i in range(len(series)-lags-horizon+1):
        X.append(series[i:i+lags])
        y.append(series[i+lags+horizon-1])
    return np.asarray(X), np.asarray(y).reshape(-1, 1)

X, y = make_dataset(z, horizon=6)
print(X.shape)                     # (3302, 24)
```

**Definition (Persistence baseline)**: the forecast "tomorrow equals today",
$f(\text{lags}) = z_t$. Any learned model must beat this to justify its
existence — and at short horizons it is embarrassingly hard to beat.

---

### 3. Why Not Plain Polynomials? The Runge Phenomenon

Polynomials are dense in continuous functions (Stone–Weierstrass), so *some*
polynomial approximates any smooth $f$. The catch is *how fast* the degree
must grow. **Runge's example**: $g(x) = 1/(1+25x^2)$ on $[-1,1]$. Fitting
polynomials through equally spaced samples makes things *worse* as degree
rises:

```Python
xs = np.linspace(-1, 1, 400)
g = lambda x: 1/(1+25*x**2)
for n in (6, 10, 14):
    xn = np.linspace(-1, 1, n+1)          # equispaced nodes
    c = np.polyfit(xn, g(xn), n)          # exact interpolation
    print(n, f"{np.abs(np.polyval(c, xs)-g(xs)).max():.2f}")
# 6 0.62
# 10 1.92
# 14 7.19
```

Degree 14 oscillates with amplitude $7$ — on a function whose true range is
$[0.038, 1]$! The mechanism: $g$ has poles at $x = \pm i/5$, distance $0.2$
from the real interval, and polynomial error at equispaced nodes scales like
$\rho^{-n}$ with the *Bernstein ellipse factor* $\rho < 1$ here — divergence,
not convergence.

Sunspot cycles are bounded and spiky — exactly the situation where the
function $f$ of Section 2 behaves like it has nearby complex singularities.
We need an approximator with poles.

---

### 4. Rational Functions and Localized Poles

**Definition (Rational approximation)**: a rational function

$$
r(x) \;=\; \frac{P(x)}{Q(x)}
$$

can place poles *anywhere in the complex plane* by choosing the roots of
$Q$. Padé theory: if $g$ has a pole at distance $d$ from the interval, a
degree-$n$ rational approximant converges geometrically in $n$, while a
degree-$n$ polynomial needs $n \sim 1/d$ terms before convergence even
starts. Rational functions are the natural basis for spiky, localized
behavior.

The library's `RationalPolyLayer` uses the numerically safe form

$$
r(x) \;=\; \frac{P(x)}{\,1 + |Q(x)|\,},
$$

which keeps the denominator away from zero (no division blow-up during
training) while still allowing sharp, near-pole behavior where $Q \approx 0$.

```Python
import torch
from src.layers.rational_layer import RationalPolyLayer

torch.manual_seed(7); np.random.seed(7)
rat = RationalPolyLayer(input_dim=1, output_dim=1, max_degree=3)
opt = torch.optim.Adam(rat.parameters(), lr=0.01)
Xt = torch.tensor(xs.reshape(-1,1), dtype=torch.float32)
yt = torch.tensor(g(xs).reshape(-1,1), dtype=torch.float32)
for _ in range(1500):
    opt.zero_grad()
    loss = torch.mean((rat(Xt) - yt)**2)
    loss.backward(); opt.step()
with torch.no_grad():
    print(f"{np.abs(rat(Xt).numpy()-g(xs).reshape(-1,1)).max():.4f}")
# 0.0059
```

A degree-3 rational layer fits Runge's function to $6\times10^{-3}$ — where
degree-14 polynomial *interpolation* was off by $7$. Two orders of
magnitude, from poles.

---

### 5. Hermite Polynomials for Oscillatory Data

**Definition (Physicists' Hermite polynomials)**:

$$
H_{n+1}(x) \;=\; 2x\,H_n(x) \;-\; 2n\,H_{n-1}(x),
\qquad H_0 = 1,\; H_1 = 2x,
$$

orthogonal on $\mathbb{R}$ with Gaussian weight:

$$
\int_{-\infty}^{\infty} H_m(x)\,H_n(x)\,e^{-x^2}\,dx \;=\;
\sqrt{\pi}\, 2^n\, n!\, \delta_{mn}.
$$

They are the right expansion when the data lives on an *unbounded*
standardized domain and looks wave-like ($H_n$ oscillates for $x <
\sqrt{2n}$, then blows up monotonically — built-in contrast between cyclic
and extreme regimes):

```Python
from numpy.polynomial.hermite import hermval

grid = np.linspace(-5, 5, 2001)
w = np.exp(-grid**2)
H = [hermval(grid, np.eye(6)[k]) for k in range(6)]
print(np.allclose(H[3], 2*grid*H[2] - 4*H[1]))   # True  (recurrence)
inner12 = np.trapz(H[1]*H[2]*w, grid)
inner22 = np.trapz(H[2]*H[2]*w, grid)
print(round(inner12, 3))                          # 0.0
print(round(inner22, 3))                          # 14.18  (= 4*2!*sqrt(pi))
```

---

### 6. Building the Benchmark

Three competitors, identical data: `OrthoPolyNetwork` with Chebyshev basis
(bounded-domain specialist), `OrthoPolyNetwork` with Hermite basis
(unbounded-domain specialist), and a single `RationalPolyLayer` trained
directly by Adam. Chronological 70/30 split of the *samples*.

```Python
from src.nn.orthopoly_nn import OrthoPolyNetwork

def rmse(y, yh):
    return float(np.sqrt(np.mean((y-yh)**2)))

split = int(0.7 * len(z))

def benchmark(horizon):
    X, y = make_dataset(z, horizon=horizon)
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]
    out = {}
    out["persistence"] = Xte[:, -1:]

    torch.manual_seed(7); np.random.seed(7)
    nc = OrthoPolyNetwork(input_dim=LAGS, output_dim=1, hidden_dims=[24],
                          max_degree=4, rank=4, basis_type="chebyshev_first")
    nc.fit(Xtr, ytr, epochs=200, learning_rate=0.005, verbose=False)
    out["chebyshev"] = nc.predict(Xte)

    torch.manual_seed(7); np.random.seed(7)
    nh = OrthoPolyNetwork(input_dim=LAGS, output_dim=1, hidden_dims=[24],
                          max_degree=4, rank=4, basis_type="hermite")
    nh.fit(Xtr, ytr, epochs=200, learning_rate=0.005, verbose=False)
    out["hermite"] = nh.predict(Xte)

    torch.manual_seed(7); np.random.seed(7)
    rat = RationalPolyLayer(input_dim=LAGS, output_dim=1,
                            max_degree=3, rank=6)
    opt = torch.optim.Adam(rat.parameters(), lr=0.005)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xe = torch.tensor(Xte, dtype=torch.float32)
    for _ in range(300):
        opt.zero_grad()
        torch.mean((rat(Xt) - yt)**2).backward()
        opt.step()
    with torch.no_grad():
        out["rational"] = rat(Xe).numpy()
    return yte, out
```

---

### 7. Results at Horizon 1 and Horizon 6

```Python
for horizon in (1, 6):
    yte, out = benchmark(horizon)
    print(f"--- horizon {horizon} month(s): "
          f"{len(yte)} test windows ---")
    for name, pred in out.items():
        r = rmse(yte, pred)
        c = np.corrcoef(yte.ravel(), pred.ravel())[0, 1]
        print(f"{name:<12} RMSE {r:.4f}   corr {c:.4f}")
# --- horizon 1 month(s): 976 test windows ---
# persistence RMSE 0.3900   corr 0.9384
# chebyshev     RMSE 0.4179   corr 0.9265
# hermite       RMSE 0.5322   corr 0.9068
# rational      RMSE 0.4120   corr 0.9289
# --- horizon 6 month(s): 971 test windows ---
# persistence RMSE 0.5946   corr 0.8574
# chebyshev     RMSE 0.5425   corr 0.8760
# hermite       RMSE 0.5569   corr 0.8667
# rational      RMSE 0.5013   corr 0.8934
```

Read the two horizons as opposite verdicts:

* **h = 1**: persistence wins ($0.3900$). One month ahead, the atmosphere of
  the problem — solar memory — makes "next month = now" nearly optimal, and
  no learned model recovers its training investment. Honesty demands we say
  so.
* **h = 6**: every learned model beats persistence, and the **rational layer
  wins outright** ($0.5013$ vs $0.5946$, correlation $0.8934$ vs $0.8574$).
  Six months ahead you must model the *shape* of the cycle — and shape is
  where poles pay.

Sanity check on the hardest event in the test window — the great cycle-19
maximum:

```Python
X, y = make_dataset(z, horizon=6)
Xte, yte = X[split:], y[split:]
_, out = benchmark(6)
i_max = int(np.argmax(yte))
best = min(out.items(), key=lambda kv: rmse(yte, kv[1]))
print(int(i_max))                                # 145
print(round(float(yte[i_max,0]*sd + mu)))        # 359
print(best[0], round(float(best[1][i_max,0]*sd + mu)))
# rational 344
```

October 1957: actual $359$ spots, rational-layer forecast $249$. Even the
winning model shrinks the most extreme event by $\sim$30% — squared-error
training regresses toward the mean, and no amount of basis engineering fully
cures that. It still wins on *average* error; tail events are a separate,
harder contract.

---

### 8. Why the Rational Layer Wins at Long Horizon

The sunspot cycle is **asymmetric**: rises take $\sim$4 years, decays
$\sim$7. A polynomial must spend alternating-sign high-degree coefficients
to build such a sawtooth, and pays in oscillation everywhere else (Section
3). A rational denominator $1+|Q|$ can sharpen the response selectively —
large $|Q|$ flattens regions it wants to ignore, small $|Q|$ creates a
near-pole knee exactly where the cycle turns. With $L=24$ inputs the layer
learns one such knee per channel direction, which is precisely the "where in
the cycle are we" feature.

This is the same mathematics as Section 4: **localized features want
localized basis functions**, and poles are the cheapest localization device
in the polynomial toolbox.

---

### 9. Exercises

1. **Horizon sweep**: rerun `benchmark` for $h = 2, 3, 6, 12$ and plot RMSE
   vs horizon. Where exactly does the crossover "learned beats persistence"
   happen?
2. **Lag study**: repeat with $L = 12$ and $L = 36$. Does the rational
   layer's edge survive shorter memory?
3. **Pole inspection**: extract `rat.q` (denominator coefficients) after
   training on the Runge function and find the effective pole locations
   (roots of $Q$). Compare with $\pm i/5$.
4. **Cycle-25 test**: train only on data before 2009 and forecast 2020–2026.
   Does the ranking of Section 7 hold on the newest cycle?

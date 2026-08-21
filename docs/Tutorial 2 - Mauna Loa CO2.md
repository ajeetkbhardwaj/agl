# Tutorial 2 - Mauna Loa CO2: Trend, Season, and Orthogonal Polynomials

*Companion to `experiments/co2_trend.py`. Every code block runs against the
real NOAA record in `data/co2_mlo.csv`.*

## Table of Contents

### 1. The Physical Problem

### 2. A Model of Trend Plus Season

### 3. Harmonics from the Angle-Addition Formula

### 4. Why Orthogonal Polynomials? Conditioning

### 5. The Chebyshev Recurrence and Discrete Orthogonality

### 6. The Rank Trick: Low-Rank Weight Sharing

### 7. Domain Normalization and the Clamp

### 8. Fitting the Real Record

### 9. Extrapolation: The Honest Plateau

### 10. Reading Off the Seasonal Cycle

### 11. Exercises

---

### 1. The Physical Problem

**Problem**: The Mauna Loa observatory has measured atmospheric CO$_2$
every month since 1958 — the most famous curve in climate science. It rises
relentlessly (fossil fuel emissions) but *wiggles* every year (plants breathe:
they inhale CO$_2$ in northern summers, exhale in winters). Model both
effects with one equation, and be honest about what happens when you ask the
model about years it has never seen.

```Python
import pandas as pd, numpy as np

df = pd.read_csv("data/co2_mlo.csv", comment="#")
df = df[df["average"] > 0]           # missing months are flagged -99.99
t = df["decimal date"].to_numpy()
y = df["average"].to_numpy().astype(np.float64)
print(len(df))                       # 821
print(t.min(), t.max())              # 1958.2027 2026.5417
print(y[0], y[-1])                   # 315.71 429.12
```

---

### 2. A Model of Trend Plus Season

**Definition (Additive decomposition)**: we model the concentration as

$$
c(t) \;=\; \underbrace{p(t)}_{\text{smooth trend}} \;+\;
\underbrace{s(t)}_{\text{annual cycle}} \;+\; \varepsilon(t),
$$

with $p$ a polynomial of low degree and $s$ a periodic function of period one
year. The two effects live at separated frequencies, so additivity is a good
physical ansatz.

The network input is therefore three features:

$$
\phi(t) \;=\; \big[\, t,\;\; \sin(2\pi t),\;\; \cos(2\pi t) \,\big],
$$

where $t$ is a decimal date (fractional part = position within the year).

```Python
def features(t):
    w = 2*np.pi*(t % 1.0)
    return np.column_stack([t, np.sin(w), np.cos(w)]).astype(np.float64)

X = features(t)
train = t < 2010.0                    # train on 1958-2009 ...
Xtr, ytr = X[train], y[train]
Xte, yte = X[~train], y[~train]       # ... test on 2010-2026
print(train.sum(), (~train).sum())    # 622 199
```

Held-out testing is *chronological*: predicting the future is strictly harder
than interpolating, and only this split measures it.

---

### 3. Harmonics from the Angle-Addition Formula

Why include *both* $\sin$ and $\cos$? Any phase-shifted sinusoid of known
frequency $\omega$ is a linear combination of them:

$$
A\sin(\omega t + \varphi)
\;=\; \underbrace{A\cos\varphi}_{a}\,\sin(\omega t)
\;+\; \underbrace{A\sin\varphi}_{b}\,\cos(\omega t),
$$

by the identity $\sin(\alpha+\beta) = \sin\alpha\cos\beta +
\cos\alpha\sin\beta$. So a linear readout on $(\sin, \cos)$ can learn the
cycle's amplitude *and* its phase (the seasonal peak is around late April,
not January) without any nonlinear help.

```Python
# amplitude/phase recovery: fit a pure sinusoid to the detrended record
w = 2*np.pi*(t % 1.0)
Phi = np.column_stack([np.sin(w), np.cos(w)])
(ab, _, _, _) = np.linalg.lstsq(Phi, y - np.polyval(np.polyfit(t, y, 2), t), rcond=None)
amp = np.hypot(*ab); phase = np.arctan2(ab[1], ab[0])
print(round(amp, 2))                 # 2.84   (ppm; peak-to-trough ~5.7)
print(round(np.degrees(phase), 1))   # -20.4  -> peak at t_frac ~ 0.31 (late April)
```

---

### 4. Why Orthogonal Polynomials? Conditioning

Section 5 of Tutorial 1 hinted that monomial bases are badly conditioned.
Here is the quantitative statement.

**Definition (Condition number)**: for the normal equations
$\Phi^{\!\top}\Phi w = \Phi^{\!\top}y$, small relative errors in the data are
amplified by up to $\kappa(\Phi^{\!\top}\Phi) = \kappa(\Phi)^2$ in the
solution. A Vandermonde matrix on $[-1,1]$ has $\kappa$ growing
*exponentially* in degree; orthogonal bases keep it nearly flat.

```Python
xs = np.linspace(-1, 1, 200)
for d in (3, 5, 8):
    Vm = np.vander(xs, d+1, increasing=True)          # monomials 1, x, x^2...
    Tc = np.polynomial.chebyshev.chebvander(xs, d)    # T_0..T_d
    print(d, f"{np.linalg.cond(Vm):.2e}", f"{np.linalg.cond(Tc):.2e}")
# 3 8.13e+00 2.38e+00
# 5 4.26e+01 2.76e+00
# 8 5.41e+02 3.04e+00
```

At degree 8 the monomial basis is $178\times$ worse conditioned — and the gap
widens exponentially. This is why `OrthoPolyLayer` expands in Chebyshev /
Legendre / Hermite polynomials rather than raw powers.

---

### 5. The Chebyshev Recurrence and Discrete Orthogonality

**Definition (Chebyshev polynomials, first kind)**:

$$
T_n(x) \;=\; \cos(n \arccos x),
\qquad
T_{n+1}(x) \;=\; 2x\,T_n(x) \;-\; T_{n-1}(x),
\qquad T_0 = 1,\; T_1 = x .
$$

The trigonometric definition makes orthogonality obvious: substituting
$x = \cos\theta$,

$$
\int_{-1}^{1} \frac{T_m(x)\,T_n(x)}{\sqrt{1-x^2}}\,dx
\;=\; \int_{0}^{\pi} \cos(m\theta)\cos(n\theta)\, d\theta \;=\; 0
\quad (m \neq n).
$$

Numerically, on Chebyshev nodes the Gram matrix is essentially diagonal:

```Python
grid = np.cos(np.pi*(np.arange(400)+0.5)/400)     # Chebyshev nodes
T = np.polynomial.chebyshev.chebvander(grid, 4)
G = T.T @ T / 400                                  # discrete Gram matrix
print(np.round(G, 3))
# [[ 1.  0. -0.  0. -0.]
#  [ 0.  0.5 0.  -0.  0. ]
#  [-0.  0.  0.5 0.  -0. ]
#  [ 0.  -0. -0.  0.5 -0. ]
#  [-0.  0. -0. -0.  0.5]]
```

Diagonal Gram matrix $\Rightarrow$ each basis direction carries independent
information $\Rightarrow$ gradient descent sees well-scaled features.

---

### 6. The Rank Trick: Low-Rank Weight Sharing

**Definition (Low-rank factorization)**: instead of a full weight matrix
$W \in \mathbb{R}^{m\times n}$ ($mn$ parameters), store

$$
W \;=\; U\,\Sigma\, V^{\!\top} \;\approx\; U_r\,\Sigma_r\, V_r^{\!\top},
$$

keeping only rank $r$: $r(m+n) + r$ numbers. **Eckart–Young theorem**: the
truncated SVD is the *best* rank-$r$ approximation in Frobenius norm, and the
error equals the energy of the discarded singular values.

```Python
rng = np.random.RandomState(0)
A, B = rng.randn(12, 3), rng.randn(3, 9)
W = A @ B + 0.05*rng.randn(12, 9)      # secretly rank 3 + noise
U, s, Vt = np.linalg.svd(W, full_matrices=False)
print(np.round(s, 2))
# [15.98  9.6   4.51  0.22  0.19  0.18  0.16  0.08  0.03]
for r in (1, 2, 3):
    Wr = U[:, :r] @ np.diag(s[:r]) @ Vt[:r]
    err = np.linalg.norm(W-Wr)/np.linalg.norm(W)
    print(r, f"{err:.4f}", U[:, :r].size + r + Vt[:r].size)
# 1 0.5534 22
# 2 0.2358 44
# 3 0.0204 66
print(W.size)                          # 108
```

Rank 3 keeps $97.96\%$ of the matrix with $66$ instead of $108$ numbers.
`OrthoPolyNetwork(rank=3)` applies the same idea to its polynomial weight
tensors: channels share spectral directions, which regularizes the fit.

---

### 7. Domain Normalization and the Clamp

Chebyshev polynomials are only well-behaved on $[-1, 1]$ ($|T_n| \le 1$
there; outside, they grow like $x^n$). The layer therefore standardizes each
input channel using statistics gathered during training:

$$
u \;=\; \operatorname{clamp}\!\left(\frac{2(x - x_{\min})}{x_{\max} - x_{\min}} - 1,\; -1,\; 1\right).
$$

For the time channel, training saw $x_{\min} \approx 1958.05$,
$x_{\max} \approx 2009.96$. Watch what the clamp does to future dates:

```Python
def clamp_norm(v, lo, hi):
    return np.clip(2*(v-lo)/(hi-lo)-1, -1, 1)

print(clamp_norm(np.array([1958.0, 1984.0, 2009.99, 2015.0, 2026.0]),
                 1958.05, 2009.96))
# [-1.0000000e+00 -1.9264111e-04  1.0000000e+00  1.0000000e+00
#  1.0000000e+00]
```

Every date after 2009.96 maps to *exactly* $u = 1$. The polynomial evaluated
at $u=1$ is a constant. **This single line of math predicts the experimental
plateau we are about to see**: the model cannot extrapolate because its
input pipeline refuses to leave the interval it was trained on.

---

### 8. Fitting the Real Record

Architecture: 3 inputs, hidden width 16, degree-5 Chebyshev expansion,
rank-3 sharing. Targets are standardized ($\mu = 346.11$ ppm,
$\sigma = 21.78$ ppm) so the network's initial near-zero output sits close to
the data level.

```Python
import torch
from src.nn.orthopoly_nn import OrthoPolyNetwork

mu, sd = float(ytr.mean()), float(ytr.std())
torch.manual_seed(7); np.random.seed(7)
net = OrthoPolyNetwork(input_dim=3, output_dim=1, hidden_dims=[16],
                       max_degree=5, rank=3, basis_type="chebyshev_first")
net.fit(Xtr, ((ytr-mu)/sd).reshape(-1, 1), epochs=300,
        learning_rate=0.01, verbose=False)

def predict(Xq):
    return net.predict(Xq).ravel()*sd + mu

def r2(yy, yh):
    return 1 - float(np.sum((yy-yh)**2))/float(np.sum((yy-yy.mean())**2))

pred_tr, pred_te = predict(Xtr), predict(Xte)
print(f"{r2(ytr, pred_tr):.6f}")     # 0.999162
print(f"{np.sqrt(np.mean((ytr-pred_tr)**2)):.3f}")   # 0.631
print(f"{r2(yte, pred_te):.6f}")     # -2.888710
print(f"{np.sqrt(np.mean((yte-pred_te)**2)):.3f}")   # 23.936
```

In-window: sub-ppm accuracy, $R^2 = 0.9992$. Out-window: $R^2$ is
*negative* — the model does worse than predicting the historical mean.
Interpolation superb, extrapolation worthless. The next section shows this
is not a defect of training but a theorem of Section 7.

---

### 9. Extrapolation: The Honest Plateau

Predict mid-year values across seven decades:

```Python
for year in (1990, 2000, 2005, 2010, 2015, 2020, 2024):
    mask = (t >= year) & (t < year+1)
    p = float(predict(features(np.array([year+0.55])))[0])
    a = float(y[mask].mean())
    print(year, round(a, 2), round(p, 2))
# 1990 354.45 354.78
# 2000 369.71 371.15
# 2005 379.98 380.83
# 2010 390.1 389.7
# 2015 401.01 389.7
# 2020 414.21 389.7
# 2024 424.6 389.7
```

From 2010 onward the prediction is frozen at exactly $389.70$ ppm while the
planet continues to emit. The clamp of Section 7 saturates all three inputs;
a constant input forces a constant output. **Moral**: orthogonal-polynomial
models are interpolation machines. For forecasting you would hand the *trend*
channel to a model that extrapolates (or feed differences), and reserve the
orthogonal expansion for the seasonal shape.

---

### 10. Reading Off the Seasonal Cycle

Freeze the year, scan the phase — the network's internal picture of "one
year":

```Python
tt = 2000.0
phases = np.linspace(0, 1, 24, endpoint=False)
preds = np.array([float(predict(features(np.array([tt+ph])))[0])
                  for ph in phases])
print(round(preds.max() - preds.min(), 2))          # 7.03

detrended = ytr - pd.Series(ytr).rolling(13, center=True,
                                         min_periods=1).mean().to_numpy()
print(round(detrended.max() - detrended.min(), 2))  # 7.8
```

The learned cycle swings $7.03$ ppm peak-to-trough against $7.80$ ppm in the
raw detrended data — about $90\%$ of the true amplitude, slightly smoothed by
the rank-3 bottleneck. The *shape* (phase, skew) is captured faithfully even
though the model never received a "month" label.

---

### 11. Exercises

1. **Difference trick**: fit the network to monthly *increments*
   $\Delta c(t)$ instead of levels, then integrate predictions forward. Does
   the 2010–2026 forecast improve?
2. **Second harmonic**: add $\sin(4\pi t), \cos(4\pi t)$ features. The
   seasonal cycle is not a perfect sinusoid — quantify the RMSE gain.
3. **Rank study**: refit with `rank` = 1, 2, 3, 4 and compare in-window RMSE
   and seasonal amplitude. Where does under-parameterizing start to hurt?
4. **No-clamp experiment**: bypass normalization (feed pre-normalized
   features and use a plain polynomial layer) and confirm extrapolation now
   diverges instead of plateauing — worse behavior, but instructive.

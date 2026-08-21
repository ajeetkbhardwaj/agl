# Tutorial 5 - Power Plant Output: Additive vs Multiplicative Structure

*Companion to `experiments/power_plant.py`. Every code block runs against
the real UCI dataset in `data/ccpp/CCPP/Folds5x2_pp.xlsx`.*

## Table of Contents

### 1. The Physical Problem

### 2. Heat Engines: Why Temperature Dominates

### 3. Additive Models and Their Exact Blind Spot

### 4. Multiplicative Features and the Segre Embedding

### 5. Stone-Weierstrass: Why Polynomials Suffice

### 6. The Benchmark Setup

### 7. Results and the Verdict on Interactions

### 8. Exercises

---

### 1. The Physical Problem

**Problem**: A combined-cycle power plant generates $9568$ six-year hourly
sensor readings of net electrical output $PE$ (MW) alongside four ambient
conditions: temperature $AT$ (C), exhaust vacuum $V$ (cm Hg), pressure $AP$
(mbar), relative humidity $RH$ (%). Predict output from environment — and
decide whether the physics needs *interaction* terms or pure additive
contributions.

```Python
import pandas as pd, numpy as np

df = pd.read_excel("data/ccpp/CCPP/Folds5x2_pp.xlsx")
X = df[["AT","V","AP","RH"]].to_numpy().astype(np.float64)
y = df["PE"].to_numpy().astype(np.float64).reshape(-1, 1)
print(len(X))                                          # 9568
for i, name in enumerate(["AT", "V", "AP", "RH"]):
    c = float(np.corrcoef(X[:, i], y.ravel())[0, 1])
    print(name, round(c, 3))
# AT -0.948
# V -0.87
# AP 0.518
# RH 0.39
```

Temperature alone correlates at $-0.948$. The other channels matter but are
partially redundant — a classic correlated-regressors setting where model
structure decides how gracefully parameters share the work.

---

### 2. Heat Engines: Why Temperature Dominates

**Definition (Carnot efficiency)**: any heat engine operating between a hot
reservoir $T_h$ and cold reservoir $T_c$ (kelvin) satisfies

$$
\eta_{\rm Carnot} \;=\; \frac{W}{Q_h} \;=\; 1 - \frac{T_c}{T_h}.
$$

The steam cycle's hot side is fixed by combustion ($T_h \approx 800\,$K),
but the *cold* side is cooled by ambient air: a hot day raises $T_c$ and
directly taxes efficiency.

```Python
print(round(1 - 300/700, 3))                 # 0.571   (generic cycle)

for Tatm in (288.15, 308.15):                # 15 C vs 35 C day
    print(round(1 - Tatm/800.0, 4))
# 0.6398
# 0.6148
```

A 20 C heat wave costs $\sim 2.5$ percentage points of ideal efficiency —
about $4\%$ relative. The data's strong negative $AT$ correlation is
thermodynamics, not coincidence.

---

### 3. Additive Models and Their Exact Blind Spot

**Definition (Additive model)**:

$$
f(u_1,\dots,u_n) \;=\; f_1(u_1) + \dots + f_n(u_n).
$$

**Claim**: an additive model represents the product $u_1 u_2$ *only to
accuracy zero*. Proof sketch: mixed second derivative
$\partial^2 (u_1 u_2)/\partial u_1 \partial u_2 = 1$, while every additive
function has $\partial^2 f/\partial u_1 \partial u_2 = 0$. Interactions are
not merely hard for additive models — they are invisible.

Numerically: fit the best additive approximation to $g = u_1 u_2$ by least
squares,

```Python
rng = np.random.RandomState(0)
G = rng.uniform(-1, 1, (500, 2))
g1, g2 = G[:, 0], G[:, 1]
target = g1*g2
A = np.column_stack([np.ones(500), g1, g2])
w = np.linalg.solve(A.T @ A, A.T @ target)   # best additive fit
print(np.round(w, 3))                        # [ 0.005 -0.012  0.027]
pred = w[0] + w[1]*g1 + w[2]*g2              # additive model, written out
r2 = 1 - np.sum((target-pred)**2)/np.sum((target-target.mean())**2)
print(round(float(r2), 4))                   # 0.0025
```

The optimal additive guess is essentially "predict zero": $R^2 = 0.0025$.
If a response has genuine interaction structure, additive models cannot
fake it.

---

### 4. Multiplicative Features and the Segre Embedding

**Definition (Multiplicative / rank-1 cross features)**: instead of summing
per-channel responses, form products across channels. With univariate basis
functions $\psi^{(i)}_r$ per channel, rank-$r$ multiplicative features are

$$
\varphi_r(u) \;=\; \bigotimes_{i=1}^{n} \big(\psi^{(i)}_0(u_i),\dots,
\psi^{(i)}_{D}(u_i)\big),
\qquad
f(u) \;=\; \sum_{r=1}^{R} w_r\, \prod_{i=1}^{n}
\Big(\sum_{d=0}^{D} c^{(i,r)}_d\, \psi^{(i)}_d(u_i)\Big).
$$

Each term is a product of univariate polynomials — a point on the **Segre
embedding** of the product of projective spaces. The dimension bookkeeping
is the punchline: a full degree-$D$ tensor-product grid costs $(D+1)^n$
features, while $R$ Segre terms cost $R \cdot n(D+1)$ — *exponential
expressive gain per linear parameter spend*, because each such term already
contains all $D^n$ interactions at degree exactly $nD$.

The trade-off is symmetry: products treat channels multiplicatively, so
purely additive structure needs many terms. Neither mode dominates a priori
— which is why we benchmark both.

---

### 5. Stone-Weierstrass: Why Polynomials Suffice

**Theorem (Stone–Weierstrass)**: on a compact set $K \subset
\mathbb{R}^n$, the polynomial ring $\mathbb{R}[u_1,\dots,u_n]$ is dense in
$(C(K), \|\cdot\|_\infty)$: for every continuous $f$ and every $\varepsilon
> 0$ there is a polynomial $p$ with $\sup_K |f - p| < \varepsilon$.

So polynomials lose nothing in principle. In practice the theorem says
nothing about *sample complexity*: with finite data, the structural prior
(additive? multiplicative? plain MLP?) determines which approximator wins.
That is an empirical question — Section 7 answers it for this plant.

---

### 6. The Benchmark Setup

Three models on identical standardized data (80/20 random split):

1. `OrthoPolyNetwork`, `interaction_mode="additive"`, degree 4;
2. `OrthoPolyNetwork`, `interaction_mode="multiplicative"` (Segre), degree 3;
3. a plain PyTorch MLP, two hidden layers of 64 tanh units.

```Python
import torch
import torch.nn as nn
from src.nn.orthopoly_nn import OrthoPolyNetwork

rng = np.random.RandomState(42)
idx = rng.permutation(len(X))
n_tr = int(0.8 * len(X))
tr, te = idx[:n_tr], idx[n_tr:]
mu_x, sd_x = X[tr].mean(0), X[tr].std(0)
mu_y, sd_y = y[tr].mean(), y[tr].std()
Xs, ys = (X - mu_x)/sd_x, (y - mu_y)/sd_y

def r2(yy, yh):
    return 1 - float(np.sum((yy-yh)**2))/float(np.sum((yy-yy.mean())**2))

torch.manual_seed(7); np.random.seed(7)
add = OrthoPolyNetwork(input_dim=4, output_dim=1, hidden_dims=[16],
                       max_degree=4, rank=4, basis_type="chebyshev_first",
                       interaction_mode="additive")
add.fit(Xs[tr], ys[tr], epochs=200, learning_rate=0.01, verbose=False)
pa = add.predict(Xs[te]) * sd_y + mu_y
print(sum(p.numel() for p in add.parameters()))        # 737

torch.manual_seed(7); np.random.seed(7)
mul = OrthoPolyNetwork(input_dim=4, output_dim=1, hidden_dims=[16],
                       max_degree=3, rank=4, basis_type="chebyshev_first",
                       interaction_mode="multiplicative")
mul.fit(Xs[tr], ys[tr], epochs=200, learning_rate=0.01, verbose=False)
pm = mul.predict(Xs[te]) * sd_y + mu_y
print(sum(p.numel() for p in mul.parameters()))        # 405

torch.manual_seed(7); np.random.seed(7)
mlp = nn.Sequential(nn.Linear(4, 64), nn.Tanh(),
                    nn.Linear(64, 64), nn.Tanh(),
                    nn.Linear(64, 1))
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
Xtr_t = torch.tensor(Xs[tr], dtype=torch.float32)
ytr_t = torch.tensor(ys[tr], dtype=torch.float32)
for _ in range(300):
    opt.zero_grad()
    nn.functional.mse_loss(mlp(Xtr_t), ytr_t).backward()
    opt.step()
with torch.no_grad():
    pb = mlp(torch.tensor(Xs[te], dtype=torch.float32)).numpy()*sd_y + mu_y
print(sum(p.numel() for p in mlp.parameters()))        # 4545
```

Parameter budgets differ by design: the MLP spends $4545$ parameters, the
additive net $737$, the multiplicative net just $405$.

---

### 7. Results and the Verdict on Interactions

```Python
for name, pred in [("OrthoPoly additive", pa),
                   ("OrthoPoly multiplicative", pm),
                   ("MLP 64x64", pb)]:
    print(f"{name:<25} R^2 {r2(y[te], pred):.4f}   "
          f"RMSE {np.sqrt(np.mean((y[te]-pred)**2)):.3f} MW")
# OrthoPoly additive       R^2 0.9432   RMSE 4.055 MW
# OrthoPoly multiplicative R^2 0.8944   RMSE 5.528 MW
# MLP 64x64                R^2 0.9361   RMSE 4.301 MW
```

Two findings:

* The **additive orthogonal-polynomial network beats the 6x-larger MLP**
  ($4.055$ vs $4.301$ MW) — matching published ANN benchmarks for this
  dataset (~4 MW). Structured simplicity outperforms generic capacity.
* The **multiplicative mode loses** ($5.528$ MW). Is that surprising?
  Test the raw data for a temperature-humidity interaction directly —
  compare the humidity effect on hot days versus cold days:

```Python
hot = X[:, 0] > np.percentile(X[:, 0], 90)
cold = X[:, 0] < np.percentile(X[:, 0], 10)
rh_med = float(np.median(X[:, 3]))
d_hot = y[hot & (X[:, 3] > rh_med)].mean() \
      - y[hot & (X[:, 3] <= rh_med)].mean()
d_cold = y[cold & (X[:, 3] > rh_med)].mean() \
       - y[cold & (X[:, 3] <= rh_med)].mean()
print(f"{float(d_hot):+.2f}")                # -1.51
print(f"{float(d_cold):+.2f}")               # -1.08
```

Humidity's effect shifts only from $-1.51$ to $-1.08$ MW between extreme
temperature regimes — a modest interaction, too weak to justify the
multiplicative parameterization's optimization burden here. The benchmark's
verdict is consistent with the physics: this response is dominated by
smooth, nearly additive channel effects (Section 2), and the additive model
was the right structural prior.

---

### 8. Exercises

1. **Degree sweep**: rerun additive mode with `max_degree` 2..6. Where does
   test RMSE bottom out, and does overfitting follow the parameter count?
2. **Hybrid mode**: build a feature set containing both additive channels
   and one hand-picked product $u_{AT}\,u_{RH}$; does it close the gap to
   the best model?
3. **Correlation robustness**: drop $V$ from the inputs. Which model
   degrades least, and why does rank sharing help?
4. **Efficiency residual**: after fitting, regress $PE$ minus prediction on
   $AT$. Is there a systematic secondary temperature effect the Carnot
   argument missed (e.g., air density)?

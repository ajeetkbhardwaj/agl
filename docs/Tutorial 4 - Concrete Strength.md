# Tutorial 4 - Concrete Strength: Closed-Form Laws and Importance

*Companion to `experiments/concrete_strength.py`. Every code block runs
against the real UCI dataset in `data/concrete/Concrete_Data.xls`.*

## Table of Contents

### 1. The Physical Problem

### 2. The Curse of Dimensionality in Polynomial Features

### 3. Standardization: Coordinates Matter

### 4. Model 1 - A Polynomial Network With a Closed-Form Law

### 5. Distillation: Exact Versus Pruned Equations

### 6. Model 2 - Additive Networks (Generalized Additive Models)

### 7. Architecture-Native Importance Ranking

### 8. Exercises

---

### 1. The Physical Problem

**Problem**: A civil engineer hands you $1030$ laboratory concrete mixes.
Each is described by eight quantities — cement, blast-furnace slag, fly ash,
water, superplasticizer, coarse aggregate, fine aggregate (all kg/m$^3$) and
curing age (days) — together with the resulting **compressive strength** in
MPa. Produce (a) an accurate strength predictor and (b) an *equation* an
engineer can read.

The dominant classical law is **Abrams' rule** (1918): strength decreases
with the water-to-cement ratio. Check it in the raw data:

```Python
import pandas as pd, numpy as np

SHORT = ["cement","slag","flyash","water","superpl","coarse","fine","age"]
df = pd.read_excel("data/concrete/Concrete_Data.xls")
df.columns = SHORT + ["strength"]
X = df[SHORT].to_numpy().astype(np.float64)
y = df["strength"].to_numpy().astype(np.float64).reshape(-1, 1)
print(X.shape)                                       # (1030, 8)

wcr = X[:, 3] / X[:, 0]                              # water / cement
print(round(float(np.corrcoef(y.ravel(), wcr)[0, 1]), 3))   # -0.501
```

Correlation $-0.501$: more water per cement, weaker concrete. But Abrams'
rule is only one strand — fly ash and slag *replace* cement with delayed
reaction, and age strengthens everything. We want the model to find all of
it.

---

### 2. The Curse of Dimensionality in Polynomial Features

**Definition (Monomial count)**: the number of monomials of total degree
$\le d$ in $n$ variables is

$$
\#\{x^\alpha : |\alpha| \le d\} \;=\; \binom{n+d}{d}.
$$

```Python
from math import comb
for n, d in ((8, 1), (8, 2), (8, 3), (8, 4)):
    print(n, d, comb(n + d, d))
# 8 1 9
# 8 2 45
# 8 3 165
# 8 4 495
```

A *full* degree-4 polynomial in 8 inputs has $495$ free coefficients per
output — and worse, most of those features are products like $u_1^2 u_5 u_7$
that no engineer wants to read. This tension (accuracy needs interactions;
interpretability needs fewness) drives the whole tutorial: we will use
interactions for accuracy (Section 4), then throw them away on purpose for
interpretability (Section 6).

---

### 3. Standardization: Coordinates Matter

**Definition ($z$-score)**: each feature is replaced by

$$
u_i \;=\; \frac{x_i - \bar x_i}{s_i},
\qquad \bar x_i, s_i = \text{train-split mean, std of } x_i .
$$

This is not cosmetic. Cement enters in units of $\sim 500$ kg/m$^3$, age in
days from 1 to 365: raw coordinates make the monomial basis catastrophically
ill-scaled (the same conditioning story as Tutorial 2, Section 4). All
learned equations below therefore live in $u$-coordinates, and we report the
conversion table once:

```Python
rng = np.random.RandomState(42)
idx = rng.permutation(len(X))
n_tr = int(0.8 * len(X))
tr, te = idx[:n_tr], idx[n_tr:]
mu_x, sd_x = X[tr].mean(0), X[tr].std(0)
mu_y, sd_y = y[tr].mean(), y[tr].std()
Xs = (X - mu_x) / sd_x
ys = (y - mu_y) / sd_y
print(round(mu_y, 2), round(sd_y, 2))     # 35.88 17.02
for i, n in enumerate(SHORT):
    print(f"{n:<9} u = (x - {round(mu_x[i],1)}) / {round(sd_x[i],1)}")
```

To convert any discovered law back to engineering units, substitute
$u_i = (x_i - \bar x_i)/s_i$ symbolically — sympy does this in one line
(exercise 3).

---

### 4. Model 1 - A Polynomial Network With a Closed-Form Law

Architecture: one hidden layer of 16 units, degree-2 polynomial weights,
`tanh` activations. The activations break exact polynomial closure (unlike
Tutorial 1), buying nonlinear capacity at the price of a messier symbolic
form.

```Python
import torch, sympy as sp
from src.nn.poly_nn import PolynomialNeuralNetwork

torch.manual_seed(7); np.random.seed(7)
pnn = PolynomialNeuralNetwork(input_dim=8, output_dim=1, hidden_dims=[16],
                              polynomial_degree=2, activation='tanh')
pnn.fit(Xs[tr], ys[tr], epochs=250, learning_rate=0.01, verbose=False)

pred_te = pnn.predict(Xs[te]) * sd_y + mu_y
ss_res = float(np.sum((y[te]-pred_te)**2))
ss_tot = float(np.sum((y[te]-y[te].mean())**2))
print(f"{1-ss_res/ss_tot:.4f}")                      # 0.8476
print(f"{np.sqrt(np.mean((y[te]-pred_te)**2)):.2f}") # 6.00
```

$R^2 = 0.85$, RMSE $= 6.0$ MPa — respectable for concrete, where batch
heterogeneity alone contributes several MPa of irreducible noise.

---

### 5. Distillation: Exact Versus Pruned Equations

`get_global_equation` walks the network graph symbolically. Two regimes:

* **Pruned** (`threshold=5e-3`, rounded): human-readable, but dropping terms
  introduces error.
* **Exact** (`threshold=1e-9`, `round_to=None`): must reproduce `forward()`
  to floating-point precision.

```Python
names = [f"u{i}_{n}" for i, n in enumerate(SHORT)]
eq_pruned = pnn.get_global_equation(names, threshold=5e-3)
print(str(eq_pruned)[:300])
# 0.462*tanh(0.091*u0_cement**2*u6_fine - 0.059*u0_cement*u4_superpl -
# 0.608*u0_cement*u6_fine + 0.086*u0_cement - 0.13*u1_slag**3 +
# 0.214*u1_slag**2*u2_flyash + 0.048*u1_slag + 2.131*u2_flyash -
# 1.2*u3_water + 0.329*u4_superpl - 0.241*u5_coarse + 0.077*u6_fine +
# 0.358*u7_age + 0.537)**2*tanh(0.008*u0

eq_exact = pnn.get_global_equation(names, threshold=1e-9, round_to=None)
f_eq = sp.lambdify(sp.symbols(" ".join(names)), eq_exact, "numpy")
sym_vals = np.asarray(f_eq(*[Xs[te][:, i] for i in range(8)]),
                      dtype=float).reshape(-1, 1)
print(f"{np.abs(sym_vals - pnn.predict(Xs[te])).max():.2e}")   # 5.22e-07

f_pr = sp.lambdify(sp.symbols(" ".join(names)), eq_pruned, "numpy")
sym_pr = np.asarray(f_pr(*[Xs[te][:, i] for i in range(8)]),
                    dtype=float).reshape(-1, 1)
print(f"{np.abs(sym_pr - pnn.predict(Xs[te])).max():.2e}")     # 8.77e-03
```

Even visible in the pruned snippet are physically sensible terms:
$-1.2\,u_{\rm water}$ (Abrams!), $+2.131\,u_{\rm flyash}$ interacting with
slag, $+0.358\,u_{\rm age}$. The exact equation agrees with the network to
$5\times10^{-7}$; pruning to readable form costs $9\times10^{-3}$ — three
orders of magnitude, but still far below one MPa after unscaling
($\times 17.02$: about $0.15$ MPa). Choose per purpose: *deploy* the exact
equation, *read* the pruned one.

---

### 6. Model 2 - Additive Networks (Generalized Additive Models)

**Definition (Generalized additive model)**: a model of the form

$$
f(u_1,\dots,u_n) \;=\; \sum_{i=1}^{n} f_i(u_i),
$$

a sum of *univariate* functions. By construction it cannot represent
interactions ($u_1 u_2$ is not additive), but every term $f_i$ can be
plotted, printed and reasoned about individually.

The `OrthoPolyNetwork`'s first layer in additive mode builds exactly this:
one Chebyshev polynomial per input channel. Dropping all cross-terms costs
surprisingly little accuracy here:

```Python
from src.nn.orthopoly_nn import OrthoPolyNetwork

torch.manual_seed(7); np.random.seed(7)
opn = OrthoPolyNetwork(input_dim=8, output_dim=1, hidden_dims=[24],
                       max_degree=4, rank=4, basis_type="chebyshev_first")
opn.fit(Xs[tr], ys[tr], epochs=250, learning_rate=0.01, verbose=False)

pred2 = opn.predict(Xs[te]) * sd_y + mu_y
ss_res2 = float(np.sum((y[te]-pred2)**2))
print(f"{1-ss_res2/ss_tot:.4f}")                     # 0.8916
print(f"{np.sqrt(np.mean((y[te]-pred2)**2)):.2f}")   # 5.06
```

The *additive* model ($R^2 = 0.8916$) beats the interaction-rich PNN
($0.8476$): for this dataset, univariate response curves carry nearly all
the signal, and removing interactions acts as regularization.

---

### 7. Architecture-Native Importance Ranking

Additivity gives us importance for free. Each channel contributes
$p_i(u_i)$; its **spread over the test population**

$$
I_i \;=\; \operatorname{std}_{j}\!\big(p_i(u_i^{(j)})\big)
$$

measures how much channel $i$ actually moves the prediction across real
mixes — a first-order variance decomposition (the same spirit as Sobol
indices, computed analytically from the extracted polynomials rather than by
resampling).

```Python
from src.alggeom.polynomial import Variable

opn.eval()
polys = opn.layers[0].get_polynomial_weights()[0]
contribs = []
for i, poly in enumerate(polys):
    var = Variable(f"x{i}")
    vals = np.array([float(poly.evaluate({var: v}))
                     for v in Xs[te][:, i]])
    contribs.append(vals.std())
    print(f"{SHORT[i]:<9} std={vals.std():.3f}  mean={vals.mean():+.3f}")
# cement    std=1.029  mean=+0.431
# slag      std=3.881  mean=-1.234
# flyash    std=8.005  mean=+2.464
# water     std=1.034  mean=+0.171
# superpl   std=9.049  mean=+1.757
# coarse    std=2.584  mean=+1.237
# fine      std=4.587  mean=+1.559
# age       std=133.304  mean=+18.011

order = np.argsort(-np.asarray(contribs))
print(" > ".join(SHORT[i] for i in order))
# age > superpl > flyash > fine > slag > coarse > water > cement
```

**Age dominates by an order of magnitude** ($133$ vs $9$) — correct
physics: curing from 1 to 365 days transforms strength more than any
composition change. Note also what the ranking is *not*: it is not
correlation with the target (water's Abrams effect has small spread because
most mixes sit near similar ratios); it measures contribution *variance over
the actual design space*, which is what "which knob moves my concrete"
means to an engineer.

---

### 8. Exercises

1. **Unscaling**: substitute $u_i = (x_i-\bar x_i)/s_i$ into the pruned
   equation with sympy and expand. How do coefficients transform? Which
   single term dominates in engineering units?
2. **Interaction audit**: train the PNN with `polynomial_degree=1` (pure
   affine layers). How much of the $R^2$ gap to Section 4 was really due to
   interactions versus optimization?
3. **Sobol comparison**: estimate first-order Sobol indices by Monte Carlo
   on the additive model and compare with the $I_i$ of Section 7 (they
   should agree up to sampling error — why?).
4. **Age curve**: plot $p_{\rm age}(u)$ over the test range. Is the shape
   closer to $\sqrt{t}$ (diffusion-limited hydration) or to $\log t$?

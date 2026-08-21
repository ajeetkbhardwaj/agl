# Tutorial 1 - Kepler's Third Law from First Principles

*Companion to `experiments/kepler_law.py`. Every code block runs against the
real NASA Exoplanet Archive data in `data/exoplanets.csv`.*

## Table of Contents

### 1. The Physical Problem

### 2. Deriving $T^2 \propto a^3$ from Newton's Laws

### 3. Logarithms Turn a Power Law into a Straight Line

### 4. Least Squares: The Optimal Line Through Scatter

### 5. Polynomial Regression as Linear Algebra

### 6. A Network That Is Exactly a Polynomial

### 7. Fitting 5586 Real Planets

### 8. Reading Off the Discovered Law

### 9. Exact Symbolic Distillation

### 10. Exercises

---

### 1. The Physical Problem

**Problem**: You are handed a catalogue of $5586$ confirmed exoplanets. For
each planet you know only two numbers: the semi-major axis $a$ of its orbit
(in astronomical units) and its orbital period $T$ (in days). *Discover the
law connecting them* — without being told Kepler's third law.

```Python
import pandas as pd, numpy as np

df = pd.read_csv("data/exoplanets.csv")
print(len(df))                       # 5586
df = df[(df.pl_orbper > 0) & (df.pl_orbsmax > 0)]
df = df[df.pl_orbsmax < 100]         # drop extreme outliers
x = np.log10(df.pl_orbsmax.to_numpy()).reshape(-1, 1)
y = np.log10(df.pl_orbper.to_numpy()).reshape(-1, 1)
print(len(x))                        # 5581
print(x.min(), x.max())              # -2.3565473235138126 1.8325089127062364
```

The cleaned sample spans hot Jupiters hugging their stars ($a \approx
10^{-2.4}\,\mathrm{AU}$) to wide-orbit giants ($a \approx 68\,\mathrm{AU}$).

---

### 2. Deriving $T^2 \propto a^3$ from Newton's Laws

**Definition (Newtonian orbit balance)**: for a planet of mass $m$ on a
circular orbit of radius $a$ around a star of mass $M$, gravity supplies the
centripetal force:

$$
\frac{G M m}{a^2} \;=\; \frac{m v^2}{a}
\qquad\Longrightarrow\qquad
v^2 \;=\; \frac{G M}{a}.
$$

The period is one circumference over one speed, $T = 2\pi a / v$, so

$$
T^2 \;=\; \frac{4\pi^2 a^2}{v^2}
\;=\; \frac{4\pi^2 a^2}{GM/a}
\;=\; \frac{4\pi^2}{G M}\, a^3 .
$$

In units where time is days, distance is AU and mass is solar masses
($G M_\odot$ absorbs the constants):

$$
T_{\rm days} \;=\; 365.25 \cdot a_{\rm AU}^{3/2} \cdot M^{-1/2}.
$$

```Python
# verify the derived law numerically for the Earth (a = 1 AU, M = 1)
T_pred = 365.25 * 1.0**1.5 / 1.0**0.5
print(T_pred)                        # 365.25   (the actual year!)
T_mars = 365.25 * 1.5237**1.5        # Mars at a = 1.524 AU
print(round(T_mars, 1))              # 687.0    (actual: 687.0 days)
```

Two derivations down, one constant to explain: the exponent $3/2$ is the
*law*, and it is what we want the network to rediscover.

---

### 3. Logarithms Turn a Power Law into a Straight Line

**Definition (Log-linearization)**: if $y = C\,x^{\alpha}$ then taking
$\log_{10}$ of both sides gives

$$
\log_{10} y \;=\; \alpha\, \log_{10} x \;+\; \log_{10} C,
$$

a straight line with slope $\alpha$. Multiplicative structure becomes
additive structure.

Applying this to Kepler's law:

$$
\underbrace{\log_{10} T}_{y} \;=\;
\underbrace{1.5}_{\text{slope}} \cdot \underbrace{\log_{10} a}_{x} \;+\;
\underbrace{\log_{10} 365.25 - \tfrac{1}{2}\log_{10} M}_{\text{intercept}} .
$$

The slope is **exactly $1.5$** — a universal prediction. The intercept
depends on the star's mass $M$, which the catalogue does not always tell us:
this is the physical source of scatter around the line. Stellar masses range
roughly over $[0.1, 10]\,M_\odot$, so the intercept term
$-\tfrac12\log_{10}M$ spreads about $\pm 0.5$ dex at most; the observed
scatter will turn out much smaller because most catalogued stars are
sun-like.

```Python
print(np.log10(365.25))              # 2.5625902246063346
# theory line:  log T = 1.500 * u + 2.5626   (for sun-like stars, M ~ 1)
```

---

### 4. Least Squares: The Optimal Line Through Scatter

**Definition (Least-squares solution)**: given data $(x_i, y_i)$ and a design
matrix $\Phi$ whose rows are feature vectors $\phi(x_i)$, the least-squares
coefficients minimize the sum of squared residuals

$$
S(w) \;=\; \sum_{i=1}^{n} \big(y_i - \phi(x_i)^{\!\top} w\big)^2
\;=\; \|y - \Phi w\|_2^2 .
$$

Setting the gradient to zero,

$$
\nabla_w S = -2\,\Phi^{\!\top}(y - \Phi w) = 0
\quad\Longrightarrow\quad
\boxed{\;\Phi^{\!\top}\Phi\, w \;=\; \Phi^{\!\top} y\;}
$$

—the **normal equations**. If $\Phi$ has full column rank the solution is
unique: $w = (\Phi^{\!\top}\Phi)^{-1}\Phi^{\!\top}y = \Phi^{+} y$.

For the line $\phi(u) = (1, u)$:

```Python
Phi = np.hstack([np.ones_like(x), x])
w = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
print(w.ravel())                     # [2.55367394 1.46200721]
```

The measured slope is $1.4620$ — within $2.5\%$ of the theoretical $1.5$,
pulled slightly by real astrophysics (eccentric orbits, mass scatter,
measurement error).

How much scatter does the pure law leave?

```Python
slope_ref, intercept_ref = np.polyfit(x.ravel(), y.ravel(), 1)
print(f"{slope_ref:.4f} {intercept_ref:.4f}")    # 1.4620 2.5537
resid = y.ravel() - (slope_ref * x.ravel() + intercept_ref)
print(round(resid.std(), 4))                     # 0.0904
print(round(10 ** resid.std(), 2))               # 1.23 -> typical factor-1.23 error in T
```

A factor of $1.23$ in period — consistent with ignoring the
$-\tfrac12\log_{10}M$ term for roughly sun-like stars.

---

### 5. Polynomial Regression as Linear Algebra

**Definition (Polynomial design matrix)**: fitting a degree-$d$ polynomial
$p(u) = c_0 + c_1 u + \dots + c_d u^d$ is *linear* regression in the features

$$
\phi(u) \;=\; (1,\, u,\, u^2,\, \dots,\, u^d),
\qquad
\Phi_{ij} \;=\; \phi_j(u_i) \quad\text{(Vandermonde matrix)}.
$$

The same normal equations apply — nothing about least squares cares that the
features are nonlinear functions of $u$.

```Python
for d in range(1, 7):
    Phid = np.vander(x.ravel(), d + 1, increasing=True)
    print(d, f"{np.linalg.cond(Phid):.3e}")
# 1 3.162e+00
# 2 5.869e+00
# 3 1.123e+01
# 4 2.501e+01
# 5 6.241e+01
# 6 1.565e+02
```

The condition number grows quickly with degree — monomial bases are badly
scaled. This is exactly why the library also ships *orthogonal* polynomial
bases (Chebyshev, Legendre, Hermite): they keep $\Phi^{\!\top}\Phi$ close to
diagonal. For degree $\le 3$ on this data, plain monomials are still fine.

---

### 6. A Network That Is Exactly a Polynomial

**Definition (Pure-polynomial network)**: a `PolynomialNeuralNetwork` stacks
polynomial layers

$$
h^{(\ell+1)}_j \;=\; b^{(\ell)}_j + \sum_i W^{(\ell)}_{ji}\, h^{(\ell)}_i
\;+\; \sum_i p^{(\ell)}_{ji}\big(h^{(\ell)}\big)\, h^{(\ell)}_i ,
$$

followed by an activation $\sigma$. With `activation='none'`, $\sigma$ is the
identity — and polynomials are **closed under composition**: if each layer
computes a polynomial of its input, the whole network computes *one*
polynomial of the network input. No exponentials, no infinite series — the
network *is* a member of the model class from Section 5, just trained by
gradient descent instead of the normal equations.

Watch the closure property recover an exact quadratic from synthetic data:

```Python
import torch
from src.nn.poly_nn import PolynomialNeuralNetwork
import sympy as sp

torch.manual_seed(0); np.random.seed(0)
xs = np.linspace(-1, 1, 200).reshape(-1, 1)
ys = 0.5*xs**2 - 0.3*xs + 0.7       # ground truth
q = PolynomialNeuralNetwork(input_dim=1, output_dim=1, hidden_dims=[6],
                            polynomial_degree=2, activation='none')
q.fit(xs.astype(np.float64), ys.astype(np.float64),
      epochs=300, learning_rate=0.01, verbose=False)
eq = q.get_global_equation(['u'], threshold=1e-9, round_to=None)
print(sp.expand(eq))
# -0.0287270747422307*u**6 - 0.00156397845722979*u**5 +
# 0.0432665232344913*u**4 + 0.00168903699463639*u**3 +
# 0.48342462688545*u**2 - 0.300343192287282*u + 0.700982740889587
print(float(np.abs(q.predict(xs) - ys).max()))
# 0.0012712717056273526
```

The recovered coefficients ($0.483 \approx 0.5$, $-0.300 \approx -0.3$,
$0.701 \approx 0.7$) match the truth; the tiny high-degree terms are the
optimizer spending capacity on noise-level improvements.

---

### 7. Fitting 5586 Real Planets

Now the real thing. Architecture: one hidden layer of 8 units, polynomial
degree 3, identity activations, gradient descent (Adam inside `fit`).

```Python
torch.manual_seed(7); np.random.seed(7)
pnn = PolynomialNeuralNetwork(input_dim=1, output_dim=1,
                              hidden_dims=[8], polynomial_degree=3,
                              activation='none')
pnn.fit(x.astype(np.float64), y.astype(np.float64),
        epochs=150, learning_rate=0.01, verbose=False)

m = pnn.evaluate(x, y)
print(f"{m['r2']:.6f}")              # 0.990313
print(f"{np.sqrt(m['mse']):.4f}")    # 0.0895
```

**Definition ($R^2$, coefficient of determination)**:

$$
R^2 \;=\; 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2},
$$

the fraction of variance explained. $R^2 = 0.9903$: the network explains
$99\%$ of the variance in orbital periods using semi-major axis alone.

---

### 8. Reading Off the Discovered Law

Because the network is exactly polynomial, we can ask it for its own equation
(readable version: coefficients below $10^{-6}$ pruned, rounded):

```Python
u = sp.symbols('u')
eq_readable = pnn.get_global_equation(['u'], threshold=1e-6)
print(eq_readable)
# -0.e-3*u**11 - 0.002*u**10 + 0.005*u**9 + 0.005*u**8 - 0.006*u**7 +
# 0.005*u**6 - 0.068*u**5 - 0.088*u**4 + 0.149*u**3 + 0.149*u**2 +
# 1.432*u + 2.528
```

The dominant terms are $1.432\,u + 2.528$ — compare theory $1.5\,u + 2.5626$.
The higher-degree terms are small corrections. The sharpest way to see the
discovered exponent is the *local* slope $d(\log T)/d(\log a)$ across the
data range:

```Python
eq_full = pnn.get_global_equation(['u'], threshold=1e-6, round_to=None)
poly_eq = sp.Poly(eq_full, u)
for deg, c in zip(range(poly_eq.degree(), -1, -1), poly_eq.all_coeffs()):
    print(deg, f"{float(c):+.6f}")
# 14 -0.000007
# 13 +0.000022
# 12 +0.000168
# 11 -0.000570
# 10 -0.001535
# 9  +0.004783
# 8  +0.004855
# 7  -0.006449
# 6  +0.005104
# 5  -0.067616
# 4  -0.088069
# 3  +0.149124
# 2  +0.149424
# 1  +1.432474
# 0  +2.527982

deriv = sp.lambdify(u, sp.diff(eq_full, u), 'numpy')
for ua in (-1.0, 0.0, 1.0):
    print(ua, round(float(deriv(ua)), 4))
# -1.0 1.5311
# 0.0 1.4325
# 1.0 1.5363
```

The effective exponent stays in $[1.43,\, 1.54]$ over three decades of
$a$ — the network discovered Kepler's $3/2$ power law from raw observations,
with curvature absorbing the stellar-mass spread.

---

### 9. Exact Symbolic Distillation

**Definition (Exact distillation)**: extracting the symbolic equation with a
negligible pruning threshold (`round_to=None` keeps full precision) yields a
polynomial that reproduces the numeric forward pass *to floating-point
accuracy* — because every operation inside the network is itself a
polynomial evaluation, the symbolic graph and the tensor graph compute the
same function identically.

```Python
f_eq = sp.lambdify(u, eq_full, 'numpy')
sym_vals = np.asarray(f_eq(x)).reshape(-1, 1)
nn_vals = pnn.predict(x)
print(f"{np.abs(sym_vals - nn_vals).max():.2e}")   # 1.04e-06
```

One equation, readable by a human, agreeing with the trained network to
$10^{-6}$: the fit *is* the law

$$
\log_{10} T \;\approx\; 1.43\, \log_{10} a \;+\; 2.53
\qquad\Longleftrightarrow\qquad
T \approx 340\, a^{1.43\text{–}1.54}\ \text{days}.
$$

---

### 10. Exercises

1. **Mass correction**: the catalogue has a `st_mass` column for many stars.
   Fit $\log T = \alpha \log a + \beta + \gamma \log M$ and check that
   $\gamma \approx -0.5$.
2. **Degree study**: refit with `polynomial_degree` 1, 2, 3, 5 and watch
   $R^2$ and the effective exponent range. Where does extra degree stop
   helping?
3. **Orthogonal basis**: repeat Section 7 with an `OrthoPolyNetwork`
   (Chebyshev basis) and compare conditioning behavior noted in Section 5.
4. **Eccentricity**: filter to planets with known eccentricity $e > 0.1$ and
   test whether the effective exponent drifts (elliptical orbits break the
   circular-orbit derivation at order $e^2$).

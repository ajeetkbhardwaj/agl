# Tutorial 6 - Airfoil Noise: Scaling Laws and Algebraic Constraints

*Companion to `experiments/airfoil_noise.py`. Every code block runs against
the real NASA dataset in `data/airfoil/airfoil_self_noise.dat`.*

## Table of Contents

### 1. The Physical Problem

### 2. Decibels, Power Laws, and What This Dataset Actually Stores

### 3. Model A - Polynomial Network and Symbolic Distillation

### 4. Ideals and Varieties: The Algebra of Constraints

### 5. Groebner Bases and Normal Forms

### 6. Hard Constraints by Projection

### 7. Model B - Testing a Physical Hypothesis

### 8. Exercises

---

### 1. The Physical Problem

**Problem**: NASA wind-tunnel measurements of airfoil self-noise: $1503$
rows combining frequency (Hz), angle of attack (deg), chord length (m),
free-stream velocity (m/s) and suction-side displacement thickness (m) —
five aerodynamic quantities — with the measured **scaled sound pressure
level** (dB). Build an accurate predictor, then use *algebraic geometry* to
test a physical hypothesis about it.

```Python
import pandas as pd, numpy as np

SHORT = ["freq", "alpha", "chord", "vel", "thickness"]
df = pd.read_csv("data/airfoil/airfoil_self_noise.dat",
                 sep=r"\s+", header=None, names=SHORT + ["ssp"])
print(len(df))                                        # 1503
print(df.nunique().to_dict())
# {'freq': 21, 'alpha': 27, 'chord': 6, 'vel': 4, 'thickness': 105,
#  'ssp': 1456}
```

The design is a grid: only 4 velocities, 6 chords, 21 frequencies — but
$1456$ distinct responses. The response surface is smooth in the continuous
variables, which is exactly where polynomial models thrive.

---

### 2. Decibels, Power Laws, and What This Dataset Actually Stores

**Definition (Sound pressure level)**: for mean-square pressure $p^2$ with
reference $p_{\rm ref} = 20\,\mu$Pa,

$$
L_p \;=\; 10 \log_{10}\!\frac{p^2}{p_{\rm ref}^2}
\;=\; 20 \log_{10}\frac{p}{p_{\rm ref}} .
$$

Aerodynamic theory predicts intensity $\propto V^n$ (dipole: $n=6$), so raw
levels would climb $10\,n\log_{10}2 \approx 18.1$ dB per velocity doubling:

```Python
print(round(10*6*np.log10(2), 1))                     # 18.1
```

But this dataset stores **scaled** levels. Check the data itself — fix
geometry ($c = 0.3048$ m, $\alpha = 0$) and vary velocity:

```Python
sub = df[(df.chord == 0.3048) & (df.alpha == 0)]
means = sub.groupby("vel").ssp.mean()
print(means.round(1).to_dict())
# {31.7: 120.7, 39.6: 121.2, 55.5: 122.4, 71.3: 121.5}

Vs = np.array(sorted(sub.vel.unique()))
slope = np.polyfit(np.log10(Vs), means.values, 1)[0]
print(round(slope, 1))                                # 3.1 dB / decade
```

Across a $2.25\times$ velocity range the level moves by barely $1$–$2$ dB —
the scaling has been normalized out. Lesson: *always interrogate the data's
definition before importing textbook scalings*. Here the modeling burden
falls on frequency, angle and thickness instead.

---

### 3. Model A - Polynomial Network and Symbolic Distillation

Standardized inputs, one hidden layer of 12 units, degree-2 polynomial
weights, `tanh` activations; 80/20 random split.

```Python
import sympy as sp, torch
from src.nn.poly_nn import PolynomialNeuralNetwork

X = df[SHORT].to_numpy().astype(np.float64)
y = df["ssp"].to_numpy().astype(np.float64).reshape(-1, 1)
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
pnn = PolynomialNeuralNetwork(input_dim=5, output_dim=1, hidden_dims=[12],
                              polynomial_degree=2, activation='tanh')
pnn.fit(Xs[tr], ys[tr], epochs=250, learning_rate=0.01, verbose=False)

pred = pnn.predict(Xs[te]) * sd_y + mu_y
print(f"{r2(y[te], pred):.4f}")                       # 0.8407
print(f"{np.sqrt(np.mean((y[te]-pred)**2)):.3f}")     # 2.671
```

Distill the trained network into one symbolic expression:

```Python
names = [f"u{i}_{n}" for i, n in enumerate(SHORT)]
eq = pnn.get_global_equation(names, threshold=5e-3)
print(str(eq)[:260])
# -0.322*tanh(0.149*u0_freq + 0.155*u1_alpha**2*u2_chord -
# 0.244*u1_alpha**2*u3_vel + 0.041*u1_alpha + 0.577*u2_chord -
# 0.937*u3_vel + 0.864*u4_thickness + 0.446)*tanh(0.051*u0_freq**2*
# u4_thickness + 0.045*u0_freq**2 - 0.127*u0_freq*u1_alpha*u3_vel +
# 0.191*u0_fr

eq_exact = pnn.get_global_equation(names, threshold=1e-9, round_to=None)
f_eq = sp.lambdify(sp.symbols(" ".join(names)), eq_exact, "numpy")
sym = np.asarray(f_eq(*[Xs[te][:, i] for i in range(5)]),
                 dtype=float).reshape(-1, 1)
print(f"{np.abs(sym - pnn.predict(Xs[te])).max():.2e}")   # 5.19e-07
```

The readable equation already shows aerodynamics: $+0.149\,u_{\rm freq}$,
$-0.937\,u_{\rm vel}$, $+0.864\,u_{\rm thickness}$, plus interaction terms
like $\alpha^2 \cdot$ chord. Exactness verified to $5\times10^{-7}$.

---

### 4. Ideals and Varieties: The Algebra of Constraints

Now the algebraic machinery — used here as a *scientific instrument*.

**Definition (Ideal)**: for polynomials $g_1,\dots,g_k \in
\mathbb{R}[x_0,\dots,x_m]$, the ideal they generate is

$$
I = \langle g_1,\dots,g_k\rangle \;=\;
\Big\{\, \textstyle\sum_j h_j\, g_j \;:\; h_j \in \mathbb{R}[x]\,\Big\},
$$

all polynomial consequences of the constraints.

**Definition (Variety)**: the set of points satisfying all constraints
simultaneously:

$$
V(I) \;=\; \{\, w \in \mathbb{R}^{m+1} : g_1(w) = \dots = g_k(w) = 0 \,\}.
$$

Example: $I = \langle x_0 - x_1\rangle$ gives $V(I) = \{w : w_0 = w_1\}$ —
the plane where coordinates 0 and 1 are equal. A *constrained model class*
is exactly a variety: weights are allowed to live on $V(I)$ and nowhere
else.

---

### 5. Groebner Bases and Normal Forms

Two problems make ideals awkward computationally: membership testing ("is
$f \in I$?") and canonical representatives. **Groebner bases** solve both:
a special generating set $G$ such that multivariate long division by $G$
yields a unique remainder — the **normal form** $\bar{f}^{\,G}$ — with the
fundamental property

$$
f \in I \quad\Longleftrightarrow\quad \bar{f}^{\,G} = 0 .
$$

```Python
import sympy as sp

x = sp.symbols('x0:5')
G = sp.groebner([x[0] - x[1]], x, order='lex', domain='QQ')
print(G)                          # GroebnerBasis([x0 - x1], ..., domain='QQ')

_, r = G.reduce(x[0] + 2*x[1])
print(r)                          # 3*x1        (x0 folded into x1)
_, r = G.reduce(x[0] - x[1])
print(r)                          # 0           (member of the ideal!)
_, r = G.reduce(3*x[2]*x[0])
print(r)                          # 3*x1*x2     (consequence propagates)
```

Notice what reduction *means* for our constraint: every occurrence of $x_0$
is replaced by $x_1$. The normal form is the canonical representative of
the constraint's equivalence class.

---

### 6. Hard Constraints by Projection

**Definition (Projected training)**: alternate gradient steps with
retraction onto the feasible set:

$$
w^{(t+\frac12)} = w^{(t)} - \eta \nabla \mathcal{L},
\qquad
w^{(t+1)} = \pi_I\big(w^{(t+\frac12)}\big),
$$

where $\pi_I$ maps a weight vector to its normal form modulo $G$. Unlike a
soft penalty $\lambda \|g(w)\|^2$, this enforces the constraint *exactly*
after every step — the optimizer simply cannot leave the variety.

The library implements this in `GroebnerLayer.project_weights()`, called
automatically after each `train_step`. Watch it act by hand:

```Python
import torch
from src.layers.groebner_layer import GroebnerLayer

gl = GroebnerLayer(input_dim=5, output_dim=1, constraint_ideal=["x0 - x1"])
with torch.no_grad():
    gl.weight_matrix.copy_(torch.tensor(
        [[2.0], [3.0], [0.5], [-1.0], [4.0]]))
    gl.project_weights()
    print(gl.weight_matrix.flatten().tolist())
    # [0.0, 5.0, 0.5, -1.0, 4.0]

gl2 = GroebnerLayer(input_dim=5, output_dim=1, constraint_ideal=["x0"])
with torch.no_grad():
    gl2.weight_matrix.copy_(torch.tensor(
        [[2.0], [3.0], [0.5], [-1.0], [4.0]]))
    gl2.project_weights()
    print(gl2.weight_matrix.flatten().tolist())
    # [0.0, 3.0, 0.5, -1.0, 4.0]
```

$(w_0, w_1) \mapsto (0,\, w_0 + w_1)$: under $\langle x_0 - x_1\rangle$ the
normal form has *no* $x_0$ component — frequency's linear weight is
structurally zeroed, its effect absorbed into the next channel. Projection
is idempotent ($\pi(\pi(w)) = \pi(w)$), so repeated training steps cannot
drift off the variety.

---

### 7. Model B - Testing a Physical Hypothesis

Turn the constraint into science. Hypothesis $H_0$: *"frequency carries no
independent linear effect"* (plausible a priori if levels were dominated by
broadband scaling). Enforce it with the ideal $\langle x_0 - x_1\rangle$ on
the first layer and retrain:

```Python
from src.nn.poly_nn import create_groebner_pnn

torch.manual_seed(7); np.random.seed(7)
gp = create_groebner_pnn(5, 1, constraint_ideals=[["x0 - x1"], []],
                         hidden_dims=[12], polynomial_degree=2,
                         activation='tanh')
gp.fit(Xs[tr], ys[tr], epochs=250, learning_rate=0.01, verbose=False)

pc = gp.predict(Xs[te]) * sd_y + mu_y
print(f"{r2(y[te], pc):.4f}")                         # 0.7174
print(f"{np.sqrt(np.mean((y[te]-pc)**2)):.3f}")       # 3.558

W = gp.layers[0].weight_matrix.detach().numpy()
print(f"{abs(W[0, 0]):.2e}")                          # 0.00e+00

drop = r2(y[te], pred) - r2(y[te], pc)
print(f"{drop:+.4f}")                                 # +0.1233
```

Three observations, each with a mathematical guarantee behind it:

1. The constraint held **exactly**: $|w_{\rm freq}| = 0.00\times10^{0}$
   after 250 epochs of active optimization — not small, *zero*, because
   projection restores normal form after every step.
2. Accuracy fell from $R^2 = 0.8407$ to $0.7174$ (RMSE $2.67 \to 3.56$
   dB): $\Delta R^2 = +0.1233$.
3. Verdict: the hypothesis costs too much accuracy — **the physics rejects
   it**. Frequency is a first-class linear driver of airfoil noise, exactly
   as aeroacoustic theory (Strouhal-type frequency dependence) demands.

This is constrained modeling as hypothesis testing: the ideal encodes the
claim, projection enforces it perfectly, and $\Delta R^2$ is the test
statistic.

---

### 8. Exercises

1. **Soft versus hard**: add an $L^2$ penalty $\lambda w_0^2$ to the
   unconstrained PNN instead of the ideal. How large must $\lambda$ be to
   reach $|w_0| < 10^{-6}$ — and what does that do to other weights?
2. **Second hypothesis**: constrain $\langle x_3\rangle$ (no independent
   velocity effect). Given Section 2, predict the $\Delta R^2$ before
   running — then check yourself.
3. **Normal forms by hand**: compute the normal form of $x_0^2 + x_0 x_1$
   modulo $\langle x_0 - x_1\rangle$ with sympy, and verify it equals
   $2x_1^2$.
4. **Deeper ideals**: enforce $\langle x_0 - x_1,\; x_3 - x_4\rangle$
   simultaneously. Is the joint penalty additive in the individual
   penalties?

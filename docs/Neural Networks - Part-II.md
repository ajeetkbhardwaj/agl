# Neural Networks - Part-II : Polynomial Architectures

## Table of Contents

### 1. The Orthogonal Polynomial Network
### 2. Training an OrthoPoly Network
### 3. Spectral Regularization with OrthoPolyTrainer
### 4. The Polynomial Neural Network (PNN)
### 5. The Global Equation Extractor
### 6. Exactness, Rounding and the `round_to` Switch
### 7. Groebner-Constrained Networks
### 8. Deep Variant and Factories

---

### 1. The Orthogonal Polynomial Network

**Definition (Spectral Network)**: An *OrthoPoly network* stacks orthogonal
polynomial layers with a bounded activation (default `tanh`, keeping signals in
$[-1,1]$ where orthogonality lives). Each layer factorizes its coefficients as

$$
e_{oid} \;=\; \sum_{r=1}^{R} U_{oir}\, C_{rid},
$$

a low-rank decomposition that tames the parameter count from
$\mathcal{O}(n^2 D)$ to $\mathcal{O}(n^2 R + n R D)$.

```Python
import numpy as np
from src.nn.orthopoly_nn import OrthoPolyNetwork, OrthoPolyConfig, OrthogonalBasisType

net = OrthoPolyNetwork(input_dim=1, output_dim=1, hidden_dims=[8],
                       max_degree=4, rank=3)
print(repr(net))       # OrthoPolyNetwork(input=1, hidden=[8], output=1)
print(net.summary().splitlines()[0])   # OrthoPoly Neural Network Summary
```

Bases are selected through an enum (strings are accepted too):

```Python
cfg = OrthoPolyConfig(input_dim=2, output_dim=1, hidden_dims=[8],
                      basis_type=OrthogonalBasisType.LEGENDRE)
net2 = OrthoPolyNetwork(cfg)
print(net2.config.basis_type.value)    # legendre
```

---

### 2. Training an OrthoPoly Network

`fit` provides a full training loop (shuffling, gradient clipping, best-weight
checkpointing); `evaluate` reports MSE/MAE/$R^2$:

```Python
import numpy as np
import torch
from src.nn.orthopoly_nn import OrthoPolyNetwork

torch.manual_seed(7)

net = OrthoPolyNetwork(input_dim=1, output_dim=1, hidden_dims=[8],
                       max_degree=4, rank=3)
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X)

hist = net.fit(X, y, epochs=50, learning_rate=0.01, verbose=False)
print(round(hist['loss'][0], 4), round(hist['loss'][-1], 4))
# 0.3637 0.0004

m = net.evaluate(X, y)
print({k: round(v, 4) for k, v in m.items()})
# {'mse': 0.0004, 'mae': 0.0154, 'r2': 0.9992}

pred = net.predict(np.array([[0.25]]))
print(round(float(pred[0][0]), 4), round(float(np.sin(2*np.pi*0.25)), 4))
# 1.0013 1.0
```

---

### 3. Spectral Regularization with OrthoPolyTrainer

**Definition (Spectral Penalty)**: High-degree coefficients are penalized by
the inverse basis norms,

$$
\mathcal{L}_{\text{spec}} \;=\; \sum_{\text{layers}}\sum_{o,i,d}
\frac{\big(C_{oid}\big)^2}{h_d},
$$

so families whose norms $h_d$ grow with $d$ (Hermite: $h_n = 2^n n!\sqrt{\pi}$)
suppress high degrees automatically. The trainer pairs this with AdamW and
cosine-annealed warm restarts:

```Python
import torch
from src.nn.orthopoly_nn import OrthoPolyTrainer

trainer = OrthoPolyTrainer(net, learning_rate=1e-3, spectral_reg=0.01)
loss = trainer.train_step(torch.tensor(X[:16], dtype=torch.float32),
                          torch.tensor(y[:16], dtype=torch.float32))
print(bool(np.isfinite(loss)))     # True
```

---

### 4. The Polynomial Neural Network (PNN)

The PNN generalizes the architecture along three axes selected by config:
plain polynomial layers, Groebner-constrained layers, or variety-constrained
layers. Activations are stored as modules inside `self.layers`, so the whole
network is one uniform sequence.

```Python
import numpy as np
import torch
from src.nn.poly_nn import (
    PolynomialNeuralNetwork, PNNConfig,
    create_pnn, create_groebner_pnn, DeepPolynomialNetwork)

torch.manual_seed(7)
np.random.seed(7)

pnn = PolynomialNeuralNetwork(input_dim=2, output_dim=1, hidden_dims=[6],
                              polynomial_degree=2, activation='tanh')

Xp = np.random.randn(120, 2)
yp = (0.8 * Xp[:, :1] ** 2 + 0.5 * Xp[:, 1:] + 0.3).astype(np.float64)
h = pnn.fit(Xp, yp, epochs=60, learning_rate=0.01, verbose=False)
print(round(pnn.evaluate(Xp, yp)['r2'], 4))          # 0.8767

print([type(l).__name__ for l in pnn.layers])
# ['PolynomialLayer', 'Tanh', 'PolynomialLayer']
print(repr(pnn))                                     # PNN(input=2, hidden=[6], output=1)
```

After each training step the network re-projects any constrained layer onto
its variety/ideal — the "constraint promise" that hard constraints survive
gradient updates.

---

### 5. The Global Equation Extractor

**Definition (Everything Equation)**: Composition of layers is function
composition; the mixin walks the layer sequence symbolically and returns one
closed-form expression $F(x_1,\dots,x_n)$ for each output, bypassing all
hidden units:

$$
F \;=\; L_k \circ a_{k-1} \circ L_{k-1} \circ \cdots \circ a_1 \circ L_1 ,
$$

where each polynomial layer contributes its exact symbolic form, each
orthogonal layer its Chebyshev expansion under the running normalization, and
each activation its sympy lift (`tanh`, `sigmoid`, `SiLU`, `GELU`, `ReLU`).

```Python
eq = pnn.get_global_equation(['a', 'b'], threshold=5e-3)
print(str(eq)[:96])
# -0.175*tanh(-0.265*a + 0.035*b + 0.375)**2*tanh(0.495*a**2 + 0.407*a +
# 0.404*b + 0.08) + 0.686*t
```

The extracted equation is not an approximation scheme — it is the network,
rewritten. Evaluating it agrees with `forward` to float precision (next
section).

---

### 6. Exactness, Rounding and the `round_to` Switch

By default coefficients are rounded to three decimals for readability. Pass
`round_to=None` for the numerically exact equation:

```Python
import sympy as sp

eq_exact = pnn.get_global_equation(['a', 'b'], threshold=1e-6, round_to=None)
f = sp.lambdify(sp.symbols('a b'), eq_exact, 'numpy')
sym_vals = np.asarray(f(Xp[:, 0], Xp[:, 1]), dtype=float)

nn_vals = pnn.predict(Xp).squeeze()
print(np.abs(nn_vals - sym_vals).max() < 1e-4)      # True  (max diff ~4.1e-7)
```

Two honest caveats:

- thresholds (`threshold`) prune negligible terms — set them small for exact
  agreement;
- the domain-normalization clamp appears as `Max(-1.0, Min(1.0, ...))`
  whenever inputs fall outside the tracked range.

---

### 7. Groebner-Constrained Networks

Factories wire constrained layers into full networks. Constraint ideals are
given per layer as sympy-evaluable strings over the flattened weight
variables `x0, x1, ...`:

```Python
gp = create_groebner_pnn(2, 2, constraint_ideals=[["x0 - x1"], []],
                         hidden_dims=[4], activation='none')
print([type(l).__name__ for l in gp.layers])
# ['GroebnerLayer', 'Identity', 'GroebnerLayer']
```

Every `train_step` ends with `project_weights()`, restoring each weight matrix
to its normal form modulo the constraint ideal after gradients have flowed.

---

### 8. Deep Variant and Factories

```Python
std = create_pnn(3, 2, hidden_dims=[8], polynomial_degree=2)
print(repr(std))                    # PNN(input=3, hidden=[8], output=2)

dp = DeepPolynomialNetwork(PNNConfig(input_dim=2, output_dim=1,
                                     hidden_dims=[6]), residual=True)
print(isinstance(dp, PolynomialNeuralNetwork))    # True
```

**Problem**: A PNN with two polynomial layers of degree $d$ per connection has
an exact global equation. What is its maximal degree in the inputs?

**Solution**: The composition of degree-$d$ maps has degree at most $d^k$
after $k$ layers ($d^2$ here), multiplied by the extra factor $x_i$ each
polynomial weight carries — bounded by $d(d+1)^{k-1}$ in general. Use
`max_global_degree` in `get_global_equation` to truncate the extracted
expression at any chosen physical degree.

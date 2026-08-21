# Neural Networks - Part-III : Training Utilities

## Table of Contents

### 1. Classical Loss Functions
### 2. Ideal Membership Loss
### 3. Variety Constraint Loss
### 4. Syzygy Loss
### 5. Gradient Correctness by Finite Differences
### 6. The Loss Factory
### 7. NumPy Optimizers on the Layer Protocol
### 8. Momentum and Adaptive Methods
### 9. The Polynomial-Aware Optimizer and Variety Projection
### 10. Bridging into PyTorch
### 11. Early Stopping

---

### 1. Classical Loss Functions

All losses follow one protocol — `__call__(y_true, y_pred) -> float` plus an
analytic `gradient(y_true, y_pred)` with respect to predictions:

$$
L_{\text{MSE}} = \frac{1}{N}\sum_k (y_k - \hat y_k)^2,
\qquad
L_{\text{Huber}} = \frac{1}{N}\sum_k
\begin{cases}
\frac{1}{2}e_k^2 & |e_k|\le\delta\\
\delta(|e_k| - \frac{\delta}{2}) & \text{else}
\end{cases}
$$

```Python
import numpy as np
from src.training.loss_functions import (
    MeanSquaredError, MeanAbsoluteError, HuberLoss, HingeLoss, CrossEntropy)

yt = np.array([1.0, 2.0, 3.0])
yp = np.array([1.5, 2.5, 2.0])

print(MeanSquaredError()(yt, yp))       # 0.5
print(MeanAbsoluteError()(yt, yp))      # 0.6666666666666666
print(HuberLoss(delta=1.0)(np.array([0.0]), np.array([3.0])))   # 2.5
print(HingeLoss()(np.array([1.0, -1.0]), np.array([0.5, -0.2])))  # 0.65
```

Binary cross-entropy clips predictions away from $\{0,1\}$; multi-class mode
applies a numerically stable softmax to raw logits:

```Python
ce = CrossEntropy()
print(round(ce(np.array([1.0, 0.0]), np.array([0.9, 0.1])), 6))
# 0.105361     -(log(0.9) + log(0.9)) / 2

cem = CrossEntropy(multi_class=True)
logits = np.array([[1.0, 2.0, 3.0]])
probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
val = cem(probs, logits)
expected = float(-np.sum(probs * np.log(probs)))    # entropy H(q)
print(round(val, 6), round(expected, 6))            # 0.832396 0.832396
```

---

### 2. Ideal Membership Loss

**Definition (Constraint Violation)**: Given generators $g_1,\dots,g_r$ of an
ideal $I$ (as callables over predictions), the loss adds the squared norm of
the violation vector:

$$
L \;=\; \frac{1}{N}\sum_k (y_k-\hat y_k)^2
\;+\; \lambda \cdot \frac{1}{N}\sum_k \sum_j g_j(\hat y_k)^2 .
$$

Predictions already on $V(I)$ pay no penalty; off-variety points are pulled
back through the gradient term.

```Python
import numpy as np
from src.training.loss_functions import IdealMembershipLoss

gen = lambda Y: Y[..., 0]**2 + Y[..., 1]**2 - 1.0     # unit circle
iml = IdealMembershipLoss(ideal_generators=[gen], weight=1.0)

Y_true = np.zeros((4, 2))
Y_pred = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
print(round(iml(Y_true, Y_pred), 6))     # 0.875
# base MSE 0.5625  +  mean(viol^2) 0.3125   ([0,0,1,-0.5] -> sq mean .3125)

g = iml.gradient(Y_true, Y_pred)
print([round(float(v), 6) for v in g[0]])    # [0.25, 0.0]
```

---

### 3. Variety Constraint Loss

Same geometry, different bookkeeping: violations are accumulated per sample as
a distance surrogate $\sum_j g_j(\hat y)^2$, and the weight multiplies only
the geometric term:

```Python
from src.training.loss_functions import VarietyConstraintLoss

vcl = VarietyConstraintLoss(variety_repr={'generators': [gen]}, weight=2.0)
print(round(vcl(Y_true, Y_pred), 6))     # 1.1875
# base 0.5625 + 2 * 0.3125
```

---

### 4. Syzygy Loss

**Definition (Syzygy Relation)**: A syzygy is a polynomial relation
$r(y_1,\dots,y_m)=0$ that outputs must satisfy identically. The loss penalizes
$\frac{1}{N}\sum_k r(\hat y_k)^2$ for each relation:

```Python
from src.training.loss_functions import SyzygyLoss

rel = lambda Y: Y[..., 0] * Y[..., 1] - 1.0     # outputs must be reciprocal
sl = SyzygyLoss(syzygy_relations=[rel], weight=1.0)
print(round(sl(Y_true, Y_pred), 6))
# 1.203125
```

---

### 5. Gradient Correctness by Finite Differences

The algebraic gradients use central differences internally where closed forms
are impractical; the external contract is that `gradient` matches the numeric
derivative of `__call__`:

```Python
import numpy as np

eps = 1e-6
g_fd = np.zeros_like(Y_pred)
for i in range(Y_pred.shape[0]):
    for j in range(Y_pred.shape[1]):
        yp_plus = Y_pred.copy();  yp_plus[i, j] += eps
        yp_minus = Y_pred.copy(); yp_minus[i, j] -= eps
        g_fd[i, j] = (iml(Y_true, yp_plus) - iml(Y_true, yp_minus)) / (2*eps)

print(np.allclose(iml.gradient(Y_true, Y_pred), g_fd, atol=1e-4))    # True
```

The same check passes for `VarietyConstraintLoss` and `SyzygyLoss`.

---

### 6. The Loss Factory

```Python
from src.training.loss_functions import create_loss

lf = create_loss('huber', delta=0.5)
print(lf.name, lf.delta)     # huber 0.5
```

Unknown names raise immediately with the list of available losses.

---

### 7. NumPy Optimizers on the Layer Protocol

**Definition (Layer Protocol)**: The custom optimizers update any model whose
layers expose `weight_matrix`, `grad_weight` (and optionally `bias`,
`grad_bias`) as NumPy arrays:

$$
w^{(t+1)} = w^{(t)} - \eta\, m^{(t)}, \qquad
m^{(t)} = \beta m^{(t-1)} + \nabla L(w^{(t)})
$$

Updates are clipped elementwise to $[-10,10]$ for stability.

```Python
import numpy as np
from src.training.optimizers import SGD

class DummyLayer:
    def __init__(self):
        self.weight_matrix = np.zeros((1, 1))
        self.grad_weight = None
class DummyModel:
    def __init__(self):
        self.layers = [DummyLayer()]

mdl = DummyModel()
sgd = SGD(learning_rate=0.05)
for _ in range(200):
    w = mdl.layers[0].weight_matrix
    mdl.layers[0].grad_weight = 2 * (w - 3.0)
    sgd.step(mdl)
print(round(float(mdl.layers[0].weight_matrix[0, 0]), 4))    # 3.0
```

---

### 8. Momentum and Adaptive Methods

Adam keeps exponential moving averages of first and second moments with bias
correction:

$$
m^{(t)} = \beta_1 m^{(t-1)} + (1-\beta_1) g,\quad
v^{(t)} = \beta_2 v^{(t-1)} + (1-\beta_2) g^2,
$$
$$
w \leftarrow w - \eta\, \frac{m^{(t)}/(1-\beta_1^t)}
{\sqrt{v^{(t)}/(1-\beta_2^t)} + \epsilon}.
$$

```Python
from src.training.optimizers import Adam, SGD

mdl = DummyModel()
opt = Adam(learning_rate=0.1)
for _ in range(200):
    w = mdl.layers[0].weight_matrix
    mdl.layers[0].grad_weight = 2 * (w - 3.0)
    opt.step(mdl)
print(round(float(mdl.layers[0].weight_matrix[0, 0]), 6))    # 3.000053

m2 = DummyModel()
nes = SGD(learning_rate=0.05, momentum=0.9, nesterov=True)
for _ in range(200):
    w = m2.layers[0].weight_matrix
    m2.layers[0].grad_weight = 2 * (w - 3.0)
    nes.step(m2)
print(round(float(m2.layers[0].weight_matrix[0, 0]), 4))     # 3.0
```

`RMSprop` and `AdaGrad` follow the same protocol with their classical
per-coordinate scalings.

---

### 9. The Polynomial-Aware Optimizer and Variety Projection

**Definition (Projection Step)**: After a gradient step, weights can be
projected onto a variety by iterating

$$
w \leftarrow w - \alpha\, J_g(w)^\top g(w),
$$

with the Jacobian estimated by central differences. Iterating drives
$\|g(w)\|$ below tolerance:

```Python
import numpy as np
from src.training.optimizers import PolynomialAwareOptimizer

pao = PolynomialAwareOptimizer(learning_rate=0.01)
circle_gen = lambda W: W[0, 0]**2 + W[0, 1]**2 - 1.0

Wp = pao.project_to_variety(np.array([[2.0, 0.0], [0.0, 0.0]]),
                            {'generators': [circle_gen],
                             'max_iterations': 500})
print(bool(circle_gen(Wp) < 1e-3))                    # True
print([round(float(v), 4) for v in Wp[0]])            # [1.0, 0.0]
```

The optimizer's `step(model)` applies this projection automatically whenever
the model exposes per-layer `variety_representations`.

---

### 10. Bridging into PyTorch

`PyTorchOptimizer` lets the NumPy optimizers drive real `torch.nn.Parameters`
by shuttling data and gradients across the boundary each step:

```Python
import torch
from src.training.optimizers import PyTorchOptimizer, SGD

torch.manual_seed(7)
pt_opt = PyTorchOptimizer(SGD, learning_rate=0.1)
lin = torch.nn.Linear(2, 1)
pt_opt.add_param_group(lin.parameters())

xin = torch.randn(16, 2); yin = torch.randn(16, 1)
first = last = None
for _ in range(40):
    pt_opt.zero_grad()
    loss = torch.nn.functional.mse_loss(lin(xin), yin)
    first = first if first is not None else loss.item()
    loss.backward()
    pt_opt.step()
    last = loss.item()
print(round(first, 4), round(last, 4))       # 0.9552 0.7274
```

---

### 11. Early Stopping

**Definition (Patience Rule)**: Stop when the monitored loss has failed to
improve by `min_delta` for `patience` consecutive checks:

```Python
from src.layers.polynomial_layer import EarlyStopping

es = EarlyStopping(patience=2)
seq = [1.0, 0.9, 0.95, 0.96, 0.97]
flags = [es(v) for v in seq]
print(flags)
# [False, False, False, True, True]
#   best=1.0 best=0.9 | no improvement twice -> stop signaled
```

**Problem**: Why does `IdealMembershipLoss` need a finite-difference Jacobian
while its MSE part has an exact gradient?

**Solution**: For arbitrary polynomial generators $g_j$ the chain rule needs
$\partial g_j/\partial \hat y$ evaluated at each prediction. Rather than
symbolically differentiating user-supplied callables, the implementation
estimates $J_g$ centrally once per call — second-order accurate at cost
$2 \times n_{\text{out}}$ evaluations — and reuses it for all samples.

# Neural Networks - Part-I : Algebraic Layers

## Table of Contents

### 1. The Polynomial Layer

### 2. Forward Pass of a Polynomial Layer

### 3. Symbolic Extraction and Dense Weights

### 4. Orthogonal Polynomial Layers

### 5. Classical Basis Families

### 6. From Recurrence to Monomial Coefficients

### 7. Adaptive Domain Normalization

### 8. Injecting Symbolic Weights into an Orthogonal Layer

### 9. The Rational Polynomial Layer

### 10. Groebner-Constrained Layers

### 11. Variety-Constrained Layers

### 12. Product Varieties and Tangent Spaces

---

### 1. The Polynomial Layer

**Definition (Polynomial Weight)**: A *polynomial layer* replaces the constant
weight matrix of a dense layer by a matrix of polynomials. Each connection
$(i \to j)$ carries a polynomial weight $p_{ij} \in \mathbb{R}[x_1,\dots,x_n]$
of total degree at most $d$:

$$
y_j \;=\; b_j + \sum_{i=1}^{n} W_{ji}\, x_i \;+\; \sum_{i=1}^{n} p_{ij}(x)\, x_i ,
$$

so every coefficient of every $p_{ij}$ is a learnable parameter registered in a
`ParameterDict`.

```Python
import torch
from src.layers.polynomial_layer import PolynomialLayer

torch.manual_seed(7)

layer = PolynomialLayer(input_dim=2, output_dim=2, max_degree=2, seed=42)
x = torch.tensor([[0.5, -1.0], [2.0, 0.5]])
out = layer(x)
print(out.shape)          # torch.Size([2, 2])
print(out[0].tolist())    # [1.0886788368225098, 0.6600738167762756]
```

The layer also supports batch normalization and dropout, controlled by
`use_batchnorm` and `dropout_rate`, and three random initialization regimes via
`polynomial_type`: `'general'`, `'homogeneous'` (single monomial of degree $d$)
and `'quadratic'`.

---

### 2. Forward Pass of a Polynomial Layer

The forward pass evaluates every monomial with pure tensor operations so that
gradients flow exactly through the polynomial structure:

$$
p_{ij}(x) \;=\; \sum_{\alpha} c^{(ij)}_{\alpha}\, x^\alpha
\qquad\Longrightarrow\qquad
\frac{\partial y_j}{\partial c^{(ij)}_{\alpha}} \;=\; x^{\alpha}\, x_i .
$$

Because the map is smooth in both inputs and coefficients, it passes
`torch.autograd.gradcheck`:

```Python
layer.double()                       # gradcheck wants float64
ok = layer.check_gradients()         # finite-difference verification
layer.float()
print(ok)                            # True
```

---

### 3. Symbolic Extraction and Dense Weights

Each $p_{ij}$ lives as a genuine `alggeom.Polynomial`, so the layer can hand
back its own symbolic form:

```Python
polys = layer.symbolic_forward()     # one output polynomial per output unit
print(len(polys))                    # 2
print(str(polys[0])[:52])            # -0.1118*x0^3 - 1.3796*x0 * x1 - 0.7354*x0

dense = layer.to_dense_weights()     # p_ij evaluated at x = 0
print(dense.shape)                   # (2, 2)

info = layer.get_polynomial_info()
print(info['total_polynomials'])     # 4
print(info['polynomial_details'][0]['degree'])   # 2
```

`symbolic_forward` returns $\sum_i p_{ij}(x)\, x_i$ per output; adding the
linear part $W x + b$ reproduces the numeric forward pass identically.

---

### 4. Orthogonal Polynomial Layers

**Definition (Three-Term Recurrence)**: Let $\{P_n\}_{n\ge 0}$ be a family of
orthogonal polynomials on an interval $I$. Every classical family satisfies a
recurrence

$$
P_{n+1}(x) \;=\; \big(A_n x + B_n\big) P_n(x) \;-\; C_n P_{n-1}(x),
\qquad P_0 = 1,\; P_1 = x .
$$

`OrthoPolyLayer` stores $(A_n, B_n, C_n)$ as buffers and evaluates all degrees
$0..D$ per input feature in one vectorized sweep:

```Python
import torch
from src.layers.orthopoly_layer import OrthoPolyLayer

op = OrthoPolyLayer(input_dim=1, output_dim=1, max_degree=4, rank=2,
                    basis_type="chebyshev_T", interaction_mode="additive")
op.eval()
with torch.no_grad():                    # isolate the degree-1 basis element
    op.linear_weight.zero_(); op.cheby_coeffs.zero_(); op.bias.zero_()
    op.linear_weight[0, 0, 0] = 1.0      # rank-mixing weight
    op.cheby_coeffs[0, 0, 1] = 1.0       # spectral coefficient of T1 = x

xs = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
with torch.no_grad():
    vals = op(xs).squeeze().tolist()
print([round(v, 4) for v in vals])
# [-1.0, -1.0, -1.0, 0.0, 1.0]
```

(The first three entries coincide because the domain normalization below maps
$x = -1, -0.5, 0$ to the same clamped value $-1$; with only $P_1 = x$ selected,
the output is exactly the normalized input.)

Two interaction modes are available:

- **`additive`** – each output is a sum of univariate expansions,
  $y_o=\sum_i \sum_d e_{oid} P_d(\tilde x_i)$;
- **`multiplicative`** – rank-1 cross terms are multiplied across features
  (Segre embedding), capturing interactions at exponential expressive gain.

---

### 5. Classical Basis Families

| Family          | Interval       | Recurrence constants                                                       |
| --------------- | -------------- | -------------------------------------------------------------------------- |
| `chebyshev_T` | $[-1,1]$     | $A_n{=}2,\ C_n{=}1$                                                      |
| `chebyshev_U` | $[-1,1]$     | $A_n{=}2,\ C_n{=}1$                                                      |
| `legendre`    | $[-1,1]$     | $A_n{=}\frac{2n+1}{n+1},\ C_n{=}\frac{n}{n+1}$                           |
| `hermite`     | $\mathbb{R}$ | $A_n{=}\sqrt{2},\ C_n{=}2n$                                              |
| `laguerre`    | $[0,\infty)$ | $A_n{=}-\tfrac{1}{n+1},\ B_n{=}\tfrac{2n+1}{n+1},\ C_n{=}\tfrac{n}{n+1}$ |

Picking out a single basis element confirms the recurrence numerically — here
Legendre $P_2(x) = \tfrac{1}{2}(3x^2-1)$:

```Python
opl = OrthoPolyLayer(input_dim=1, output_dim=1, max_degree=2, rank=1,
                     basis_type="legendre", interaction_mode="additive")
opl.eval()
with torch.no_grad():
    opl.linear_weight.zero_(); opl.cheby_coeffs.zero_(); opl.bias.zero_()
    opl.linear_weight[0, 0, 0] = 1.0
    opl.cheby_coeffs[0, 0, 2] = 1.0        # select degree 2 only
    p2 = opl(xs).squeeze().tolist()

xn = [max(-1.0, min(1.0, 2*float(v)-1)) for v in xs.squeeze().tolist()]
print([round(a, 4) for a in p2])           # [1.0, 1.0, 1.0, -0.5, 1.0]
print([round((3*s**2 - 1)/2, 4) for s in xn])  # [1.0, 1.0, 1.0, -0.5, 1.0]
```

Unbounded families report their interval honestly:

```Python
oh = OrthoPolyLayer(input_dim=1, output_dim=1, basis_type="hermite")
print(tuple(oh.interval.tolist()))         # (-inf, inf)
```

---

### 6. From Recurrence to Monomial Coefficients

**Definition (Basis Transform)**: If $P_d(x)=\sum_k M_{dk} x^k$, the matrix $M$
converts spectral coefficients to ordinary monomial coefficients:

$$
\sum_d c_d P_d(x) \;=\; \sum_k \Big(\sum_d c_d M_{dk}\Big) x^k .
$$

$M$ is built once from the recurrence and stored as a buffer. Its rows are the
familiar Chebyshev expansions:

```Python
M = op.basis_transform.numpy()
print(M[2].tolist())    # [-1.0, 0.0, 2.0, 0.0, 0.0]   T2 = 2x^2 - 1
print(M[3].tolist())    # [0.0, -3.0, 0.0, 4.0, 0.0]   T3 = 4x^3 - 3x
```

---

### 7. Adaptive Domain Normalization

Orthogonality holds on a fixed interval, so the layer maintains exponential
moving averages of per-feature min/max and rescales inputs:

$$
\tilde x_i \;=\; \operatorname{clip}\!\left(\frac{2(x_i-\mu_i)}{s_i}-1,\,-1,\,1\right),
\qquad s_i = \sigma_i - \mu_i .
$$

Statistics update only in training mode, exactly like batch-norm running stats:

```Python
op_t = OrthoPolyLayer(input_dim=1, output_dim=1, max_degree=2, rank=2)
op_t.train()
_ = op_t(torch.tensor([[0.0], [10.0]]))
print(op_t.running_min.item(), op_t.running_max.item())   # 0.0 10.0
```

Symbolic extraction mirrors this normalization (including the clamp), which is
why extracted equations contain `Max(-1.0, Min(1.0, ...))` terms whenever the
clamp is active.

---

### 8. Injecting Symbolic Weights into an Orthogonal Layer

`get_polynomial_weights` collapses the low-rank factorization and converts to
standard polynomials; `set_polynomial_weights` performs the inverse — solving
the linear system $c_{\text{std}} = c_{\text{spec}} M$ and factorizing with a
batched SVD:

```Python
import torch
from src.layers.orthopoly_layer import OrthoPolyLayer
from src.alggeom.polynomial import Polynomial, Monomial, Variable

torch.manual_seed(17)
opg = OrthoPolyLayer(input_dim=1, output_dim=2, max_degree=3, rank=4)
opg.eval()
pw = opg.get_polynomial_weights()
print(len(pw), len(pw[0]))      # 2 1   (outputs x inputs)
print(str(pw[0][0])[:46])       # -0.5966*x0^3 - 0.5936*x0^2 + 0.9582*x0 - 0.246

# inject p(x) = 0.5 + 2x^2 on every connection (rank-1 target)
target = Polynomial({Monomial(()): 0.5, Monomial(((Variable('x'), 2),)): 2.0})
targets = [[target] for _ in range(2)]
opg.set_polynomial_weights(targets)

with torch.no_grad():
    got = opg(torch.tensor([[-1.0], [1.0]]))
print(got.flatten().tolist())
# [1.0, 1.0]  ->  0.5 + 2*(+-1)^2 evaluated per input, summed over 1 input
```

Rank-1 targets are reproduced exactly; higher-rank targets are recovered up to
the truncation of the SVD at `rank` components.

---

### 9. The Rational Polynomial Layer

**Definition (Padé-Style Quotient)**: A rational layer divides two orthogonal
expansions with a positivity guard in the denominator:

$$
R(x) \;=\; \frac{P(x)}{1 + |Q(x)|} .
$$

The absolute value keeps the denominator strictly positive, giving the layer
localized poles without division-by-zero pathologies:

```Python
import torch
from src.layers.rational_layer import RationalPolyLayer

torch.manual_seed(21)
rl = RationalPolyLayer(input_dim=1, output_dim=1, max_degree=3, rank=4)
xr = torch.linspace(-2, 2, 5).unsqueeze(1)
out = rl(xr)
print([round(v, 4) for v in out.squeeze().tolist()])
# [-0.8189, -0.1616, -0.116, -0.2396, -0.0903]

q = rl.denominator(xr)
print(bool(torch.all(1 + q.abs() > 0)))    # True
```

Because poles are localized, rational layers fit sharp spikes that oscillatory
global polynomials cannot:

```Python
torch.manual_seed(1)
rl2 = RationalPolyLayer(input_dim=1, output_dim=1, max_degree=4, rank=8)
opt = torch.optim.Adam(rl2.parameters(), lr=0.01)
xt = torch.linspace(-3, 3, 64).unsqueeze(1)
yt = 1.0 / (1.0 + 5 * xt**2)               # sharp Lorentzian peak
first = last = None
for _ in range(300):
    opt.zero_grad()
    loss = torch.mean((rl2(xt) - yt)**2)
    first = first if first is not None else loss.item()
    last = loss.item()
    loss.backward(); opt.step()
print(last < 0.1 * first)                  # True
```

---

### 10. Groebner-Constrained Layers

**Definition (Ideal Constraint)**: Given constraint polynomials
$f_1,\dots,f_r$, the weight vector $w \in \mathbb{R}^{n}$ of a
`GroebnerLayer` is reduced modulo the ideal $I=\langle f_1,\dots,f_r\rangle$.
Reduction replaces $w$ by its normal form $\bar w$ with

$$
w - \bar w \;\in\; I
\qquad\Longrightarrow\qquad
w|_{V(I)} \;=\; \bar w|_{V(I)} ,
$$

so the layer's function on the constraint variety is unchanged while the
parameterization becomes canonical.

```Python
import numpy as np
import torch
from src.layers.groebner_layer import GroebnerLayer, GroebnerResidualLayer

gl = GroebnerLayer(input_dim=2, output_dim=1, constraint_ideal=["x0 - x1"])
print(gl.groebner_basis is not None)       # True  (sympy GB over QQ)

xo = torch.tensor([[3.0, 2.0]])
with torch.no_grad():
    gl.weight_matrix.copy_(torch.tensor([[2.0], [1.0]])); gl.bias.zero_()
before = gl(xo).item()
gl.project_weights()                        # hard projection onto V(I)
after = gl(xo).item()
print(before, gl.weight_matrix.flatten().tolist(), after)
# 8.0 [0.0, 3.0] 6.0        2x0+x1  reduces to  3x1  mod <x0-x1>

# agreement ON the variety x0 = x1:
print(float(gl(torch.tensor([[3.0, 3.0]]))))    # 9.0  (= 2*3+1*3)
```

Membership queries use the same machinery — a linear form given by its
coefficient vector lies in $I$ iff its remainder vanishes:

```Python
print(gl.ideal_membership_test(np.array([1.0, -1.0])))   # True   x0-x1 in I
print(gl.ideal_membership_test(np.array([1.0,  0.0])))   # False  x0 notin I
print(gl.compute_variety_representation())
# {'groebner_basis': "GroebnerBasis([x0 - x1], ...)", 'dimension': 1, 'n_constraints': 1}
print(sorted(gl.to_symbolic().keys()))
# ['constraint_ideal', 'groebner_basis', 'groebner_order', 'has_bias',
#  'input_dim', 'output_dim', 'type', 'weight_shape']
```

A residual variant adds skip connections with optional dimension projection:

```Python
grl = GroebnerResidualLayer(input_dim=2, output_dim=2, residual_dim=2)
out = grl(np.array([[1.0, 2.0]]), residual=np.ones((1, 2)))
print(tuple(out.shape))                    # (1, 2)
```

---

### 11. Variety-Constrained Layers

**Definition (Variety Constraint)**: A `VarietyLayer` parameterizes its weight
matrix through latent coordinates $z$ and projects $z$ so that $W(z)$ lands on
an algebraic variety $V(I)$. Projection minimizes the constraint violation

$$
g(W) \;=\; \big(f_1(W),\dots,f_r(W)\big),
\qquad \|g\|_2 \;\longrightarrow\; 0
$$

by gradient descent on the Lagrangian $\mathcal{L}= f + \lambda^\top g(W(z))$,
with backtracking line search, using autograd to pull gradients back through
any parameterization.

```Python
import numpy as np
import torch
from src.layers.variety_layer import VarietyLayer
from src.alggeom.polynomial import Variable, parse_polynomial_string

vs = [Variable('x0'), Variable('x1')]
circle = parse_polynomial_string("x0^2 + x1^2 - 1", vs)

vl = VarietyLayer(input_dim=2, output_dim=2, ideal_generators=[circle])
with torch.no_grad():
    vl.latent_params.copy_(torch.tensor([2.0, 0.0, 0.5, 0.5]))

w0 = vl._compute_weight_matrix().detach().numpy().flatten()
v_before = abs(float(vl.variety.constraint_violations(w0)[0]))
vl.project_to_variety(max_iter=1000)
w1 = vl._compute_weight_matrix().detach().numpy().flatten()
v_after = abs(float(vl.variety.constraint_violations(w1)[0]))
print(round(v_before, 6), round(v_after, 12))    # 3.0 0.0

info = vl.get_variety_info()
print(info['latent_dim'], info['num_defining_equations'])   # 4 1
```

---

### 12. Product Varieties and Tangent Spaces

Constraints on disjoint variable blocks combine into a product variety
$V_1 \times V_2$: generators of each factor are index-shifted so they act on
their own coordinates.

```Python
import numpy as np
from src.layers.variety_layer import ProductVarietyLayer, TangentSpaceLayer
from src.alggeom.polynomial import Variable, parse_polynomial_string
from src.alggeom.algvariety import AlgebraicVariety

h1 = parse_polynomial_string("x0^2 + x1^2 - 1", [Variable('x0'), Variable('x1')])
h2 = parse_polynomial_string("x2^2 + x3^2 - 1", [Variable('x2'), Variable('x3')])

pv = ProductVarietyLayer(input_dim=4, output_dim=2,
                         row_varieties=[AlgebraicVariety([h1]),
                                        AlgebraicVariety([h2])])
print(pv.latent_params.shape[0])           # 8
print(pv.get_row_constraint_info())
# [{'num_generators': 1, 'dimension': 1}, {'num_generators': 1, 'dimension': 1}]
```

**Definition (Tangent Space)**: At a point $p \in V(I)$ the tangent space is
the kernel of the Jacobian $J_g(p)$; an orthonormal basis is read off the SVD
null space. `TangentSpaceLayer` precomputes this basis and projects arbitrary
updates onto it:

```Python
tsl = TangentSpaceLayer(input_dim=2, output_dim=2, variety=pv.variety,
                        base_point=np.array([1.0, 0.0, 0.0, 1.0]))
B = tsl.tangent_space_basis
print(B.shape)                              # (4, 2)  torus -> 2-dim tangent

v = np.array([1.0, 2.0, 3.0, 4.0])
vp = tsl.project_to_tangent(v)
print(np.linalg.norm(vp) <= np.linalg.norm(v) + 1e-12)   # True
print(np.allclose(tsl.project_to_tangent(vp), vp, atol=1e-8))  # True (idempotent)
```

**Problem**: The unit circle has a one-dimensional tangent space at every
point. Why does the tangent basis above have two columns?

**Solution**: The product variety constrains two disjoint circles embedded in
$\mathbb{R}^4$; its tangent space at any point is the direct sum of the two
circle tangents, hence $\dim = 1 + 1 = 2$.

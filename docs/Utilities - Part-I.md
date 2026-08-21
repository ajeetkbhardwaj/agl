# Utilities - Part-I : Symbolic Helpers and Visualization

## Table of Contents

### 1. Variables, Monomials and Polynomials

### 2. Monomial Orders and Leading Terms

### 3. Polynomial Division

### 4. GCD and LCM (Univariate)

### 5. Parsing Polynomial Strings

### 6. Coefficient Arrays

### 7. Systems of Polynomials and Their Jacobians

### 8. Numerical Gradient and Jacobian Helpers

### 9. Visualizing Varieties

### 10. Training Curves and Decision Boundaries

---

### 1. Variables, Monomials and Polynomials

The symbolic module mirrors the `alggeom` data model: a `Monomial` is a sorted
tuple of `(Variable, exponent)` pairs and a `Polynomial` is a dictionary
mapping monomials to float coefficients:

$$
p \;=\; \sum_{\alpha} c_{\alpha}\, x^{\alpha},
\qquad
x^{\alpha}\cdot x^{\beta} \;=\; x^{\alpha+\beta}.
$$

```Python
from src.utils.symbolic import Variable, Monomial, Polynomial

x, y = Variable('x'), Variable('y')

p = Polynomial({Monomial(((x, 2),)): 1.0, Monomial(((y, 1),)): -4.0})
q = Polynomial({Monomial(((x, 1),)): 2.0})

print(str(p))            # x^2 + -4.0000*y
print(p.degree())        # 2
print(str(p * q))        # 2.0000*x^3 + -8.0000*x * y
print(str(p ** 2)[:12])  # x^4 + -8.000
print([v.name for v in p.variables()])   # ['x', 'y']
```

Arithmetic is closed: subtraction cancels exactly, near-zero coefficients are
pruned at tolerance $10^{-12}$, and powers use binary exponentiation:

```Python
print((p - p).is_zero())                 # True
print(str(Polynomial.constant(5)))       # 5.0000
print(Polynomial.constant(1).is_one())   # True
print(Polynomial.constant(0).is_zero())  # True
```

---

### 2. Monomial Orders and Leading Terms

**Definition (Monomial Order)**: A total order on exponent vectors. Three
classical orders are supported via sort keys over a shared variable list
$[v_1,\dots,v_n]$:

$$
\text{lex: } (\alpha_1,\dots,\alpha_n),
\quad
\text{grlex: } (|\alpha|, \alpha),
\quad
\text{grevlex: } (|\alpha|, -\alpha_n, \dots, -\alpha_1).
$$

`lex` prioritizes earlier variables even at lower degree; `grevlex`
prioritizes total degree first:

```Python
p2 = Polynomial({Monomial(((x, 1), (y, 1),)): 3.0,
                 Monomial(((x, 2),)): 1.0})
p3 = Polynomial({Monomial(((x, 1),)): 1.0,
                 Monomial(((y, 3),)): 1.0})

print(str(p2.leading_monomial('grevlex')))   # x^2
print(str(p2.leading_monomial('lex')))       # x^2
print(str(p3.leading_monomial('lex')))       # x
print(str(p3.leading_monomial('grevlex')))   # y^3
print(p2.leading_coefficient('lex'))         # 1.0
```

---

### 3. Polynomial Division

**Definition (Division Identity)**: Dividing $f$ by a single divisor $g$
produces $(q, r)$ with

$$
f \;=\; q\cdot g \;+\; r,
\qquad
r \text{ has no term divisible by } \operatorname{LM}(g).
$$

Undivisible leading terms accumulate into $r$ while the working polynomial
shrinks, so the loop always terminates:

```Python
from src.utils.symbolic import parse_polynomial_string

dividend = parse_polynomial_string("x^2 - 4y", [x, y])
quot, rem = dividend.divide_by(Polynomial({Monomial(((x, 1),)): 1.0}))

print(str(quot))         # x
print(str(rem))          # -4.0000*y

recon = quot * Polynomial({Monomial(((x, 1),)): 1.0}) + rem
print(recon.terms == dividend.terms)    # True
```

Exact division leaves a zero remainder:

```Python
qq, rr = parse_polynomial_string("x^2 - 1", [x]).divide_by(
    parse_polynomial_string("x - 1", [x]))
print(rr.is_zero(), str(qq))     # True x + 1.0000
```

---

### 4. GCD and LCM (Univariate)

`polynomial_gcd` runs the Euclidean algorithm and returns a monic result.
It is deliberately restricted to univariate inputs: for multivariate
polynomials, leading terms can be mutually indivisible (`xy` vs `x^2`) and
the remainder sequence would never terminate — multivariate elimination
belongs to the Groebner machinery of `alggeom`.

```Python
from src.utils.symbolic import polynomial_gcd, polynomial_lcm

f1 = parse_polynomial_string("x^2 - 1", [x])
g1 = parse_polynomial_string("x - 1", [x])
h1 = parse_polynomial_string("x + 1", [x])

print(str(polynomial_gcd(f1, g1)))   # x + -1.0000
print(str(polynomial_gcd(f1, h1)))   # x + 1.0000
print(str(polynomial_lcm(f1, h1)))   # x^2 + -1.0000
```

The guard fails loudly rather than hanging:

```Python
try:
    polynomial_gcd(Polynomial({Monomial(((x, 1),)): 1.0}),
                   Polynomial({Monomial(((y, 1),)): 1.0}))
except NotImplementedError as e:
    print("refused:", str(e)[:44])
# refused: polynomial_gcd supports univariate po
```

---

### 5. Parsing Polynomial Strings

The parser handles implicit multiplication (`3x^2`), parentheses with
exponents (`(x+1)^2`), and negative terms:

```Python
pp = parse_polynomial_string("(x+1)^2 - y", [x, y])
print(pp.evaluate({x: 2.0, y: 3.0}))     # 6.0    (3^2 - 3)

pp2 = parse_polynomial_string("3x^2 + 2x*y - y", [x, y])
print(pp2.evaluate({x: 1.0, y: 2.0}))    # 5.0    (3 + 4 - 2)
```

---

### 6. Coefficient Arrays

Dense coefficient tensors convert to polynomials and back losslessly. Entry
$(i_1,\dots,i_n)$ of the array is the coefficient of
$x_1^{i_1}\cdots x_n^{i_n}$:

```Python
import numpy as np
from src.utils.symbolic import (
    create_polynomial_from_coefficients, polynomial_to_array)

coeffs = np.zeros((3, 3))
coeffs[2, 0] = 1.0           # x^2
coeffs[0, 1] = -4.0          # y

pc = create_polynomial_from_coefficients(coeffs, [x, y])
print(str(pc))               # x^2 + -4.0000*y
print(np.allclose(polynomial_to_array(pc, [x, y], max_degree=2), coeffs))
# True
```

---

### 7. Systems of Polynomials and Their Jacobians

A `SymbolicPolynomialSystem` evaluates all equations at once and builds the
Jacobian by symbolic differentiation:

$$
F(v) \;=\; \big(f_1(v),\dots,f_r(v)\big)^{\top},
\qquad
J_{ij} \;=\; \frac{\partial f_i}{\partial v_j}.
$$

```Python
from src.utils.symbolic import SymbolicPolynomialSystem

sys_ = SymbolicPolynomialSystem(
    [x, y],
    [parse_polynomial_string("x^2 + y^2 - 1", [x, y]),
     parse_polynomial_string("x - y", [x, y])])

print(sys_.evaluate(np.array([1.0, 2.0])))   # [ 4. -1.]
print(sys_.jacobian(np.array([1.0, 2.0])))
# [[ 2.  4.]
#  [ 1. -1.]]
```

---

### 8. Numerical Gradient and Jacobian Helpers

For black-box callables, central differences give second-order accurate
derivatives:

$$
\frac{\partial f}{\partial x_i}(v) \;\approx\;
\frac{f(v + \varepsilon e_i) - f(v - \varepsilon e_i)}{2\varepsilon}.
$$

```Python
from src.utils.symbolic import symbolic_gradient, symbolic_jacobian

f = lambda v: v[0] ** 2 + 3 * v[1]
print(np.allclose(symbolic_gradient(f, np.array([2.0, 1.0])), [4.0, 3.0],
                  atol=1e-6))       # True

F = lambda v: np.array([v[0] ** 2, v[0] * v[1]])
print(np.allclose(symbolic_jacobian(F, np.array([2.0, 3.0])),
                  [[4.0, 0.0], [3.0, 2.0]], atol=1e-5))   # True
```

---

### 9. Visualizing Varieties

`VarietyVisualizer` routes evaluation through each polynomial's own variable
list, so `alggeom.Polynomial` objects plot correctly. The 2D plot draws the
zero contour; the point helper reports membership. Headless sessions should
select the `Agg` backend first:

```Python
import matplotlib
matplotlib.use('Agg')
import numpy as np
from src.utils.visualization import VarietyVisualizer
from src.utils.symbolic import parse_polynomial_string as uparse

circle = uparse("x^2 + y^2 - 1", [x, y])
viz = VarietyVisualizer()

fig = viz.plot_variety_2d(circle, (-2, 2), (-2, 2))
pts = fig.axes[0].collections[0].get_paths()[0].vertices
radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
print(bool(np.abs(radii - 1).max() < 0.05))    # True  (contour IS the circle)

figp = viz.plot_point_on_variety(circle, np.array([1.0, 0.0]), (-2, 2), (-2, 2))
print(figp.axes[0].get_legend().get_texts()[1].get_text())
# Point (on variety: True)
```

Off-variety points are flagged honestly, and the 3D view plots the graph
surface $z = f(x,y)$ together with its zero set:

```Python
figo = viz.plot_point_on_variety(circle, np.array([2.0, 2.0]), (-3, 3), (-3, 3))
print(figo.axes[0].get_legend().get_texts()[1].get_text())
# Point (on variety: False)

fig3 = viz.plot_variety_3d(circle, (-1.5, 1.5), (-1.5, 1.5))
print(fig3.axes[0].get_title())
# Surface z = f(x, y); red curve: f = 0
```

`PolynomialVisualizer` offers 1D curves and 2D heatmaps with the same
evaluation routing:

```Python
from src.utils.visualization import PolynomialVisualizer

pviz = PolynomialVisualizer()
fp = pviz.plot_polynomial_1d(uparse("x^2 - 1", [x]), (-2, 2))
yd = fp.axes[0].get_lines()[0].get_ydata()
print(round(float(yd[0]), 4), abs(yd[len(yd)//2] + 1.0) < 0.01)
# 3.0 True

fh = pviz.plot_polynomial_2d_heatmap(circle, (-2, 2), (-2, 2), resolution=50)
print(abs(fh.axes[0].images[0].get_array().data[25, 25] + 1.0) < 0.2)
# True   (center of circle grid ~ f(0,0) = -1)
```

---

### 10. Training Curves and Decision Boundaries

`TrainingVisualizer` accepts both attribute-style histories and plain
dictionaries like the one returned by `OrthoPolyNetwork.fit`:

```Python
from src.utils.visualization import (
    TrainingVisualizer, plot_decision_boundary,
    plot_groebner_basis_convergence)

tviz = TrainingVisualizer()
ft = tviz.plot_training_history({'loss': [1.0, 0.5, 0.25]})
print(ft.axes[0].get_lines()[0].get_ydata().tolist())
# [1.0, 0.5, 0.25]

flr = tviz.plot_learning_rate_schedule({'learning_rates': [0.1, 0.01]})
print(len(flr.axes[0].get_lines()))      # 1
```

Decision boundaries accept any model with `forward`; torch tensors are
converted automatically. Gröbner computation statistics render as two panels:

```Python
class ThresholdModel:
    def forward(self, X):
        return (X[:, :1] > 0).astype(float)

Xc = np.vstack([np.random.randn(20, 2) + [2, 0],
                np.random.randn(20, 2) - [2, 0]])
yc = np.array([0] * 20 + [1] * 20)

fb = plot_decision_boundary(ThresholdModel(), Xc, yc)
print(len(fb.axes[0].collections) >= 2)      # True

fg = plot_groebner_basis_convergence([2, 3, 3], [0.1, 0.2, 0.15])
print(len(fg.axes))                          # 2
```

**Problem**: Why does `polynomial_gcd` refuse `gcd(x, y)` instead of simply
returning 1?

**Solution**: The Euclidean loop needs leading terms that divide each other
to make progress. With incomparable leading monomials ($x$ vs $y$ under any
global order neither divides the other), division just swaps the arguments
forever: $f, g \to g, f \to f, g \to \cdots$. Detecting this requires either
a Groebner basis (which normalizes the ideal so remainders terminate) or an
explicit domain restriction — the module chooses the honest restriction and
points to `src.alggeom.groebnerbasis`.

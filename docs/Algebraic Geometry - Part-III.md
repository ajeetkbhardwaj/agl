# Algebraic Geometry - Part-III

## Table of Contents

### 1. Affine Varieties

**Definition (Affine Variety)**: Given an ideal $I \subseteq \mathbb{K}[x_1,\dots,x_n]$, the **affine variety** defined by $I$ is the set of common zeros

$$
V(I) = \{\, x \in \mathbb{K}^n : f(x) = 0 \text{ for all } f \in I \,\}
$$

**The ideal-variety correspondence** reverses inclusions:

$$
I \subseteq J \quad \Longleftrightarrow \quad V(J) \subseteq V(I)
$$

and links algebra to geometry:

| Algebra | Geometry |
|---------|----------|
| $I + J$ | $V(I) \cap V(J)$ |
| $IJ$, $\sqrt{I \cap J}$ | $V(I) \cup V(J)$ |
| $\sqrt{I}$ | Zariski closure of $V(I)$ |
| prime ideal | irreducible variety |
| $\dim \mathbb{K}[x]/I$ | dimension of $V(I)$ |

```Python
from src.alggeom.polynomial import Variable, Polynomial
from src.alggeom.algvariety import (
    Point, AlgebraicVariety, Hypersurface, AffineSpace,
    solve_polynomial_system, intersection, union,
)

x, y = Variable("x"), Variable("y")
v = lambda name: Polynomial.variable(Variable(name))
one = Polynomial.constant

V = AlgebraicVariety([v('x') - one(1), v('y') - one(2)])
print(V)                # AlgebraicVariety(ideal with 2 generators)
print(V.groebner_basis) # computed automatically on construction
```

### 2. Points and Membership

A `Point` carries coordinates as a dictionary. Membership testing evaluates the Groebner basis of the variety at the point - by the definition of a Groebner basis this is equivalent to checking every polynomial of the ideal.

```Python
print(V.contains(Point({x: 1.0, y: 2.0})))   # True
print(V.contains(Point({x: 1.0, y: 3.0})))   # False
```

Coordinate keys may be `Variable` objects or plain name strings.

### 3. Dimension

**Definition (Dimension)**: The dimension of $V(I)$ is the maximal number of algebraically independent coordinates on it:

$$
\dim V(I) = \dim(\mathbb{K}[x]/I)
$$

It is computed from the leading terms of the Groebner basis (Part II, Section 9): find the largest set of variables untouched by any leading monomial.

```Python
point_set   = AlgebraicVariety([v('x') - one(1), v('y') - one(2)])  # two points
curve       = Hypersurface(v('y')**2 - v('x')**3)                   # plane curve
space_curve = AlgebraicVariety([v('z') - v('x')**2, v('y') - v('x')**2])
ambient     = AffineSpace(3, ['x', 'y', 'z'])

print(point_set.dimension())    # 0
print(curve.dimension())        # 1
print(space_curve.dimension())  # 1
print(ambient.dimension())      # 3
```

Intuition: a variety is **0-dimensional** when it is a finite set of points; each extra parameter adds one dimension.

### 4. Solving Polynomial Systems

Solving combines Part II's two tools - **lex elimination** and **saturation-free back-substitution**:

1. Compute a lex Groebner basis
2. Extract the univariate polynomial in the last variable and find its roots numerically (`np.roots`)
3. Substitute each root into the remaining equations and recurse

```Python
# line x = y meets circle x^2 + y^2 = 2
solutions = solve_polynomial_system(
    [v('x')**2 + v('y')**2 - one(2), v('x') - v('y')],
    [x, y]
)
print([(round(float(s.coordinates['x']), 6),
        round(float(s.coordinates['y']), 6)) for s in solutions])
# [(1.0, 1.0), (-1.0, -1.0)]
```

Real solutions are returned as `Point` objects; positive-dimensional components (curves inside the solution set) are reported as empty rather than enumerated.

### 5. Singular Points and the Jacobian Criterion

**Definition (Singular point)**: A point $p \in V(I)$ is **singular** if the Jacobian matrix drops rank there:

$$
\text{rank}\, J_F(p) < \operatorname{codim} V(I), \qquad (J_F)_{ij} = \frac{\partial f_i}{\partial x_j}
$$

For a hypersurface $f = 0$ this means solving the system

$$
f = 0, \quad \frac{\partial f}{\partial x_1} = 0, \quad \dots, \quad \frac{\partial f}{\partial x_n} = 0
$$

```Python
cusp = Hypersurface(v('y')**2 - v('x')**3)

singulars = cusp.singular_points()
print([(float(s.coordinates['x']), float(s.coordinates['y'])) for s in singulars])
# [(0.0, 0.0)]     <- the cusp y^2 = x^3 has exactly one singular point
```

The cusp is smooth everywhere except at the origin, where both partial derivatives vanish simultaneously.

### 6. Hypersurfaces: Smoothness and Genus

A **hypersurface** is a variety defined by a single equation. It is **smooth** when its singular locus is empty, i.e. the gradient never vanishes on it.

For a smooth plane curve of degree $d$, the **genus formula** counts the number of handles topologically:

$$
g = \frac{(d-1)(d-2)}{2}
$$

```Python
circle = Hypersurface(v('x')**2 + v('y')**2 - one(1))
cubic  = Hypersurface(v('y')**2 - v('x')**3)

print(circle.is_smooth())   # True   <- gradient never zero on the circle
print(cubic.genus())        # 1      <- d = 3: elliptic curve
```

### 7. Set Operations on Varieties

By the correspondence table of Section 1:

```Python
V1 = AlgebraicVariety([v('x') - one(1)])       # vertical line x = 1
V2 = AlgebraicVariety([v('y') - one(2)])       # horizontal line y = 2

VI = intersection(V1, V2)                      # V(I + J): the point (1, 2)
print([str(g) for g in VI.ideal])
# ['x - 1.0000', 'y - 2.0000']

VU = union(V1, V2)                             # V(IJ): the cross
print([str(g) for g in VU.ideal])
# ['x * y - 2.0000*x - y + 2.0000']             <- (x-1)(y-2)
```

Note how union multiplies the defining polynomials: $(x-1)(y-2) = 0$ holds iff $x = 1$ **or** $y = 2$. The Zariski closure of an affine variety is itself, so `closure` returns its argument unchanged.

### 8. Varieties as Constraints in Optimization

Polynomial constraints define feasible sets, and neural networks can be trained against them. The variety API exposes the three ingredients required by constrained optimization:

- **Constraint violations** $f(x)$ - how far a point is from satisfying each equation
- **Constraint Jacobian** $J_F(x)$ - first-order sensitivity of violations
- **Lagrangian gradient** - for $L(x,\lambda) = f(x) + \lambda^\top c(x)$, the term $J_c(x)^\top \lambda$

```Python
import numpy as np

VI = intersection(V1, V2)          # constraints x - 1 = 0, y - 2 = 0
p = np.array([1.0, 2.0])           # a feasible point

print(VI.constraint_violations(p))         # [0. 0.]
print(VI.constraint_jacobian(p))           # [[1. 0.]
                                           #  [0. 1.]]

lam = np.array([0.5, -1.0])
print(VI.constraint_lagrangian_gradient(p, lam))   # [ 0.5 -1. ]
```

These utilities connect the symbolic layer directly to the training loop of `OrthoPolyNNs`: polynomial varieties act as exact, differentiable constraint manifolds.

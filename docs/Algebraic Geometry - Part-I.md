# Algebraic Geometry - Part-I

## Table of Contents

### 1. Symbolic Representation

In algebraic geometry variable is a formal symbolic representation of indeterminate element of a polynomial ring$\mathbb{F}[x_1, \dots, x_n]$ over a field $\mathbb{F}$. It has following key properties

- Variables are **formal -** has no numeric value until evaluation
- Variables are **distinct -** $x \neq y$ even if they evaluate to same number
- Variables support **indexing -** $x_1, x_2, \dots$ for multivariate systems

```Python
from src.utils.symbolic import Variable

theta = Variable(name="theta", index=None)
omega = Variable(name="omega", index=None)
print(theta.name)
print(theta.index)

## Variable Comparision

def VarComparision(theta, omega):
    if theta.name != omega.name:
        return False
    if theta.index != omega.index:
        return False
    return True

print(VarComparision(theta, omega))
```

### 2. Monomial Representation as Product of Variable's Power

**Definition (Monomial)**: A monomial in variables $x_1, \dots, x_n$ is a product

$$
m = x_1^{e_1} x_2^{e_2} \cdots x_n^{e_n} = \prod_{i=1}^n x_i^{e_i}
$$

where $e_i \in \mathbb{Z}_{\geq 0}$ are non-negative integer exponents.

**Key Operations**:

- **Multiplication**: $x^a \cdot x^b = x^{a+b}$
- **Power**: $(x^a)^b = x^{ab}$
- **Degree**: $\deg(x_1^{e_1}\cdots x_n^{e_n}) = \sum e_i$
- **Division**: $x^a / x^b = x^{a-b}$ if $a \geq b$ (component-wise)


### 3. Polynomial Representation as a linear combination of monomials

**Definition (Polynomial)**: A polynomial in variables $x_1, \dots, x_n$ is a finite linear combination of monomials:

$$
p = \sum_{i=1}^m c_i \cdot m_i = \sum_{\alpha \in \mathbb{Z}_{\geq 0}^n} c_\alpha x^\alpha
$$

where $c_i \in \mathbb{K}$ are coefficients and $m_i$ are distinct monomials.

**Key Operations**:

- **Addition**: Combine like terms (same monomial)
- **Multiplication**: Distribute and combine: $(a+b)(c+d) = ac+ad+bc+bd$
- **Differentiation**: $\frac{\partial}{\partial x_j}(c \cdot x^\alpha) = c \cdot \alpha_j \cdot x^{\alpha - e_j}$
- **Evaluation**: Substitute numeric values for variables

**Problem**: Given $p = 3x^2y + 2xy^2 - 5$ and $q = x^2y - xy^2 + 1$, compute:

1. $p + q$
2. $p \cdot q$ (first two terms only)
3. $\frac{\partial p}{\partial x}$
4. $p$ evaluated at $(x,y) = (2, -1)$

### 4. Polynomial Arithmetic

A polynomial is stored internally as a dictionary mapping monomials to coefficients:

$$
p \quad \longleftrightarrow \quad \{\, m_1 : c_1,\ \dots,\ m_k : c_k \,\}
$$

Coefficients whose magnitude falls below `TOLERANCE = 1e-12` are treated as zero and dropped automatically, so like terms always combine cleanly.

```Python
from src.alggeom.polynomial import Variable, Monomial, Polynomial, parse_polynomial_string

x, y = Variable("x"), Variable("y")

# Build p = 3x^2*y + 2x*y^2 - 5 directly ...
p = Polynomial({
    Monomial(((x, 2), (y, 1))): 3.0,
    Monomial(((x, 1), (y, 2))): 2.0,
    Monomial(()): -5.0
})
print(p)            # 3.0000*x^2 * y + 2.0000*x * y^2 - 5.0000
print(p.degree())   # 3

# ... or parse it from a string
q = parse_polynomial_string('x^2*y - x*y^2 + 1', [x, y])
print(q)            # x^2 * y - x * y^2 + 1.0000
```

**Key Operations**:

- **Addition / Subtraction** - combine like terms, cancel near-zero results
- **Multiplication** - distribute over all term pairs, then collect
- **Scalar multiplication** - `2 * p` and `p * 2` both work (`__rmul__`)
- **Power** - binary exponentiation, so `p ** 8` costs 3 multiplications, not 7

```Python
print(p + q)    # 4.0000*x^2 * y + x * y^2 - 4.0000
print(p - q)    # 2.0000*x^2 * y + 3.0000*x * y^2 - 6.0000
print(p * q)    # degree 6 polynomial, leading terms 3x^4*y^2 - x^3*y^3 + ...
print(2 * q)    # 2.0000*x^2 * y - 2.0000*x * y^2 + 2.0000
```

**Solution to the Problem of Section 3**:

```Python
# 1. p + q
print(p + q)                        # 4.0000*x^2 * y + x * y^2 - 4.0000

# 2. p * q, first two terms (highest total degree)
prod = p * q
# 3.0000*x^4*y^2 + (-1.0)*x^3*y^3 + ...

# 3. partial derivative
print(p.differentiate(x))           # 6.0000*x * y + 2.0000*y^2

# 4. evaluation
print(p.evaluate({x: 2.0, y: -1.0}))   # -13.0
```

### 5. Calculus on Polynomials

Differentiation follows the power rule term-wise:

$$
\frac{\partial}{\partial x_j}\left(c \cdot x_1^{e_1} \cdots x_j^{e_j} \cdots x_n^{e_n}\right) = c \cdot e_j \cdot x_1^{e_1} \cdots x_j^{e_j - 1} \cdots x_n^{e_n}
$$

Evaluation substitutes numeric values for every variable; missing variables are treated as $0$.

```Python
print(p.differentiate(x))              # 6.0000*x * y + 2.0000*y^2
print(p.differentiate(y))              # 3.0000*x^2 + 4.0000*x * y
print(p.evaluate({x: 2.0, y: -1.0}))   # -13.0
```

### 6. Monomial Orderings and Leading Terms

A **monomial ordering** is a total order on $\mathbb{Z}_{\geq 0}^n$ compatible with multiplication. Three classical orders are supported:

- **lex** - compare exponent vectors left to right: $x^a >_{lex} x^b$ iff the leftmost nonzero entry of $a - b$ is positive
- **grlex** - total degree first, then lex as tie-breaker
- **grevlex** - total degree first, then *reverse* lex: the smaller exponent vector in reversed order wins

The **leading monomial** $\text{LM}(f)$, **leading coefficient** $\text{LC}(f)$ and **leading term** $\text{LT}(f) = \text{LC}(f)\cdot\text{LM}(f)$ depend on this choice.

```Python
from src.alggeom.polynomial import parse_polynomial_string

x, y, z = Variable("x"), Variable("y"), Variable("z")
g = parse_polynomial_string('x^2*z + y^3', [x, y, z])

for order in ('lex', 'grlex', 'grevlex'):
    lm, lc = g.leading_term(order)
    print(order, "->", lm, lc)

# lex     -> x^2 * z 1.0      (x dominates)
# grlex   -> x^2 * z 1.0      (both degree 3, lex tie-break)
# grevlex -> y^3 1.0          (smaller last exponents win)
```

Leading terms drive everything in Parts II and III: division, Groebner bases, elimination and dimension.

### 7. Multivariate Division

**Theorem (Quotient-Remainder)**: Fix a monomial order. For polynomials $f, g$ with $g \neq 0$ there exist unique $r$ with $\text{LM}(r)$ not divisible by $\text{LM}(g)$ such that

$$
f = q \cdot g + r
$$

`divide_by` returns the pair `(quotient, remainder)`:

```Python
a = parse_polynomial_string('x^3*y^2 + x^4', [x, y])
b = parse_polynomial_string('x^2*y', [x, y])

quo, rem = a.divide_by(b)
print(quo)   # x * y
print(rem)   # x^4        <- LM(x^4) is NOT divisible by LM(x^2*y)
```

The remainder is fully reduced: no term of it can be cancelled by the divisor any more.

### 8. Parsing Polynomials from Strings

`parse_polynomial_string` is a small recursive-descent parser for the grammar

```
expr   := term (('+' | '-') term)*
term   := factor ('*' factor)*
factor := ('-')? atom ('^' integer)?
atom   := number | variable | '(' expr ')'
```

so parentheses, powers of sums and distribution all behave correctly. Implicit multiplication (`2x`, `3(x+1)`) is inserted automatically.

```Python
print(parse_polynomial_string('(x+y)^2', [x, y]))
# x^2 + 2.0000*x * y + y^2

print(parse_polynomial_string('3*(x+1)^2', [x, y]))
# 3.0000*x^2 + 6.0000*x + 3.0000

print(parse_polynomial_string('(x+1)*(y-1)', [x, y]))
# x * y - x + y - 1.0000
```

Unknown variable names raise a `ValueError` listing the known variables.

### 9. Polynomial Systems and the Jacobian

A `SymbolicPolynomialSystem` bundles variables with equations and provides matrix calculus:

$$
J_F(x)_{ij} = \frac{\partial f_i}{\partial x_j}(x)
$$

```Python
import numpy as np
from src.alggeom.polynomial import SymbolicPolynomialSystem

system = SymbolicPolynomialSystem([x, y], [p, q])
print(system)
# Variables: ['x', 'y']
# Polynomials:
#   f_0 = 3.0000*x^2 * y + 2.0000*x * y^2 - 5.0000
#   f_1 = x^2 * y - x * y^2 + 1.0000

print(system.evaluate(np.array([1.0, 2.0])))    # [ 9. -1.]
print(system.jacobian(np.array([1.0, 2.0])))
# [[20. 11.]
#  [ 0. -3.]]
```

The Jacobian is used in Part III for detecting singular points of varieties.


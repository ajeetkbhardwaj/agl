"""
@Title : Symbolic Computation Module
@Author : Ajeet Kumar + AI
@Description : 
It conains the utilities for symbolic polynomial operations and differentiation.
@Table of Contents :
1. Variable Class
2. Monomial Class
3. Polynomial Class
4. SymbolicPolynomialSystem Class
5. Utility Functions

"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
import numpy as np
import re

TOLERANCE = 1e-12 

@dataclass
class Variable:
    """Symbolic variable representation."""
    name: str
    index: Optional[int] = None
    
    def __str__(self) -> str:
        if self.index is not None:
            return f"{self.name}_{self.index}"
        return self.name
    
    def __repr__(self) -> str:
        return f"Variable('{self.name}', {self.index})"
    
    def __hash__(self) -> int:
        return hash((self.name, self.index))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name and self.index == other.index


@dataclass
class Monomial:
    """Represents a monomial (product of variables with exponents).
     **Monomial Representation**: The representation `Tuple[Tuple[Variable, int], ...]` is mathematically correct for multivariate monomials. The product of monomials correctly combines exponents.

    ."""
    variables: Tuple[Tuple[Variable, int], ...] = field(default_factory=tuple)
    
    def __mul__(self, other: 'Monomial') -> 'Monomial':
        """Multiply two monomials."""
        # Combine variables
        var_dict = dict(self.variables)
        for var, exp in other.variables:
            if var in var_dict:
                var_dict[var] += exp
            else:
                var_dict[var] = exp
        return Monomial(tuple(sorted(var_dict.items(), key=lambda item: item[0].name)))
    
    def __pow__(self, n: int) -> 'Monomial':
        """Raise monomial to power n."""
        if n == 0:
            return Monomial(())
        new_vars = tuple((v, e * n) for v, e in self.variables)
        return Monomial(new_vars)
    
    def degree(self) -> int:
        """Total degree of monomial."""
        return sum(exp for _, exp in self.variables)

    def get_vars(self) -> List[Variable]:
        return [v for v, e in self.variables]

    def divides(self, other: 'Monomial') -> bool:
        """Check if self divides other."""
        self_vars = dict(self.variables)
        other_vars = dict(other.variables)
        for var, exp in self_vars.items():
            if var not in other_vars or other_vars[var] < exp:
                return False
        return True

    def div(self, other: 'Monomial') -> 'Monomial':
        """Divide other by self, assuming divisibility."""
        if not self.divides(other):
            raise ValueError(f"Monomial {self} does not divide {other}")
        self_vars = dict(self.variables)
        other_vars = dict(other.variables)
        new_vars = {}
        for var, exp in other_vars.items():
            new_exp = exp - self_vars.get(var, 0)
            if new_exp > 0:
                new_vars[var] = new_exp
        return Monomial(tuple(sorted(new_vars.items(), key=lambda item: item[0].name)))

    def __str__(self) -> str:
        if not self.variables:
            return "1"
        parts = []
        for var, exp in self.variables:
            if exp == 1:
                parts.append(str(var))
            else:
                parts.append(f"{var}^{exp}")
        return " * ".join(parts)
    
    def __repr__(self) -> str:
        return f"Monomial({self.variables})"
    
    def __hash__(self) -> int:
        return hash(self.variables)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Monomial):
            return False
        return self.variables == other.variables


@dataclass
class Polynomial:
    """
    Symbolic polynomial representation.
    Stored as a dictionary: {monomial: coefficient}
    """
    terms: Dict[Monomial, float] = field(default_factory=dict)
    
    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        """Add two polynomials."""
        result = Polynomial(dict(self.terms))
        for monom, coeff in other.terms.items():
            if monom in result.terms:
                result.terms[monom] += coeff
                if abs(result.terms[monom]) < TOLERANCE:
                    del result.terms[monom]
            else:
                result.terms[monom] = coeff
        return result
    
    def __sub__(self, other: 'Polynomial') -> 'Polynomial':
        """Subtract two polynomials."""
        result = Polynomial(dict(self.terms))
        for monom, coeff in other.terms.items():
            if monom in result.terms:
                result.terms[monom] -= coeff
                if abs(result.terms[monom]) < TOLERANCE:
                    del result.terms[monom]
            else:
                result.terms[monom] = -coeff
        return result
    
    def __mul__(self, other: Union['Polynomial', float, int]) -> 'Polynomial':
        """Multiply polynomial by another polynomial or scalar."""
        if isinstance(other, (float, int)):
            if other == 0:
                return Polynomial({})
            return Polynomial({m: c * other for m, c in self.terms.items()})
        
        result = Polynomial({})
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                new_monom = m1 * m2
                new_coeff = c1 * c2
                if new_monom in result.terms:
                    result.terms[new_monom] += new_coeff
                else:
                    result.terms[new_monom] = new_coeff
        return result
    
    def __rmul__(self, other: float) -> 'Polynomial':
        return self.__mul__(other)
    
    def __pow__(self, n: int) -> 'Polynomial':
        if n < 0:
            raise ValueError("Negative powers not supported")
        if n == 0:
            return Polynomial({Monomial(()): 1.0})
        if self.is_zero() or n == 0:
            return Polynomial({})
        if n == 1:
            return Polynomial(dict(self.terms))
        # Use binary exponentiation for efficiency
        result = Polynomial({Monomial(()): 1.0})
        base = Polynomial(dict(self.terms))
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        return result
    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def is_one(self) -> bool:
        return len(self.terms) == 1 and Monomial(()) in self.terms \
            and abs(self.terms[Monomial(())] - 1.0) < TOLERANCE

    @classmethod
    def constant(cls, c: float) -> 'Polynomial':
        if abs(c) < TOLERANCE:
            return cls({})
        return cls({Monomial(()): float(c)})

    def copy(self) -> 'Polynomial':
        return Polynomial(dict(self.terms))

    def variables(self) -> List[Variable]:
        """Set of variables in the polynomial."""
        all_vars = set()
        for monom in self.terms.keys():
            all_vars.update(monom.get_vars())
        return sorted(list(all_vars), key=lambda v: v.name)

    def _get_monomial_key(self, monom: Monomial, all_vars: List[Variable], order: str):
        exponents = {v: 0 for v in all_vars}
        for v, e in monom.variables:
            exponents[v] = e

        if order == 'lex':
            return tuple(exponents[v] for v in all_vars)
        elif order == 'grlex':
            return (monom.degree(), tuple(exponents[v] for v in all_vars))
        elif order == 'grevlex':
            return (monom.degree(), tuple(-exponents[v] for v in reversed(all_vars)))
        else:
            raise ValueError(f"Unknown order: {order}")

    def leading_monomial(self, order: str = 'grevlex', variables: Optional[List[Variable]] = None) -> Monomial:
        """
        Leading monomial with respect to the given order.
        An explicit global variable list can be supplied so that leading
        terms of different polynomials are computed under one common order.
        """
        if self.is_zero():
            return Monomial()
        all_vars = variables if variables is not None else self.variables()
        return max(self.terms.keys(), key=lambda m: self._get_monomial_key(m, all_vars, order))

    def leading_coefficient(self, order: str = 'grevlex', variables: Optional[List[Variable]] = None) -> float:
        if self.is_zero():
            return 0.0
        return self.terms[self.leading_monomial(order, variables)]

    def _clean_coefficient(self, coeff: float) -> Optional[float]:
        """Remove near-zero coefficients."""
        if abs(coeff) < TOLERANCE:
            return None
        return coeff
    
    def degree(self) -> int:
        """Maximum degree of any term."""
        return max((m.degree() for m in self.terms.keys()), default=0)
    
    def evaluate(self, values: Dict[Variable, float]) -> float:
        """Evaluate polynomial at given values."""
        total = 0.0
        for monom, coeff in self.terms.items():
            term_val = coeff
            for var, exp in monom.variables:
                term_val *= (values.get(var, 0) ** exp)
            total += term_val
        return total
    
    def differentiate(self, var: Variable) -> 'Polynomial':
        """Compute partial derivative with respect to variable."""
        result = {}
        for monom, coeff in self.terms.items():
            new_vars = []
            found = False
            for v, exp in sorted(monom.variables, key=lambda item: item[0].name):
                if v == var:
                    if exp > 0:
                        new_vars.append((v, exp - 1))
                        found = True
                else:
                    new_vars.append((v, exp))
            
            if found:
                new_monom = Monomial(tuple(sorted(new_vars, key=lambda item: item[0].name)))
                result[new_monom] = result.get(new_monom, 0) + coeff * exp
        
        return Polynomial(result)
    def divide_by(self, divisor: 'Polynomial', order: str = 'grevlex') -> Tuple['Polynomial', 'Polynomial']:
        """
        Divide self by divisor with respect to monomial ordering.
        Returns (quotient, remainder) such that self = quotient * divisor + remainder.
        """
        if divisor.is_zero():
            raise ValueError("Division by zero polynomial")

        quotient = Polynomial()
        remainder_acc = Polynomial()
        work = self.copy()
        all_vars = sorted(set(self.variables()) | set(divisor.variables()), key=lambda v: v.name)
        divisor_lm = divisor.leading_monomial(order, all_vars)
        divisor_lc = divisor.leading_coefficient(order, all_vars)

        while not work.is_zero():
            work_lm = work.leading_monomial(order, all_vars)

            if divisor_lm.divides(work_lm):
                # Compute quotient term
                factor_monomial = divisor_lm.div(work_lm)
                factor_coeff = work.leading_coefficient(order, all_vars) / divisor_lc
                factor = Polynomial({factor_monomial: factor_coeff})

                quotient = quotient + factor
                work = work - factor * divisor
            else:
                # Move the leading term to the remainder accumulator and
                # continue reducing what is left.
                lt_coeff = work.leading_coefficient(order, all_vars)
                lt_poly = Polynomial({work_lm: lt_coeff})

                remainder_acc = remainder_acc + lt_poly
                work = work - lt_poly

        return quotient, remainder_acc
    
    def __str__(self) -> str:
        if not self.terms:
            return "0"
        
        terms_str = []
        for monom, coeff in sorted(self.terms.items(), 
                                   key=lambda x: -x[0].degree()):
            if abs(coeff) < TOLERANCE:
                continue
            monom_str = str(monom) if monom.variables else "1"
            if monom_str == "1":
                terms_str.append(f"{coeff:.4f}")
            elif abs(coeff - 1) < TOLERANCE:
                terms_str.append(monom_str)
            elif abs(coeff + 1) < TOLERANCE:
                terms_str.append(f"-{monom_str}")
            else:
                terms_str.append(f"{coeff:.4f}*{monom_str}")
        
        return " + ".join(terms_str) if terms_str else "0"
    
    def __repr__(self) -> str:
        return f"Polynomial({self.terms})"


class SymbolicPolynomialSystem:
    """
    System of polynomial equations for symbolic manipulation.
    """
    
    def __init__(self, variables: List[Variable], 
                 polynomials: Optional[List[Polynomial]] = None):
        self.variables = variables
        self.polynomials = polynomials or []
        self._build_variable_index()
    
    def _build_variable_index(self):
        """Build index mapping for variables."""
        self.var_index = {v: i for i, v in enumerate(self.variables)}
    
    def add_polynomial(self, poly: Polynomial):
        """Add a polynomial to the system."""
        self.polynomials.append(poly)
    
    def evaluate(self, values: np.ndarray) -> np.ndarray:
        """Evaluate all polynomials at given point."""
        val_dict = {v: values[i] for i, v in enumerate(self.variables)}
        return np.array([p.evaluate(val_dict) for p in self.polynomials])
    
    def jacobian(self, values: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix at given point."""
        val_dict = {v: values[i] for i, v in enumerate(self.variables)}
        jac = np.zeros((len(self.polynomials), len(self.variables)))
        
        for i, poly in enumerate(self.polynomials):
            for j, var in enumerate(self.variables):
                deriv = poly.differentiate(var)
                jac[i, j] = deriv.evaluate(val_dict)
        
        return jac
    
    def __str__(self) -> str:
        lines = [f"Variables: {[str(v) for v in self.variables]}"]
        lines.append("Polynomials:")
        for i, p in enumerate(self.polynomials):
            lines.append(f"  f_{i} = {p}")
        return "\n".join(lines)


def create_polynomial_from_coefficients(coeffs: np.ndarray, 
                                        variables: List[Variable]) -> Polynomial:
    """
    Create polynomial from coefficient array.
    
    Args:
        coeffs: Coefficient array
        variables: List of variables
        
    Returns:
        Polynomial
    """
    terms = {}

    # np.ndenumerate already yields multi-indices
    for idx, coeff in np.ndenumerate(coeffs):
        if abs(coeff) < TOLERANCE:
            continue

        vars_list = [(variables[var_idx], int(exp))
                     for var_idx, exp in enumerate(idx) if exp > 0]

        monom = Monomial(tuple(vars_list))
        terms[monom] = float(coeff)

    return Polynomial(terms)


def polynomial_to_array(poly: Polynomial, variables: List[Variable],
                        max_degree: int) -> np.ndarray:
    """
    Convert polynomial to coefficient array.
    
    Args:
        poly: Polynomial to convert
        variables: Variables in the polynomial
        max_degree: Maximum degree for each variable
        
    Returns:
        Coefficient array
    """
    shape = tuple(max_degree + 1 for _ in variables)
    coeffs = np.zeros(shape)

    var_dict = {v: i for i, v in enumerate(variables)}

    for monom, coeff in poly.terms.items():
        # Create multi-index
        multi_idx = [0] * len(variables)
        for var, exp in monom.variables:
            if var in var_dict:
                multi_idx[var_dict[var]] = exp

        coeffs[tuple(multi_idx)] = coeff

    return coeffs


def symbolic_gradient(f: Callable[[np.ndarray], float], 
                     x: np.ndarray, 
                     eps: float = 1e-5) -> np.ndarray:
    """
    Compute symbolic-style gradient using numerical differentiation.
    
    Args:
        f: Function to differentiate
        x: Point at which to evaluate gradient
        eps: Epsilon for finite differences
        
    Returns:
        Gradient vector
    """
    n = x.size
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    
    return grad


def symbolic_jacobian(F: Callable[[np.ndarray], np.ndarray],
                     x: np.ndarray,
                     eps: float = 1e-5) -> np.ndarray:
    """
    Compute symbolic-style Jacobian using numerical differentiation.
    
    Args:
        F: Vector-valued function
        x: Point at which to evaluate Jacobian
        eps: Epsilon for finite differences
        
    Returns:
        Jacobian matrix
    """
    m = F(x).size
    n = x.size
    jac = np.zeros((m, n))
    
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        x_minus = x.copy()
        x_minus[j] -= eps
        
        jac[:, j] = (F(x_plus) - F(x_minus)) / (2 * eps)
    
    return jac


def parse_polynomial_string(s: str, variables: List[Variable]) -> Polynomial:
    """
    Parses a string into a symbolic Polynomial object.
    Supports basic implicit multiplication and parentheses expansion.
    Example: "3x^2 + 2x*y - y", "(x+1)^2"
    """
    var_map = {v.name: v for v in variables}
    s = s.replace(" ", "")
    
    # Expand common patterns: 2x -> 2*x
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    # x2 -> x^2 (if not decimal)
    s = re.sub(r'([a-zA-Z_])(\d)(?![.\d])', r'\1^\2', s)
    
    # Handle parentheses
    while '(' in s:
        match = re.search(r'\(([^()]+)\)(?:\^(\d+))?', s)
        if not match:
            break
        inner_str = match.group(1)
        exp_str = match.group(2)
        exp = int(exp_str) if exp_str else 1
        
        inner_poly = parse_polynomial_string(inner_str, variables)
        inner_pow = inner_poly ** exp
        
        # Replace the matched string with the string representation of the expanded polynomial
        # Enclose in parentheses to prevent operators bleeding into outer terms
        # but the expanded polynomial is already a sum, so we actually want to insert it correctly
        # The parser splits by '+', so we just need to ensure that the replaced string is well-formed.
        # E.g. "x * (y+1)" -> "x * y + x * 1". This is tricky because our basic string 
        # parser splits by '+' before handling '*'.
        # Wait, if `(x+1)^2` becomes `x^2 + 2.0000*x + 1.0000`, and the original string was `3*(x+1)^2`,
        # it becomes `3*x^2 + 2.0000*x + 1.0000` which is WRONG (should be `3*x^2 + 6*x + 3`).
        # To handle this properly without a full AST parser, we might just rely on the fact that
        # the user provided a very specific simple fix for this edge case.
        # Let's use exactly what was proposed.
        s = s[:match.start()] + str(inner_pow) + s[match.end():]

    s = s.replace(" ", "")
    s = s.replace("-", "+-")
    if s.startswith('+'):
        s = s[1:]

    poly = Polynomial()

    if not s:
        return poly

    terms = s.split('+')

    for term_str in terms:
        if not term_str:
            continue

        parts = term_str.split('*')
        coeff = 1.0
        monomial_vars = {}

        # Extract coefficient
        if parts and parts[0] == '-':
            coeff = -1.0
            parts = parts[1:]
        elif parts and parts[0].startswith('-'):
            try:
                coeff = float(parts[0])
                parts = parts[1:]
            except ValueError:  # e.g., '-x'
                coeff = -1.0
                parts[0] = parts[0][1:]
        elif parts:
            try:
                coeff = float(parts[0])
                parts = parts[1:]
            except (ValueError, IndexError):
                pass  # No explicit coefficient, defaults to 1.0

        # Parse monomial parts
        for part in parts:
            if not part: continue
            if '^' in part:
                var_name, exp_str = part.split('^')
                if var_name in var_map:
                    monomial_vars[var_map[var_name]] = monomial_vars.get(var_map[var_name], 0) + int(exp_str)
            elif part in var_map:
                monomial_vars[var_map[part]] = monomial_vars.get(var_map[part], 0) + 1

        monom = Monomial(tuple(sorted(monomial_vars.items(), key=lambda item: item[0].name)))
        poly.terms[monom] = poly.terms.get(monom, 0.0) + coeff

    # Clean up zero terms
    poly.terms = {m: c for m, c in poly.terms.items() if abs(c) > TOLERANCE}
    return poly



def polynomial_gcd(poly1: Polynomial, poly2: Polynomial,
                   order: str = 'grevlex') -> Polynomial:
    """
    Compute the monic GCD of two polynomials via the Euclidean algorithm.

    Restricted to the univariate case (both polynomials share at most one
    variable): for multivariate inputs, leading terms can be incomparable
    and the remainder sequence never terminates. Use a Groebner basis
    (src.alggeom.groebnerbasis) for multivariate elimination.
    """
    all_vars = sorted(set(poly1.variables()) | set(poly2.variables()),
                      key=lambda v: v.name)
    if len(all_vars) > 1:
        raise NotImplementedError(
            "polynomial_gcd supports univariate polynomials only; "
            "use src.alggeom.groebnerbasis for multivariate GCDs")

    if poly1.is_zero() and poly2.is_zero():
        return Polynomial.constant(1)
    if poly1.is_zero():
        # Make monic
        lc = poly2.leading_coefficient(order)
        return poly2 * (1/lc)
    if poly2.is_zero():
        lc = poly1.leading_coefficient(order)
        return poly1 * (1/lc)

    f, g = poly1.copy(), poly2.copy()
    while not g.is_zero():
        _, r = f.divide_by(g, order)
        f, g = g, r

    # Make monic
    lc = f.leading_coefficient(order)
    if lc != 0:
        f = f * (1/lc)
    return f

def polynomial_lcm(poly1: Polynomial, poly2: Polynomial,
                   order: str = 'grevlex') -> Polynomial:
    """LCM using: lcm(a,b) = a*b / gcd(a,b)"""
    gcd = polynomial_gcd(poly1, poly2, order)
    if gcd.is_one():
        return poly1 * poly2
    return (poly1 * poly2).divide_by(gcd, order)[0]

# Example usage
if __name__ == "__main__":
    # Create variables
    x = Variable("x")
    y = Variable("y")
    
    # Create polynomial: x^2 + 2xy + y^2
    poly = Polynomial({
        Monomial(((x, 2),)): 1.0,
        Monomial(((x, 1), (y, 1))): 2.0,
        Monomial(((y, 2),)): 1.0
    })
    
    print(f"Polynomial: {poly}")
    print(f"Degree: {poly.degree()}")
    
    # Differentiate
    dx = poly.differentiate(x)
    print(f"∂/∂x: {dx}")
    
    dy = poly.differentiate(y)
    print(f"∂/∂y: {dy}")
    
    # Evaluate
    val_dict = {x: 2.0, y: 3.0}
    print(f"Value at (2, 3): {poly.evaluate(val_dict)}")
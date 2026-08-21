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

    def __post_init__(self):
        # Guard against the easy mistake Variable(some_variable), which
        # otherwise produces variables whose name is another Variable.
        if not isinstance(self.name, str):
            raise TypeError(
                f"Variable.name must be a str, got {type(self).__name__}"
                f"({self.name!r}). Pass the plain name string instead."
            )

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

    def lcm(self, other: 'Monomial') -> 'Monomial':
        """Compute least common multiple."""
        lcm_vars = dict(self.variables)
        for var, exp in other.variables:
            lcm_vars[var] = max(lcm_vars.get(var, 0), exp)
        return Monomial(tuple(sorted(lcm_vars.items(), key=lambda item: item[0].name)))
    
    def is_relatively_prime(self, other: 'Monomial') -> bool:
        """Check if self and other share no common variable."""
        self_vars = dict(self.variables)
        for var, exp in other.variables:
            if var in self_vars and exp > 0 and self_vars[var] > 0:
                return False
        return True
    
    def total_degree(self) -> int:
        """Total degree of the monomial."""
        return self.degree()
    
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

    def is_constant(self) -> bool:
        return len(self.terms) == 0 or (len(self.terms) == 1 and self.terms.get(Monomial(), 0) != 0 and Monomial() in self.terms)

    def copy(self) -> 'Polynomial':
        return Polynomial(dict(self.terms))

    def variables(self) -> List[Variable]:
        """Set of variables in the polynomial"""
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

    def leading_term(self, order: str = 'grevlex', variables: Optional[List[Variable]] = None) -> Tuple[Monomial, float]:
        if self.is_zero():
            return Monomial(), 0.0
        lm = self.leading_monomial(order, variables)
        return lm, self.terms[lm]

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
            var_exp = 0
            for v, exp in sorted(monom.variables, key=lambda item: item[0].name):
                if v == var:
                    if exp > 0:
                        new_vars.append((v, exp - 1))
                        found = True
                        var_exp = exp
                else:
                    new_vars.append((v, exp))
            
            if found:
                new_monom_vars = tuple(sorted([v for v in new_vars if v[1] > 0], key=lambda item: item[0].name))
                new_monom = Monomial(new_monom_vars)
                result[new_monom] = result.get(new_monom, 0) + coeff * var_exp
        
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
                lt_monom, lt_coeff = work.leading_term(order, all_vars)
                lt_poly = Polynomial({lt_monom: lt_coeff})
                
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
        
        return " + ".join(terms_str).replace(" + -", " - ") if terms_str else "0"
    
    def __repr__(self) -> str:
        return f"Polynomial({self.terms})"

    @staticmethod
    def from_string(s: str, variables: List[Variable]) -> 'Polynomial':
        return parse_polynomial_string(s, variables)

    @staticmethod
    def constant(c: float) -> 'Polynomial':
        return Polynomial({Monomial(): c})

    @staticmethod
    def variable(v: Variable) -> 'Polynomial':
        return Polynomial({Monomial(((v, 1),)): 1.0})


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


def parse_polynomial_string(s: str, variables: List[Variable]) -> Polynomial:
    """
    Parses a string into a symbolic Polynomial object.

    Uses a small recursive-descent parser, so parentheses, exponentiation
    and distribution work correctly:
        "3x^2 + 2x*y - y", "(x+1)^2", "3*(x+1)^2", "(x+1)*(y-1)"
    Implicit multiplication ("2x", "3(x+1)") is also supported.
    """
    return _PolynomialParser(s, variables).parse()


class _PolynomialParser:
    """
    Recursive-descent parser for polynomial expressions.

    Grammar:
        expr   := term (('+' | '-') term)*
        term   := factor ('*' factor)*
        factor := ('-')? atom ('^' integer)?
        atom   := number | variable | '(' expr ')'
    """

    _TOKEN_RE = re.compile(r'\s*(?:(\d+\.\d+|\d+)|([A-Za-z_]\w*)|([+\-*^()]))')

    def __init__(self, s: str, variables: List[Variable]):
        self.var_map = {v.name: v for v in variables}
        self.tokens = self._tokenize(s)
        self.pos = 0

    def _tokenize(self, s: str) -> List[str]:
        tokens = []
        pos = 0
        prev = None
        while pos < len(s):
            match = self._TOKEN_RE.match(s, pos)
            if not match or match.end() == pos:
                if s[pos:].strip() == '':
                    break
                raise ValueError(f"Unexpected character at position {pos}: '{s[pos:]}'")
            token = match.group(1) or match.group(2) or match.group(3)
            # Insert implicit multiplication: 2x -> 2*x, x(y+1) -> x*(y+1), )( -> )*(
            if prev is not None and (
                (prev.isdigit() and re.match(r'[A-Za-z_(]', token)) or
                (prev == ')' and re.match(r'[A-Za-z0-9.(]', token))
            ):
                tokens.append('*')
            tokens.append(token)
            prev = token
            pos = match.end()
        return tokens

    def parse(self) -> Polynomial:
        poly = self._expr()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token: '{self.tokens[self.pos]}'")
        return poly

    def _peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _expr(self) -> Polynomial:
        result = self._term()
        while self._peek() in ('+', '-'):
            op = self._next()
            rhs = self._term()
            result = result + rhs if op == '+' else result - rhs
        return result

    def _term(self) -> Polynomial:
        result = self._factor()
        while self._peek() == '*':
            self._next()
            result = result * self._factor()
        return result

    def _factor(self) -> Polynomial:
        sign = 1.0
        while self._peek() in ('+', '-'):
            if self._next() == '-':
                sign = -sign
        
        atom = self._atom()
        
        if self._peek() == '^':
            self._next()
            exp_token = self._next()
            if not exp_token.isdigit():
                raise ValueError(f"Expected integer exponent, got '{exp_token}'")
            atom = atom ** int(exp_token)
        
        return atom * sign if sign < 0 else atom
    
    def _atom(self) -> Polynomial:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of expression")
        
        if token == '(':
            self._next()
            inner = self._expr()
            if self._next() != ')':
                raise ValueError("Missing closing parenthesis")
            return inner
        
        self._next()
        if re.match(r'^\d+\.\d+$|^\d+$', token):
            return Polynomial.constant(float(token))
        
        if token in self.var_map:
            return Polynomial.variable(self.var_map[token])
        
        raise ValueError(f"Unknown variable '{token}'. Known variables: "
                         f"{sorted(self.var_map.keys())}")

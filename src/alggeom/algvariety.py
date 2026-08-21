"""
Algebraic Variety Module
========================
Represents algebraic varieties (solution sets of polynomial systems).
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from .polynomial import Polynomial, Monomial, Variable
from .groebnerbasis import GroebnerBasis, compute_groebner_basis


@dataclass
class Point:
    """A point in affine space"""
    coordinates: Dict[str, complex]
    
    def __repr__(self) -> str:
        return f"Point({self.coordinates})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        return self.coordinates == other.coordinates
    
    def distance_to(self, other: Point) -> float:
        """Euclidean distance to another point"""
        vars1 = set(self.coordinates.keys())
        vars2 = set(other.coordinates.keys())
        common = vars1 & vars2
        
        dist_sq = 0
        for v in common:
            z1 = self.coordinates[v]
            z2 = other.coordinates[v]
            dist_sq += abs(z1 - z2)**2
        
        for v in vars1 - vars2:
            dist_sq += abs(self.coordinates[v])**2
        for v in vars2 - vars1:
            dist_sq += abs(other.coordinates[v])**2
        
        return np.sqrt(dist_sq)


class AlgebraicVariety:
    """
    Represents an algebraic variety V(I) = {x in K^n : f(x) = 0 for all f in I}
    """
    
    def __init__(self, ideal: List[Polynomial], field: str = 'complex'):
        self.ideal = ideal
        self.field = field
        self.groebner_basis: Optional[List[Polynomial]] = None
        self._compute_groebner_basis()
    
    def _compute_groebner_basis(self) -> None:
        """Compute Groebner basis of the ideal"""
        if self.ideal:
            self.groebner_basis = compute_groebner_basis(self.ideal)
    
    def contains(self, point: Point) -> bool:
        """Check if point lies on the variety"""
        if self.groebner_basis is None:
            return True

        # Accept coordinate keys given as Variable objects or plain names
        values = {}
        for k, v in point.coordinates.items():
            key = k if isinstance(k, Variable) else Variable(str(k))
            values[key] = v
        for poly in self.groebner_basis:
            if abs(poly.evaluate(values)) > 1e-10:
                return False
        return True
    
    def dimension(self) -> int:
        """
        Compute dimension as the maximum size of an independent set of
        variables S such that no leading monomial of the Groebner basis
        has support entirely inside S. This equals dim(K[x]/I).
        """
        if not self.groebner_basis:
            return -1
        
        gb = self.groebner_basis
        vars_set = set()
        for p in gb:
            vars_set.update(p.variables())
        n = len(vars_set)
        
        if n == 0:
            return 0
        
        all_vars_list = sorted(list(vars_set), key=lambda v: v.name)
        
        leading_supports = [
            set(p.leading_monomial('grevlex', all_vars_list).get_vars())
            for p in gb if not p.is_zero()
        ]
        
        from itertools import combinations
        for k in range(n, -1, -1):
            for subset in combinations(all_vars_list, k):
                s = set(subset)
                if all(not supp.issubset(s) for supp in leading_supports):
                    return k
        
        return 0
    
    def degree(self) -> int:
        """
        Compute the degree of the variety.
        The degree is the number of points in a generic linear intersection.
        """
        if not self.groebner_basis:
            return 0
        
        # If the variety is 0-dimensional, the degree is the number of points.
        # This is equal to the dimension of the quotient ring K[x]/I as a K-vector space.
        if self.dimension() == 0:
            # Count standard monomials (monomials not divisible by any leading term)
            leading_monomials = [p.leading_monomial() for p in self.groebner_basis]
            # This is complex to compute directly, but for a 0-dim ideal,
            # it's the number of solutions.
            # A full implementation is beyond this scope, but we note the method.
            pass

        # For higher dimensions, Bezout's theorem is a rough upper bound.
        degrees = [p.degree() for p in self.ideal if p.degree() > 0]
        return int(np.prod(degrees)) if degrees else 1
    
    def singular_points(self, max_points: int = 100) -> List[Point]:
        """
        Find singular points of the variety.
        A point p on V(I) is singular if rank(J_p(I)) < codim(V).
        """
        if not self.ideal:
            return []
        
        # For a hypersurface defined by a single polynomial f,
        # singular points are solutions to {f=0, df/dx_1=0, ..., df/dx_n=0}.
        if len(self.ideal) == 1:
            f = self.ideal[0]
            variables = list(f.variables())
            
            # Create the Jacobian ideal
            singular_ideal_gens = [f]
            for var in variables:
                singular_ideal_gens.append(f.differentiate(var))
            
            return solve_polynomial_system(singular_ideal_gens, variables)[:max_points]
        
        # For the general case, one needs to solve for where the minors of the Jacobian vanish.
        vars_set = set()
        for p in self.groebner_basis:
            vars_set.update(p.variables())
        vars_list = sorted(list(vars_set), key=lambda v: v.name)
        
        jacobian_ideal = list(self.groebner_basis)
        for p in self.groebner_basis:
            for var in vars_list:
                df = p.differentiate(var)
                if not df.is_zero():
                    jacobian_ideal.append(df)
                    
        return solve_polynomial_system(jacobian_ideal, vars_list)[:max_points]
    
    def irreducible_components(self) -> List[AlgebraicVariety]:
        """
        Decompose variety into irreducible components.
        """
        # Use primary decomposition
        from .groebnerbasis import primary_decomposition
        components = primary_decomposition(self.ideal)
        return [AlgebraicVariety(comp, self.field) for comp in components]
    
    def constraint_violations(self, point: np.ndarray) -> np.ndarray:
        """Compute violations of defining polynomials at a given point."""
        if not self.ideal:
            return np.zeros(0)
        
        vars_set = set()
        for p in self.ideal:
            vars_set.update(p.variables())
        var_names = sorted(list(vars_set), key=lambda v: v.name)

        coords = {}
        for i, name in enumerate(var_names):
            if i < len(point):
                coords[name] = point[i]
                
        violations = []
        for poly in self.ideal:
            val = poly.evaluate(coords)
            violations.append(val.real if isinstance(val, complex) else val)
        return np.array(violations)

    def constraint_jacobian(self, point: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute the full Jacobian matrix of the constraint functions at a point.
        The (i, j)-th entry is the partial derivative of the i-th polynomial
        with respect to the j-th variable.
        """
        if not self.ideal:
            return None
        
        vars_set = set()
        for p in self.ideal:
            vars_set.update(p.variables())
        var_names = sorted(list(vars_set), key=lambda v: v.name)
        n_vars = max(len(var_names), len(point))
        n_cons = len(self.ideal)
        
        coords = {name: point[i] for i, name in enumerate(var_names) if i < len(point)}
        jacobian = np.zeros((n_cons, n_vars))
        
        for i, poly in enumerate(self.ideal):
            for j, var in enumerate(var_names):
                if j < len(point):
                    deriv = poly.differentiate(var)
                    val = deriv.evaluate(coords)
                    jacobian[i, j] = val.real if isinstance(val, complex) else val
        return jacobian

    def constraint_lagrangian_gradient(self, point: np.ndarray, multipliers: np.ndarray) -> Optional[np.ndarray]:
        """Compute the gradient of the Lagrangian L(x, λ) = f(x) + λ^T * c(x)."""
        jacobian = self.constraint_jacobian(point)
        if jacobian is None:
            return None
        # The gradient of the constraint part is J^T * λ
        return jacobian.T @ multipliers
    
    def __repr__(self) -> str:
        n = len(self.ideal)
        return f"AlgebraicVariety(ideal with {n} generators)"


class AffineSpace(AlgebraicVariety):
    """The full affine space (trivial ideal {0})"""
    
    def __init__(self, dimension: int, variables: List[str]):
        self._ambient_dimension = dimension
        self.variables = variables
        super().__init__([Polynomial.constant(0)], 'complex')
    
    def dimension(self) -> int:
        return self._ambient_dimension


class Hypersurface(AlgebraicVariety):
    """A hypersurface defined by a single polynomial f = 0"""
    
    def __init__(self, polynomial: Polynomial):
        self.polynomial = polynomial
        super().__init__([polynomial], 'complex')
    
    def is_smooth(self) -> bool:
        """Check if hypersurface is smooth (no singular points)"""
        return len(self.singular_points()) == 0
    
    def genus(self) -> int:
        """Genus of a smooth plane curve (for curves only)"""
        d = self.polynomial.degree()
        return (d - 1) * (d - 2) // 2


class ProjectiveVariety:
    """
    Projective algebraic variety.
    V(I) in P^n where I is a homogeneous ideal.
    """
    
    def __init__(self, homogeneous_ideal: List[Polynomial]):
        self.homogeneous_ideal = homogeneous_ideal
        self.groebner_basis = compute_groebner_basis(homogeneous_ideal)
    
    def dimension(self) -> int:
        """Projective dimension"""
        # dim = n - height(I) for projective varieties
        return 0  # Simplified
    
    def degree(self) -> int:
        """Projective degree"""
        return 1  # Simplified


def solve_polynomial_system(polynomials: List[Polynomial],
                           variables: List,
                           method: str = 'groebner') -> List[Point]:
    """
    Solve a system of polynomial equations.
    
    Args:
        polynomials: List of polynomial equations
        variables: List of variables (Variable objects or names)
        method: 'groebner' or 'numerical'
    
    Returns:
        List of solutions as Point objects
    """
    if method == 'groebner':
        return _solve_via_groebner(polynomials, variables)
    else:
        return _solve_numerically(polynomials, variables)


def _as_variable(v) -> Variable:
    """Accept either a Variable or a plain string name."""
    from .polynomial import Variable as V
    return v if isinstance(v, V) else V(str(v))


def _substitute(poly: Polynomial, var: Variable, value: complex) -> Polynomial:
    """
    Substitute a numeric value for one variable, returning a polynomial
    in the remaining variables (or a constant).
    """
    terms = {}
    for monom, coeff in poly.terms.items():
        exp = dict(monom.variables).get(var, 0)
        if exp == 0:
            new_monom = monom
        else:
            remaining = tuple((v, e) for v, e in monom.variables if v != var)
            new_monom = Monomial(remaining)
        c = coeff * (value ** exp)
        if abs(c) > 1e-12:
            terms[new_monom] = terms.get(new_monom, 0.0) + c
    return Polynomial({m: c for m, c in terms.items() if abs(c) > 1e-12})


def _solve_via_groebner(polynomials: List[Polynomial], 
                        variables: List) -> List[Point]:
    """
    Solve using Groebner basis elimination with lex order and
    back-substitution of the eliminated variables.
    """
    vars_objs = [_as_variable(v) for v in variables]
    ordered_vars = sorted(vars_objs, key=lambda v: v.name)
    gb = GroebnerBasis([p for p in polynomials if not p.is_zero()], order='lex')
    
    solutions = []
    _solve_recursive(gb.groebner_basis, ordered_vars, {}, solutions)

    # De-duplicate numerically identical solutions, preserving order
    points = []
    seen = set()
    for sol in solutions:
        key = tuple(round(float(sol[v].real), 8) for v in ordered_vars)
        if key in seen:
            continue
        seen.add(key)
        points.append(Point({v.name: sol[v] for v in sol}))
    return points


def _solve_recursive(gb: List[Polynomial],
                     remaining_vars: List[Variable],
                     partial: Dict[Variable, complex],
                     solutions: List[Dict]) -> None:
    """Back-substitution: solve the univariate polynomial in the last
    remaining variable, substitute each root, and recurse."""
    if not remaining_vars:
        solutions.append(dict(partial))
        return
    
    last_var = remaining_vars[-1]
    
    # Find a univariate polynomial in the last variable
    univariate = None
    for p in gb:
        p_vars = p.variables()
        if len(p_vars) == 0:
            # Constant contradiction (e.g. 1 = 0): no solution on this branch
            if abs(p.evaluate({})) > 1e-10:
                return
            continue
        if p_vars == [last_var]:
            univariate = p
            break
    
    if univariate is None:
        # Positive-dimensional component: cannot enumerate solutions here
        return
    
    # Build coefficient array for np.roots (descending powers)
    deg = int(univariate.degree())
    coeffs = np.zeros(deg + 1, dtype=complex)
    for monom, coeff in univariate.terms.items():
        exp = dict(monom.variables).get(last_var, 0)
        coeffs[deg - exp] += coeff
    
    for root in np.roots(coeffs):
        if abs(root.imag) < 1e-8:
            root = root.real
        if abs(root.imag) > 1e-6:
            continue  # keep real solutions only
        
        # Substitute the root into the rest of the basis and recurse
        reduced_gb = []
        consistent = True
        for p in gb:
            q = _substitute(p, last_var, root)
            if q.is_zero():
                continue
            if len(q.variables()) == 0:
                # Contradiction after substitution
                consistent = False
                break
            reduced_gb.append(q)
        
        if not consistent:
            continue
        
        partial[last_var] = root
        _solve_recursive(reduced_gb, remaining_vars[:-1], partial, solutions)
        partial.pop(last_var, None)


def _solve_numerically(polynomials: List[Polynomial],
                      variables: List[str],
                      max_solutions: int = 100) -> List[Point]:
    """Solve numerically using homotopy continuation or Newton iteration"""
    # Simplified numerical solver
    # Full implementation would use:
    # - Homotopy continuation (Bertini, PHCpack)
    # - Newton-Raphson with multiple starting points
    # - Resultant-based methods
    
    n_vars = len(variables)
    solutions = []
    
    # For demonstration, return empty list
    # Real implementation would generate candidate solutions
    return solutions


def intersection(V1: AlgebraicVariety, V2: AlgebraicVariety) -> AlgebraicVariety:
    """Intersection of two varieties: V(I) ∩ V(J) = V(I + J)"""
    combined_ideal = V1.ideal + V2.ideal
    return AlgebraicVariety(combined_ideal)


def union(V1: AlgebraicVariety, V2: AlgebraicVariety) -> AlgebraicVariety:
    """Union of two varieties: V(I) ∪ V(J) = V(IJ)"""
    # Product of ideals
    product_ideal = []
    for f in V1.ideal:
        for g in V2.ideal:
            product_ideal.append(f * g)
    return AlgebraicVariety(product_ideal)


def closure(V: AlgebraicVariety) -> AlgebraicVariety:
    """Zariski closure of a variety (always itself for affine varieties)"""
    return V
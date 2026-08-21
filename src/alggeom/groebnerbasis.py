"""
Groebner Basis Module
=====================
Implementation of Buchberger's algorithm for computing Groebner bases.
"""

from __future__ import annotations
from typing import List, Tuple, Optional 
from .polynomial import Polynomial, Monomial, Variable
import copy
import concurrent.futures
import numpy as np


class GroebnerBasis:
    """
    Computes and stores Groebner basis of an ideal.
    """
    
    def __init__(self, polynomials: List[Polynomial], order: str = 'grevlex', n_workers: int = 1):
        self.original_polynomials = polynomials
        self.order = order
        self.n_workers = n_workers
        # One global variable list so every leading term is computed under
        # the same monomial order.
        vars_set = set()
        for p in polynomials:
            vars_set.update(p.variables())
        self.all_vars = sorted(list(vars_set), key=lambda v: v.name)
        self.groebner_basis: List[Polynomial] = []
        self._compute_basis()
    
    def compute(self) -> None:
        """Compute the basis (already done in init, provided for compatibility)."""
        pass
        
    @property
    def basis(self) -> List[Polynomial]:
        """Get the Groebner basis."""
        return self.groebner_basis

    def _compute_basis(self) -> None:
        """Compute Groebner basis using Buchberger's algorithm"""
        if not self.original_polynomials:
            self.groebner_basis = []
            return
        
        # Initialize with non-zero polynomials
        G = [p for p in self.original_polynomials if not p.is_zero()]
        
        if len(G) <= 1:
            self.groebner_basis = G
            return
        
        # Buchberger's algorithm with optimizations (Product, Chain, Normal Selection)
        pairs = []
        for i in range(1, len(G)):
            pairs = self._update_pairs(G, pairs, i)
        
        while pairs:
            # Normal selection strategy: sort by LCM degree, pick lowest
            pairs.sort(key=lambda x: x[2])
            i, j, _ = pairs.pop(0)
            Gi, Gj = G[i], G[j]
            
            # Compute S-polynomial
            spoly = self._s_polynomial(Gi, Gj)
            
            if spoly.is_zero():
                continue
            
            # Reduce S-polynomial
            remainder = self._reduce(spoly, G)
            
            if not remainder.is_zero():
                # Add new polynomial
                G.append(remainder)
                
                # Add new pairs
                pairs = self._update_pairs(G, pairs, len(G) - 1)
                
        # Reduction of existing polynomials is done after basis is generated
        # (Doing this inside the loop would invalidate pair indices)
        G = self._reduce_basis(G)
        
        self.groebner_basis = [p for p in G if not p.is_zero()]
    
    def _update_pairs(self, G: List[Polynomial], pairs: List[Tuple[int, int, int]], new_idx: int) -> List[Tuple[int, int, int]]:
        """
        Updates the pair queue using Buchberger's First and Second Criteria.
        """
        i = new_idx
        p_i = G[i]
        lm_i = p_i.leading_monomial(self.order, self.all_vars)
        
        new_pairs = []
        for j in range(i):
            p_j = G[j]
            lm_j = p_j.leading_monomial(self.order, self.all_vars)
            
            # 1. Product Criterion: If leading monomials are relatively prime, S-poly reduces to 0
            if lm_i.is_relatively_prime(lm_j):
                continue
                
            lcm_ij = lm_i.lcm(lm_j)
            
            # 2. Chain Criterion: Check if there's an intermediate polynomial G[k] covering (i, j)
            is_redundant = any(
                G[k].leading_monomial(self.order, self.all_vars).divides(lcm_ij) for k in range(i) if k != j
            )
                    
            if not is_redundant:
                # Store total degree for Normal Selection Strategy
                degree = lcm_ij.total_degree()
                new_pairs.append((j, i, degree))
                
        pairs.extend(new_pairs)
        return pairs

    def _s_polynomial(self, f: Polynomial, g: Polynomial) -> Polynomial:
        """Compute S-polynomial of f and g"""
        f_lm = f.leading_monomial(self.order, self.all_vars)
        g_lm = g.leading_monomial(self.order, self.all_vars)
        
        # LCM of leading monomials
        lcm_monomial = f_lm.lcm(g_lm)
        
        # Compute multipliers (lcm / leading monomial)
        f_mult = f_lm.div(lcm_monomial)
        g_mult = g_lm.div(lcm_monomial)
        
        return Polynomial({f_mult: 1.0/f.leading_coefficient(self.order, self.all_vars)}) * f - Polynomial({g_mult: 1.0/g.leading_coefficient(self.order, self.all_vars)}) * g
    
    def _reduce(self, poly: Polynomial, G: List[Polynomial]) -> Polynomial:
        """Reduce polynomial by Groebner basis G using optimized division."""
        remainder = poly.copy()
        reduced_poly = Polynomial()
        
        # Extend the global variable list so the order is well defined even
        # if `poly` uses variables not present in G.
        vars_set = set(self.all_vars)
        vars_set.update(poly.variables())
        all_vars = sorted(list(vars_set), key=lambda v: v.name)
                
        # Precompute leading attributes to avoid redundant linear-time recomputations
        G_precomputed = [
            (g, g.leading_monomial(self.order, all_vars), g.leading_coefficient(self.order, all_vars)) 
            for g in G if not g.is_zero()
        ]
        
        while not remainder.is_zero():
            remainder_lm = remainder.leading_monomial(self.order, all_vars)
            remainder_lc = remainder.leading_coefficient(self.order, all_vars)
            divided = False
            
            for g, g_lm, g_lc in G_precomputed:
                if g_lm.divides(remainder_lm):
                    # Compute quotient (remainder_lm / g_lm)
                    factor_monomial = g_lm.div(remainder_lm)
                    factor_coeff = remainder_lc / g_lc
                    factor = Polynomial({factor_monomial: factor_coeff})
                    
                    remainder = remainder - factor * g
                    divided = True
                    break
            
            if not divided:
                # Cannot divide further, move leading term to the reduced result
                lt_monom, lt_coeff = remainder.leading_term(self.order, all_vars)
                lt_poly = Polynomial({lt_monom: lt_coeff})
            
                reduced_poly = reduced_poly + lt_poly
                    
                remainder = remainder - lt_poly
        
        return reduced_poly
    
    def _minimize_leading_monomials(self, G: List[Polynomial]) -> List[Polynomial]:
        """
        Remove redundant generators: if the leading monomial of some other
        generator divides lm(g), then g can be removed while the remaining
        polynomials still form a Groebner basis of the same ideal.
        Ties (equal leading monomials) keep exactly one generator.
        """
        nonzero = [g for g in G if not g.is_zero()]
        kept = []
        for i, g in enumerate(nonzero):
            lm_g = g.leading_monomial(self.order, self.all_vars)
            redundant = False
            for j, h in enumerate(nonzero):
                if i == j:
                    continue
                lm_h = h.leading_monomial(self.order, self.all_vars)
                if lm_h.divides(lm_g) and (lm_h != lm_g or j < i):
                    redundant = True
                    break
            if not redundant:
                kept.append(g)
        return kept
    
    def _reduce_basis(self, G: List[Polynomial]) -> List[Polynomial]:
        """Reduce all polynomials in the basis"""
        # First remove generators with redundant leading monomials so that
        # inter-reduction cannot cancel an entire generator (which would
        # change the generated ideal).
        G = self._minimize_leading_monomials(G)
        reduced = []
        
        if self.n_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = []
                for i, g in enumerate(G):
                    others = G[:i] + G[i+1:]
                    futures.append(executor.submit(self._reduce, g, others))
                for future in futures:
                    g_reduced = future.result()
                    if not g_reduced.is_zero():
                        reduced.append(g_reduced)
        else:
            for i, g in enumerate(G):
                others = G[:i] + G[i+1:]
                g_reduced = self._reduce(g, others)
                if not g_reduced.is_zero():
                    reduced.append(g_reduced)
        return reduced
    
    def reduce(self) -> List[Polynomial]:
        """Return reduced Groebner basis"""
        return self._reduce_basis(self.groebner_basis)
    
    def is_groebner_basis(self) -> bool:
        """Check if stored basis is a Groebner basis"""
        for i, gi in enumerate(self.groebner_basis):
            for gj in self.groebner_basis[i+1:]:
                spoly = self._s_polynomial(gi, gj)
                remainder = self._reduce(spoly, self.groebner_basis)
                if not remainder.is_zero():
                    return False
        return True
    
    def ideal_membership(self, poly: Polynomial) -> Tuple[bool, Optional[Polynomial]]:
        """
        Check if polynomial is in the ideal.
        Returns (is_member, remainder)
        """
        remainder = self._reduce(poly, self.groebner_basis)
        return remainder.is_zero(), None if remainder.is_zero() else remainder
    
    def __repr__(self) -> str:
        return f"GroebnerBasis({len(self.groebner_basis)} polynomials)"


def compute_groebner_basis(polynomials: List[Polynomial],
                          order: str = 'grevlex',
                          n_workers: int = 1) -> List[Polynomial]:
    """
    Compute Groebner basis of ideal generated by polynomials.
    
    Args:
        polynomials: List of generating polynomials
        order: Monomial order ('lex', 'grlex', 'grevlex')
        n_workers: Number of threads for parallel reduction operations
    
    Returns:
        List of polynomials forming the Groebner basis
    """
    gb = GroebnerBasis(polynomials, order, n_workers)
    return gb.groebner_basis


def ideal_intersection(G1: List[Polynomial], G2: List[Polynomial],
                      variables: List[str]) -> List[Polynomial]:
    """
    Compute intersection of two ideals using elimination theory.
    
    Args:
        G1: Groebner basis of first ideal
        G2: Groebner basis of second ideal
        variables: List of original variables
    
    Returns:
        Groebner basis of the intersection ideal
    """
    # Find a safe dummy variable name to avoid collision
    used_vars = set(variables)
    new_var = 't_int_0'
    i = 0
    while new_var in used_vars:
        i += 1
        new_var = f't_int_{i}'
        
    t_var = Variable(new_var)
    t = Polynomial.variable(t_var)
    one_minus_t = Polynomial.constant(1) - t
    
    # Combine generators
    combined = []
    for p in G1:
        combined.append(t * p)
    for p in G2:
        combined.append(one_minus_t * p)
    
    # Compute Groebner basis with lex order on t
    gb = GroebnerBasis(combined, 'lex')
    
    # Extract polynomials without t
    intersection = []
    for p in gb.groebner_basis:
        if t_var not in p.variables():
            intersection.append(p)
    
    return intersection


def radical_membership(poly: Polynomial, ideal: List[Polynomial],
                       order: str = 'grevlex') -> bool:
    """
    Check if polynomial is in the radical of an ideal.
    Uses the algorithm: f in rad(I) iff f is in I + <f*x - 1> in some extension.
    """
    # Find a safe variable name
    used_vars = set(poly.variables())
    for p in ideal:
        used_vars.update(p.variables())
        
    new_var = 'y_rad_0'
    i = 0
    while new_var in used_vars:
        i += 1
        new_var = f'y_rad_{i}'
        
    y_var = Variable(new_var)
    y = Polynomial.variable(y_var)
    one = Polynomial.constant(1)
    
    # Create ideal I + <f*y - 1>
    extended = list(ideal) + [poly * y - one]
    
    # Compute Groebner basis
    gb = GroebnerBasis(extended, order)
    
    # Check if 1 is in the ideal
    for p in gb.groebner_basis:
        if p.is_constant() and p.terms.get(Monomial(()), 0) != 0:
            return True
    return False


def _remove_redundant_components(components: List[List[Polynomial]], order: str) -> List[List[Polynomial]]:
    """
    Remove redundant ideals from a decomposition.
    If A \\subseteq B, then A \\cap B = A, so B is redundant.
    """
    if not components:
        return []
        
    # Compute Groebner bases for membership testing
    bases = [GroebnerBasis(comp, order) for comp in components]
    
    cleaned = []
    for i, gb_B in enumerate(bases):
        is_redundant = False
        
        # Skip the whole ring
        if any(p.is_constant() and p.terms.get(Monomial(()), 0) != 0 for p in gb_B.groebner_basis):
            continue
            
        for j, gb_A in enumerate(bases):
            if i == j:
                continue
                
            # Check if A \\subseteq B (every generator of A is in B)
            # If true, A \\cap B = A, so B is not needed in the intersection
            if all(gb_B.ideal_membership(gen_A)[0] for gen_A in components[j]):
                # Break mutual containment ties (A == B) by discarding only if i > j
                b_subset_a = all(gb_A.ideal_membership(gen_B)[0] for gen_B in components[i])
                
                if b_subset_a and i < j:
                    pass # Keep this one, discard the other when its turn comes
                else:
                    is_redundant = True
                    break
                    
        if not is_redundant:
            cleaned.append(components[i])
            
    return cleaned


def primary_decomposition(ideal: List[Polynomial], 
                         order: str = 'grevlex') -> List[List[Polynomial]]:
    """
    Compute primary decomposition of an ideal.
    Returns list of primary ideals whose intersection equals the input ideal.
    
    Uses a splitting algorithm based on saturation and polynomial factorization 
    (or common variable extraction) to isolate geometric components.
    """
    gb = compute_groebner_basis(ideal, order)
    
    # Helper: Check if ideal is the whole ring (contains 1)
    for p in gb:
        if p.is_constant() and p.terms.get(Monomial(()), 0) != 0:
            return []
            
    # Try to find a splitting polynomial
    for p in gb:
        if p.is_zero() or p.is_constant():
            continue
            
        # 1. Check if the polynomial natively supports factorization (fallback hook)
        if hasattr(p, 'factor'):
            factors = p.factor()
            if len(factors) > 1:
                f1 = factors[0]
                part1 = gb + [f1]
                part2 = saturation(gb, f1, order)
                res1 = primary_decomposition(part1, order)
                res2 = primary_decomposition(part2, order)
                return _remove_redundant_components(res1 + res2, order)
        
        # 2. Heuristic: Splitting single monomials with multiple variables (e.g., x^2 * y)
        if len(p.terms) == 1:
            vars_in_p = p.variables()
            if len(vars_in_p) > 1:
                # Split using the first variable
                v_poly = Polynomial.variable(vars_in_p[0])
                part1 = gb + [v_poly]
                part2 = saturation(gb, v_poly, order)
                res1 = primary_decomposition(part1, order)
                res2 = primary_decomposition(part2, order)
                return _remove_redundant_components(res1 + res2, order)
        
        # 3. Heuristic: Extracting common variable factors across multiple terms (e.g., xy + xz = x(y+z))
        if len(p.terms) > 1:
            for v_name in p.variables():
                v_poly = Polynomial.variable(v_name)
                v_mono = v_poly.leading_monomial(order)
                
                # Check if v_mono divides all monomials
                if all(v_mono.divides(m) for m in p.terms.keys()):
                    # p = v * (p / v), split on v
                    part1 = gb + [v_poly]
                    part2 = saturation(gb, v_poly, order)
                    res1 = primary_decomposition(part1, order)
                    res2 = primary_decomposition(part2, order)
                    return _remove_redundant_components(res1 + res2, order)
                    
    # If no splitting is found, we assume it's a primary component 
    return [gb]


def saturation(ideal: List[Polynomial], polynomial: Polynomial,
               order: str = 'grevlex') -> List[Polynomial]:
    """
    Compute the saturation of an ideal with respect to a polynomial.
    I:f^infty = {g in K[x] | f^k * g in I for some k >= 0}
    """
    # Find a safe variable name
    used_vars = set(polynomial.variables())
    for p in ideal:
        used_vars.update(p.variables())
        
    new_var = 't_sat_0'
    i = 0
    while new_var in used_vars:
        i += 1
        new_var = f't_sat_{i}'
        
    t_var = Variable(new_var)
    t = Polynomial.variable(t_var)
    
    # Create extended ideal: I + <1 - t*f>
    extended = list(ideal) + [Polynomial.constant(1) - t * polynomial]
    
    # Compute Groebner basis with lex order on t
    gb = GroebnerBasis(extended, 'lex')
    
    # Extract polynomials without t
    saturation = []
    for p in gb.groebner_basis:
        if t_var not in p.variables():
            saturation.append(p)
    
    return saturation


def build_macaulay_matrix(polynomials: List[Polynomial], order: str = 'grevlex') -> Tuple[np.ndarray, List[Monomial]]:
    """
    Builds the Macaulay matrix for a set of polynomials.
    This is the foundational linear algebra step required for implementing the F4 algorithm.
    """
    # Collect all unique monomials across all polynomials
    monomials_set = set()
    all_vars_set = set()
    for p in polynomials:
        monomials_set.update(p.terms.keys())
        all_vars_set.update(p.variables())
        
    all_vars = sorted(list(all_vars_set), key=lambda v: v.name)
    
    # Sort monomials based on the specified order (descending)
    def exponent(m: Monomial, v: Variable) -> int:
        return dict(m.variables).get(v, 0)
    
    if order == 'lex':
        sort_key = lambda m: tuple(exponent(m, v) for v in all_vars)
    elif order == 'grlex':
        sort_key = lambda m: (m.degree(), tuple(exponent(m, v) for v in all_vars))
    else:  # grevlex
        sort_key = lambda m: (m.degree(), tuple(-exponent(m, v) for v in reversed(all_vars)))
        
    sorted_monomials = sorted(list(monomials_set), key=sort_key, reverse=True)
    monomial_to_idx = {m: i for i, m in enumerate(sorted_monomials)}
    
        
    # Build Macaulay coefficient matrix
    matrix = np.zeros((len(polynomials), len(sorted_monomials)), dtype=complex)
    for i, p in enumerate(polynomials):
        for m, coeff in p.terms.items():
            matrix[i, monomial_to_idx[m]] = coeff


            
    return matrix, sorted_monomials
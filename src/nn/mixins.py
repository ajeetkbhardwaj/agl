"""
Algebraic Extraction Mixin
==========================
Provides universal symbolic extraction capabilities to polynomial neural networks.
"""

import torch
import numpy as np
from typing import List, Any

class AlgebraicExtractionMixin:
    def get_global_equation(self, feature_names: List[str], threshold: float = 1e-3,
                            max_global_degree: int = None, round_to: int = 3) -> Any:
        """
        The "Everything Equation": Extracts a single, global multivariate polynomial
        mapping inputs directly to outputs, algebraically bypassing all hidden layers.

        Set `round_to=None` to obtain the numerically exact equation.
        """
        import sympy as sp
        syms = [sp.Symbol(name) for name in feature_names]

        for layer in self.layers:
            out_syms = []

            # 1. Handle Polynomial Layers (weights are polynomial functions)
            if hasattr(layer, 'polynomial_weights') and hasattr(layer, '_sync_polynomials'):
                layer._sync_polynomials()
                W = layer.weight_matrix.detach().cpu().numpy()
                for j in range(layer.output_dim):
                    eq = sp.Float(layer.bias[j].item()) if layer.bias is not None else sp.Integer(0)
                    for i, sym in enumerate(syms):
                        # Linear component contributed by the learnable weight matrix
                        eq += sp.Float(float(W[j, i])) * sym
                        # Polynomial component: poly_{ji}(x) * x_i
                        poly_expr = self._polynomial_to_sympy(layer.polynomial_weights[j][i], syms)
                        eq += sp.expand(poly_expr * sym)
                    out_syms.append(sp.expand(eq))

            # 2. Handle Chebyshev / Orthogonal Factorized Layers
            elif hasattr(layer, 'cheby_coeffs') and hasattr(layer, 'linear_weight'):
                if layer.linear_weight.ndim != 3:
                    raise NotImplementedError(
                        "Symbolic extraction requires 'additive' interaction mode."
                    )
                out_dim = layer.bias.shape[0] if layer.bias is not None else layer.linear_weight.shape[0]

                # Collapse the rank dimension: (output_dim, input_dim, D+1)
                effective_coeffs = torch.einsum('oir,rid->oid', layer.linear_weight, layer.cheby_coeffs)

                for j in range(out_dim):
                    eq = layer.bias[j].item() if layer.bias is not None else 0.0
                    for i, sym in enumerate(syms):
                        run_min = getattr(layer, 'running_min', torch.zeros(layer.input_dim))[i].item()
                        run_max = getattr(layer, 'running_max', torch.ones(layer.input_dim))[i].item()
                        span = run_max - run_min
                        # Mirror the layer's own domain normalization exactly,
                        # including the clamp to [-1, 1].
                        if span > 1e-6:
                            scaled_sym = 2.0 * (sym - run_min) / span - 1.0
                        else:
                            scaled_sym = 2.0 * sym - 1.0
                        scaled_sym = sp.Max(-1.0, sp.Min(1.0, scaled_sym))

                        for d_idx in range(effective_coeffs.shape[2]):
                            c = effective_coeffs[j, i, d_idx].item()
                            if abs(c) > threshold:
                                eq += c * sp.chebyshevt(d_idx, scaled_sym)
                    out_syms.append(eq)

            # 3. Handle Groebner-constrained Linear Mappings (W stored as (in, out))
            elif hasattr(layer, 'groebner_basis') and hasattr(layer, 'weight_matrix'):
                weights = layer.weight_matrix.detach().cpu().numpy()
                bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(layer.output_dim)
                for j in range(layer.output_dim):
                    eq = float(bias[j])
                    for i, sym in enumerate(syms):
                        eq += float(weights[i, j]) * sym
                    out_syms.append(eq)

            # 4. Handle Standard Linear / Dense Mappings (W stored as (out, in))
            elif hasattr(layer, 'weight_matrix'):
                out_dim = layer.output_dim
                weights = layer.weight_matrix.detach().cpu().numpy()
                bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(out_dim)
                for j in range(out_dim):
                    eq = bias[j]
                    for i, sym in enumerate(syms):
                        eq += weights[j, i] * sym
                    out_syms.append(eq)
            else:
                # 5. Handle activation / normalization modules symbolically
                out_syms = self._apply_module_symbolically(layer, syms)

            syms = out_syms

        final_eq = syms[0]
        expanded_eq = sp.expand(final_eq)

        # Filter out negligible cross-terms and clip to max physical degree (Macroscopic Truncation)
        if isinstance(expanded_eq, sp.Add):
            terms = expanded_eq.args
        else:
            terms = [expanded_eq]

        cleaned_terms = []
        for term in terms:
            coeff, factors = term.as_coeff_Mul()
            if abs(float(coeff)) < threshold:
                continue
            if max_global_degree is not None:
                term_degree = sum(sp.degree(factors, sym) for sym in factors.free_symbols)
                if term_degree > max_global_degree:
                    continue
            cleaned_terms.append(term)

        pruned_eq = sp.Add(*cleaned_terms)
        if round_to is not None:
            pruned_eq = pruned_eq.xreplace(
                {n: round(n, round_to) for n in pruned_eq.atoms(sp.Number)}
            )
        return pruned_eq

    @staticmethod
    def _polynomial_to_sympy(poly, syms: List[Any]) -> Any:
        """Convert an alggeom Polynomial into a sympy expression over `syms`."""
        import sympy as sp
        expr = sp.Integer(0)
        for monom, coeff in poly.terms.items():
            term = sp.Float(float(coeff))
            for var, exp in monom.variables:
                idx = int(var.name[1:]) if var.name.startswith('x') else 0
                term = term * syms[idx] ** exp
            expr = expr + term
        return expr

    @staticmethod
    def _apply_module_symbolically(layer: Any, syms: List[Any]) -> List[Any]:
        """Lift a non-parametric module (activation, dropout, ...) to sympy space."""
        import sympy as sp
        name = type(layer).__name__.lower()
        lifted = []
        for s in syms:
            if name == 'tanh':
                lifted.append(sp.tanh(s))
            elif name == 'relu':
                lifted.append(sp.Max(0, s))
            elif name == 'sigmoid':
                lifted.append(1 / (1 + sp.exp(-s)))
            elif name == 'silu':
                lifted.append(s / (1 + sp.exp(-s)))
            elif name == 'gelu':
                lifted.append(s * 0.5 * (1 + sp.erf(s / sp.sqrt(2))))
            elif name in ('dropout', 'identity', 'flatten'):
                lifted.append(s)
            else:
                raise NotImplementedError(
                    f"Cannot symbolically lift module '{type(layer).__name__}'."
                )
        return lifted

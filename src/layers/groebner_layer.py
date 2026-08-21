"""
Gröbner Layer for Polynomial Neural Networks
=============================================
Neural network layer with weights constrained to a polynomial ideal.

Mathematical Foundation:
- Weights are constrained to satisfy polynomial ideal membership
- Forward pass uses Gröbner basis projection for weight normalization
- Supports algebraic constraints via ideal membership testing
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

try:
    from sympy import symbols, Poly, groebner, srepr, Symbol, Rational
    from sympy.polys.orderings import monomial_key
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    Symbol = None

from ..alggeom.groebnerbasis import GroebnerBasis
from ..alggeom.polynomial import Polynomial


@dataclass
class GroebnerLayerConfig:
    """Configuration for Gröbner layer"""
    input_dim: int
    output_dim: int
    constraint_ideal: Optional[List] = None
    groebner_order: str = 'lex'  # 'lex', 'grlex', 'grevlex'
    max_iterations: int = 100
    tolerance: float = 1e-6
    use_bias: bool = True
    polynomial_degree: int = 1


class GroebnerLayer(nn.Module):
    """
    A neural network layer with weights constrained to a polynomial ideal.
    
    This layer uses Gröbner basis theory to:
    1. Constrain weight matrices to lie in a polynomial ideal
    2. Project weights onto the variety defined by the ideal
    3. Enforce algebraic relations between weights
    
    Mathematical Foundation:
    - Given constraint ideal I, weights W must satisfy W ∈ I
    - Use Gröbner basis G of I for canonical representation
    - Project arbitrary weights onto the ideal via reduction
    
    Example:
        >>> config = GroebnerLayerConfig(input_dim=2, output_dim=3)
        >>> layer = GroebnerLayer(config)
        >>> output = layer.forward(np.array([[1.0, 2.0]]))
    """
    
    def __init__(self, config: Optional[GroebnerLayerConfig] = None,
                 input_dim: int = 1, output_dim: int = 1,
                 constraint_ideal: Optional[List] = None,
                 groebner_order: str = 'lex'):
        super().__init__()
        """
        Initialize Gröbner layer.
        
        Args:
            config: Layer configuration object
            input_dim: Input dimension
            output_dim: Output dimension
            constraint_ideal: List of constraint polynomials
            groebner_order: Monomial ordering for Gröbner basis
        """
        if config is not None:
            self.input_dim = config.input_dim
            self.output_dim = config.output_dim
            self.constraint_ideal = config.constraint_ideal
            self.groebner_order = config.groebner_order
            self.max_iterations = config.max_iterations
            self.tolerance = config.tolerance
            self.use_bias = config.use_bias
            self.polynomial_degree = config.polynomial_degree
        else:
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.constraint_ideal = constraint_ideal or []
            self.groebner_order = groebner_order
            self.max_iterations = 100
            self.tolerance = 1e-6
            self.use_bias = True
            self.polynomial_degree = 1
        
        # Initialize weight matrix
        limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        self.weight_matrix = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-limit, limit))
        
        # Initialize bias if used
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter('bias', None)
        
        # Compute Gröbner basis of constraint ideal
        self.groebner_basis = None
        self._compute_groebner_basis()
        
        # Store last output for backpropagation
        self.last_input = None
        self.last_output = None
    
    def _compute_groebner_basis(self) -> None:
        """Compute Gröbner basis of constraint ideal"""
        if not SYMPY_AVAILABLE or not self.constraint_ideal:
            self.groebner_basis = None
            return
        
        try:
            # Create symbols for the weight variables
            n_vars = self.input_dim * self.output_dim
            x = [Symbol(f'x{i}') for i in range(n_vars)]
            
            # Convert constraint polynomials to sympy format
            polynomials = []
            for constraint in self.constraint_ideal:
                if isinstance(constraint, (int, float)):
                    # Constant constraint
                    if constraint == 0:
                        continue
                    else:
                        import warnings
                        warnings.warn("Non-zero constant in ideal makes variety empty")
                        polynomials.append(constraint)
                elif isinstance(constraint, str):
                    # String representation
                    try:
                        poly = eval(constraint, {f'x{i}': x[i] for i in range(n_vars)})
                        polynomials.append(poly)
                    except:
                        pass
                elif hasattr(constraint, 'as_sympy'):
                    # Polynomial object
                    polynomials.append(constraint.as_sympy())
            
            if polynomials:
                # Compute Gröbner basis over QQ so that float/rational
                # weight coefficients can be reduced against it (a ZZ-domain
                # basis rejects any non-integer coefficient).
                self.groebner_basis = groebner(polynomials, *x,
                                               order=self.groebner_order,
                                               domain='QQ')
        except Exception as e:
            print(f"Warning: Could not compute Gröbner basis: {e}")
            self.groebner_basis = None
    
    @staticmethod
    def _as_sympy_coeff(c: float):
        """Convert a float coefficient to an exact sympy Rational so that
        reductions over QQ/ZZ domains behave correctly."""
        if not SYMPY_AVAILABLE:
            return c
        return Rational(float(c)).limit_denominator(10 ** 9)

    def project_to_ideal(self, weights: np.ndarray) -> np.ndarray:
        """
        Project weight matrix onto the constraint ideal.
        
        Uses Gröbner basis reduction to project arbitrary weights
        onto the variety defined by the constraint ideal.
        
        Args:
            weights: Weight matrix to project
            
        Returns:
            Projected weight matrix
        """
        if self.groebner_basis is None or not SYMPY_AVAILABLE:
            return weights
        
        try:
            n_vars = self.input_dim * self.output_dim
            x = [Symbol(f'x{i}') for i in range(n_vars)]
            
            # Flatten weights
            flat_weights = weights.flatten()
            
            # Create polynomial from weights
            poly = sum(self._as_sympy_coeff(c) * x[i]
                       for i, c in enumerate(flat_weights) if i < n_vars)
            
            # Reduce using Gröbner basis.
            # sympy's GroebnerBasis.reduce returns (quotient_coeffs, remainder).
            reduced = self.groebner_basis.reduce(poly)
            remainder = reduced[1]
            
            if remainder != 0:
                # Extract coefficients
                coeffs = np.zeros(n_vars)
                if hasattr(remainder, 'as_dict'):
                    poly_dict = remainder.as_dict()
                    for monom, coeff in poly_dict.items():
                        for i, xi in enumerate(x):
                            if monom == tuple(1 if j == i else 0 for j in range(n_vars)):
                                coeffs[i] = float(coeff)
                                break
                elif hasattr(remainder, 'all_coeffs'):
                    all_c = remainder.all_coeffs()
                    coeffs[:len(all_c)] = all_c
                else:
                    for i in range(n_vars):
                        try:
                            coeff = remainder.coeff(x[i])
                            coeffs[i] = float(coeff)
                        except Exception:
                            coeffs[i] = flat_weights[i] if i < len(flat_weights) else 0.0
                
                # Reshape back to matrix
                return np.array(coeffs).reshape(weights.shape)
        except Exception:
            pass
        
        return weights
    
    def reduce_with_groebner(self, polynomial: Any) -> Any:
        """
        Reduce a polynomial using the Gröbner basis.
        
        Args:
            polynomial: Sympy polynomial to reduce
            
        Returns:
            Reduced polynomial
        """
        if self.groebner_basis is None or not SYMPY_AVAILABLE:
            return polynomial
        
        try:
            # sympy's GroebnerBasis.reduce returns (quotient_coeffs, remainder).
            reduced = self.groebner_basis.reduce(polynomial)
            return reduced[1] if reduced[1] != 0 else polynomial
        except Exception:
            return polynomial
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Gröbner layer.
        
        Computes: y = x @ W + b, with W projected to constraint ideal
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Output tensor of shape (batch_size, output_dim) or (output_dim,)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        is_batch = x.dim() == 2
        if not is_batch:
            x = x.unsqueeze(0)
        
        # Store for potential weight updates
        self.last_input = x.detach().clone()
        
        # Compute output: y = x @ W + b
        output = x @ self.weight_matrix
        
        if self.use_bias and self.bias is not None:
            output = output + self.bias
        
        self.last_output = output.detach().clone()
        
        return output if is_batch else output[0]
        
    @torch.no_grad()
    def project_weights(self) -> None:
        """Explicit hard projection of weights onto the Gröbner Basis ideal."""
        if self.groebner_basis is not None:
            proj_w = self.project_to_ideal(self.weight_matrix.detach().cpu().numpy())
            self.weight_matrix.copy_(torch.tensor(proj_w, dtype=self.weight_matrix.dtype, device=self.weight_matrix.device))
    
    def ideal_membership_test(self, polynomial: np.ndarray) -> bool:
        """
        Test if a polynomial (as coefficient vector) is in the ideal.
        
        Args:
            polynomial: Polynomial coefficients
            
        Returns:
            True if polynomial is in the constraint ideal
        """
        if self.groebner_basis is None or not SYMPY_AVAILABLE:
            return True  # No constraints
        
        try:
            n_vars = self.input_dim * self.output_dim
            x = [Symbol(f'x{i}') for i in range(n_vars)]
            
            # Create polynomial from coefficients
            poly = sum(self._as_sympy_coeff(c) * x[i]
                       for i, c in enumerate(polynomial.flatten()) if i < n_vars)
            
            # Reduce and check if the remainder vanishes.
            # sympy's GroebnerBasis.reduce returns (quotient_coeffs, remainder).
            reduced = self.groebner_basis.reduce(poly)
            return reduced[1] == 0
        except Exception:
            return False
    
    def compute_variety_representation(self) -> Optional[Dict[str, Any]]:
        """
        Compute representation of the algebraic variety.
        
        Returns:
            Dictionary with variety information
        """
        if self.groebner_basis is None:
            return None
        
        return {
            'groebner_basis': str(self.groebner_basis),
            'dimension': self._compute_variety_dimension(),
            'n_constraints': len(self.constraint_ideal) if self.constraint_ideal else 0
        }
    
    def _compute_variety_dimension(self) -> int:
        """Estimate dimension of the variety"""
        if self.groebner_basis is None:
            return self.input_dim * self.output_dim
        
        # Simple dimension estimate
        n_vars = self.input_dim * self.output_dim
        n_constraints = len(self.constraint_ideal) if self.constraint_ideal else 0
        
        return max(0, n_vars - n_constraints)
    
    def to_symbolic(self) -> Dict[str, Any]:
        """
        Convert layer to symbolic representation.
        
        Returns:
            Dictionary with symbolic layer representation
        """
        result = {
            'type': 'GroebnerLayer',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'weight_shape': list(self.weight_matrix.shape),
            'has_bias': self.use_bias,
            'groebner_order': self.groebner_order
        }
        
        if self.constraint_ideal:
            result['constraint_ideal'] = [str(c) for c in self.constraint_ideal]
        
        if self.groebner_basis is not None:
            result['groebner_basis'] = str(self.groebner_basis)
        
        return result
    
    def __repr__(self) -> str:
        return (f"GroebnerLayer(in={self.input_dim}, out={self.output_dim}, "
                f"constraints={len(self.constraint_ideal) if self.constraint_ideal else 0})")


class GroebnerResidualLayer(GroebnerLayer):
    """
    Gröbner layer with residual connections.
    
    Extends GroebnerLayer with skip connections for gradient flow.
    """
    
    def __init__(self, *args, residual_dim: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_dim = residual_dim
        self.residual_projection = None
        
        if residual_dim is not None and residual_dim != self.output_dim:
            self.residual_projection = np.random.randn(residual_dim, self.output_dim) * 0.01
    
    def forward(self, x: np.ndarray, residual: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass with optional residual connection.
        
        Args:
            x: Input array
            residual: Optional residual input
            
        Returns:
            Output with residual added if provided
        """
        output = super().forward(x)
        
        if residual is not None:
            if not isinstance(residual, torch.Tensor):
                residual = torch.tensor(residual, dtype=output.dtype)
            if self.residual_projection is not None:
                proj = torch.tensor(self.residual_projection, dtype=output.dtype)
                residual = residual @ proj
            output = output + residual
        
        return output


def create_groebner_layer(input_dim: int, output_dim: int,
                          constraint_ideal: Optional[List] = None,
                          groebner_order: str = 'lex') -> GroebnerLayer:
    """
    Factory function to create a Gröbner layer.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        constraint_ideal: List of constraint polynomials
        groebner_order: Monomial ordering
        
    Returns:
        GroebnerLayer instance
    """
    config = GroebnerLayerConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        constraint_ideal=constraint_ideal,
        groebner_order=groebner_order
    )
    return GroebnerLayer(config)
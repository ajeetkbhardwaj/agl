"""
Algebraic Variety Neural Network Layer
======================================
A neural network layer constrained to an algebraic variety.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np
import torch
import torch.nn as nn
from ..alggeom.polynomial import Polynomial, Monomial
from ..alggeom.algvariety import AlgebraicVariety


class VarietyLayer(nn.Module):
    """
    A neural network layer constrained to an algebraic variety.
    
    The layer's weights lie on a specified algebraic variety, ensuring
    that learned representations satisfy geometric constraints.
    
    Forward pass: y = W @ x where W ∈ V (the variety)
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 variety: Optional[AlgebraicVariety] = None,
                 parameterization: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 ideal_generators: Optional[List] = None,
                 use_groebner_projection: bool = False):
        super().__init__()
        """
        Initialize variety-constrained layer.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output
            variety: Algebraic variety constraining the weights
            parameterization: Optional parameterization of the variety
            ideal_generators: Optional list of generators for the ideal
            use_groebner_projection: Whether to project weights using Groebner basis
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        if variety is None and ideal_generators is not None:
            self.variety = AlgebraicVariety(ideal_generators)
        elif variety is not None:
            self.variety = variety
        else:
            self.variety = AlgebraicVariety([])
            
        self.parameterization = parameterization
        self.use_groebner_projection = use_groebner_projection
        
        # Latent parameter dimension
        self.latent_dim = self._compute_latent_dim()
        
        # Initialize latent parameters
        self.latent_params = nn.Parameter(torch.randn(self.latent_dim) * 0.1)
        self.project_to_variety()
        
        # Cached computations
        self.last_input: Optional[torch.Tensor] = None
        self.last_output: Optional[torch.Tensor] = None
        
    def _compute_latent_dim(self) -> int:
        """Compute dimension of latent parameter space"""
        if self.parameterization is not None:
            # Infer from parameterization function signature or explicit argument
            return getattr(self, '_explicit_latent_dim', self.variety.dimension() if self.variety.dimension() > 0 else 1)
        else:
            return self.input_dim * self.output_dim

    def _compute_weight_matrix(self) -> torch.Tensor:
        """Compute weight matrix from latent parameters"""
        if self.parameterization is not None:
            # Use parameterization
            return self.parameterization(self.latent_params).view(
                self.output_dim, self.input_dim
            )
        else:
            # Direct reshape
            return self.latent_params.view(self.output_dim, self.input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through variety-constrained layer.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        is_batch = x.dim() == 2
        if not is_batch:
            x = x.unsqueeze(0)
        
        w_matrix = self._compute_weight_matrix()
        output = x @ w_matrix.T
        
        self.last_input = x.detach().clone()
        self.last_output = output.detach().clone()
        
        return output if is_batch else output[0]
    
    def project_to_variety(self, tol: float = 1e-6, max_iter: int = 100) -> None:
        """
        Project current weights onto the algebraic variety.
        
        This uses gradient descent on the variety defining equations, properly
        backpropagating through any parameterization using PyTorch Autograd.
        """
        device = self.latent_params.device
        
        for _ in range(max_iter):
            # 1. Compute weight matrix from latent params (track gradients!)
            z = self.latent_params.detach().requires_grad_(True)
            
            if self.parameterization is not None:
                w_mat = self.parameterization(z).view(self.output_dim, self.input_dim)
            else:
                w_mat = z.view(self.output_dim, self.input_dim)

            w_np = w_mat.detach().cpu().numpy().flatten()
            
            # 2. Compute constraint violations via AlgebraicCore
            violations = self.variety.constraint_violations(w_np)
            violation_norm = float(np.linalg.norm(violations))
            
            # Check convergence
            if violation_norm < tol:
                break
                
            # 3. Compute Lagrangian gradient w.r.t the evaluated weight matrix W
            grad_w_np = self.variety.constraint_lagrangian_gradient(w_np, violations)
            
            if grad_w_np is None or np.linalg.norm(grad_w_np) < 1e-10:
                break
                
            # 4. Use Autograd to compute VJP to get exact gradient w.r.t Latent Space Z
            grad_w_tensor = torch.tensor(grad_w_np, dtype=w_mat.dtype, device=device).view(self.output_dim, self.input_dim)
            w_mat.backward(grad_w_tensor)
            grad_z = z.grad
                
            # 5. Backtracking line search on Latent Space Z
            step_size = 1.0
            improved = False
            
            with torch.no_grad():
                while step_size > 1e-10:
                    z_trial = self.latent_params.detach() - step_size * grad_z
                    
                    if self.parameterization is not None:
                        w_trial = self.parameterization(z_trial).view(self.output_dim, self.input_dim)
                    else:
                        w_trial = z_trial.view(self.output_dim, self.input_dim)
                        
                    w_trial_np = w_trial.cpu().numpy().flatten()
                    violations_trial = self.variety.constraint_violations(w_trial_np)
                    
                    if np.linalg.norm(violations_trial) < violation_norm:
                        self.latent_params.copy_(z_trial)
                        improved = True
                        break
                    step_size *= 0.5
                
            if not improved:
                break
    
    def get_variety_info(self) -> Dict:
        """Get information about the variety constraint"""
        w_matrix = self._compute_weight_matrix().detach().cpu().numpy()
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'latent_dim': self.latent_dim,
            'variety_dimension': self.variety.dimension(),
            'variety_degree': self.variety.degree(),
            'num_defining_equations': len(self.variety.ideal),
            'weight_norm': float(np.linalg.norm(w_matrix))
        }
    
    def __repr__(self) -> str:
        return (f"VarietyLayer(in={self.input_dim}, out={self.output_dim}, "
                f"variety_dim={self.variety.dimension()})")


class ProductVarietyLayer(VarietyLayer):
    """
    A layer constrained to a product of algebraic varieties.
    
    This allows different parts of the weight matrix to be constrained
    to different varieties.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 row_varieties: List[AlgebraicVariety],
                 col_varieties: Optional[List[AlgebraicVariety]] = None):
        """
        Initialize product variety layer.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output
            row_varieties: Varieties constraining each output row
            col_varieties: Optional varieties constraining each input column
        """
        self.row_varieties = row_varieties
        self.col_varieties = col_varieties
        
        # Create combined variety (product)
        combined_variety = self._create_product_variety()
        
        super().__init__(input_dim, output_dim, combined_variety)
        
    def _shift_polynomial(self, poly: Any, offset: int) -> Any:
        """Shift variable indices in polynomial."""
        from ..alggeom.polynomial import Polynomial, Monomial, Variable
        if not isinstance(poly, Polynomial):
            return poly
        terms = {}
        for monom, coeff in poly.terms.items():
            new_vars = []
            for var, exp in monom.variables:
                name = var.name
                if name.startswith('x'):
                    try:
                        idx = int(name[1:])
                        name = f'x{idx + offset}'
                    except ValueError:
                        pass
                new_vars.append((Variable(name), exp))
            new_monom = Monomial(tuple(sorted(new_vars, key=lambda t: t[0].name)))
            terms[new_monom] = coeff
        return Polynomial(terms)

    def _create_product_variety(self) -> AlgebraicVariety:
        """Create product variety V₁ × V₂ × ... × Vₖ."""
        all_generators = []
        
        offset = 0
        for variety in self.row_varieties:
            for gen in variety.ideal:
                shifted_gen = self._shift_polynomial(gen, offset)
                all_generators.append(shifted_gen)
            offset += variety.dimension() if variety.dimension() > 0 else 1
            
        if self.col_varieties:
            for variety in self.col_varieties:
                for gen in variety.ideal:
                    shifted_gen = self._shift_polynomial(gen, offset)
                    all_generators.append(shifted_gen)
                offset += variety.dimension() if variety.dimension() > 0 else 1
                
        return AlgebraicVariety(all_generators)
    
    def get_row_constraint_info(self) -> List[Dict]:
        """Get constraint info for each output row"""
        return [
            {
                'num_generators': len(v.ideal),
                'dimension': v.dimension(),
            }
            for v in self.row_varieties
        ]


class TangentSpaceLayer(VarietyLayer):
    """
    A layer that operates in the tangent space of an algebraic variety.
    
    This allows learning directions that are tangent to the variety,
    which can be useful for optimization on manifolds.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 variety: AlgebraicVariety,
                 base_point: Optional[np.ndarray] = None):
        """
        Initialize tangent space layer.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output
            variety: Algebraic variety
            base_point: Point on variety for tangent space
        """
        self.base_point = base_point
        
        super().__init__(input_dim, output_dim, variety)
        
        # Compute tangent space at base point
        self.tangent_space_basis = self._compute_tangent_space()
        
    def _compute_tangent_space(self) -> np.ndarray:
        """Compute orthonormal basis for tangent space at base point."""
        if self.base_point is None:
            return np.eye(self.input_dim * self.output_dim)

        n_vars = self.input_dim * self.output_dim
        eps = 1e-7
        flat_base = self.base_point.flatten()

        v0 = self.variety.constraint_violations(flat_base)
        if len(v0) == 0:
            return np.eye(n_vars)

        J = np.zeros((len(v0), n_vars))
        for j in range(n_vars):
            p_plus = flat_base.copy()
            p_plus[j] += eps
            J[:, j] = (self.variety.constraint_violations(p_plus) - v0) / eps

        _, s, Vh = np.linalg.svd(J)
        tolerance = max(J.shape) * np.spacing(max(s)) if len(s) > 0 else 1e-10
        rank = np.sum(s > tolerance)

        tangent_basis = Vh[rank:].T

        if tangent_basis.size == 0:
            return np.eye(n_vars)
        return tangent_basis
    
    def project_to_tangent(self, v: np.ndarray) -> np.ndarray:
        """
        Project a vector onto the tangent space.
        
        Args:
            v: Vector to project
        
        Returns:
            Projected vector in tangent space
        """
        B = self.tangent_space_basis
        flat_v = v.flatten()
        if B.shape[1] == len(flat_v):
            return B @ (B.T @ flat_v)
        # Fallback if shapes don't match for some reason
        return v
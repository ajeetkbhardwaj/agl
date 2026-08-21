import torch
import torch.nn as nn
from typing import Optional

from .orthopoly_layer import OrthoPolyLayer


class RationalPolyLayer(nn.Module):
    """
    Rational Polynomial Layer (Padé Approximant style).
    
    Computes the quotient of two orthogonal polynomial expansions:
        R(x) = P(x) / (1.0 + |Q(x)|)
        
    This gives the layer strict local support, allowing it to model
    sharp, localized spikes (like turbulent eddies) without suffering 
    from the global oscillatory artifacts known as Runge's Phenomenon.
    """
    
    def __init__(self, input_dim: int, output_dim: int, max_degree: int = 3, 
                 rank: int = 8, basis_type: str = 'chebyshev_T'):
        super().__init__()
        
        # Numerator Polynomial P(x)
        self.numerator = OrthoPolyLayer(
            input_dim=input_dim, output_dim=output_dim, 
            max_degree=max_degree, rank=rank, basis_type=basis_type
        )
        
        # Denominator Polynomial Q(x)
        self.denominator = OrthoPolyLayer(
            input_dim=input_dim, output_dim=output_dim, 
            max_degree=max_degree, rank=rank, basis_type=basis_type
        )
        
        # Initialize denominator weights very close to zero.
        # This ensures the layer starts as a pure polynomial and prevents
        # immediate vanishing gradients caused by large initial denominator poles.
        with torch.no_grad():
            self.denominator.linear_weight.mul_(0.01)
            self.denominator.cheby_coeffs.mul_(0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_x = self.numerator(x)
        q_x = self.denominator(x)
        
        # 1.0 + |Q(x)| ensures strict positivity and bounds the poles,
        # acting as a dynamic algebraic damper for high frequencies.
        return p_x / (1.0 + torch.abs(q_x))
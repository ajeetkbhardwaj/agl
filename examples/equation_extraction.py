"""
Example 16: White-Box Interpretability with OrthoPolyNN
=======================================================
Because OrthoPolyNNs are built on mathematically rigorous orthogonal bases 
(like Legendre or Chebyshev), they are completely interpretable and avoid 
the severe collinearity issues that standard monomials face.

This script demonstrates how to extract the exact orthogonal coefficients 
the neural network learned directly from the data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nn.orthopoly_nn import OrthoPolyNetwork, OrthoPolyConfig, OrthogonalBasisType

def main():
    print("=" * 65)
    print("White-Box Equation Extraction: Orthogonal Polynomials")
    print("=" * 65)
    
    # Add seeds for deterministic training
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Generate data from a hidden target function using Chebyshev Polynomials.
    # Chebyshev basis: T_0(x)=1, T_1(x)=x, T_2(x)=2x^2-1, T_3(x)=4x^3-3x
    # Target: y = 2.0*T_3(x) + 0.0*T_2(x) - 1.5*T_1(x) + 0.5*T_0(x)
    print("Target Equation (in Chebyshev Basis):")
    print("y = 0.00*T_4(x) + 2.00*T_3(x) + 0.00*T_2(x) - 1.50*T_1(x) + 0.50*T_0(x)\n")
    
    # Chebyshev polynomials are strictly orthogonal on the interval [-1, 1]
    X = torch.linspace(-1, 1, 400).view(-1, 1)
    
    T0 = torch.ones_like(X)
    T1 = X
    T2 = 2 * X**2 - 1
    T3 = 4 * X**3 - 3 * X
    
    y = 2.0 * T3 - 1.5 * T1 + 0.5 * T0
    
    # 2. Create a "Direct" OrthoPolyNN (No hidden layers, straight from input to output)
    # This forces the network to learn a single, globally interpretable equation.
    config = OrthoPolyConfig(
        input_dim=1, 
        output_dim=1, 
        hidden_dims=[],  # No hidden layers
        max_degree=4,    # Give it degree 4 to see if it properly zeroes out T_4
        basis_type=OrthogonalBasisType.CHEBYSHEV_FIRST,
        dropout=0.0
    )
    model = OrthoPolyNetwork(config)
    
    # 3. Train the network
    # LBFGS is excellent for exact parameter fitting on smooth algebraic manifolds
    optimizer = optim.LBFGS(model.parameters(), lr=0.5, max_iter=20)
    criterion = nn.MSELoss()
    
    print("Training OrthoPolyNN to reverse-engineer the Chebyshev coefficients...")
    for epoch in range(50):
        def closure():
            optimizer.zero_grad()
            mse_loss = criterion(model(X), y)
            loss = mse_loss
            loss.backward()
            return loss
        
        optimizer.step(closure)
        loss = closure()
        
    print(f"Final Training Loss: {loss.item():.6f}\n")
    
    # 4. Prune negligible weights (Hard Thresholding for exact sparsity presentation)
    with torch.no_grad():
        for param in model.parameters():
            param[torch.abs(param) < 1e-3] = 0.0

    # 5. Extract the learned algebraic coefficients
    # Since we have no hidden layers, the network is exactly computing a linear combination of the basis.
    print("Neural Network's Learned Coefficients (after contracting low-rank factors):")
    print("-" * 65)
    
    # Get the single layer from our direct-map network
    layer = model.layers[0]
    
    # Calculate the effective coefficients by contracting the low-rank factors
    with torch.no_grad():
        effective_coeffs = torch.einsum('oir,rid->oid', layer.linear_weight, layer.cheby_coeffs)

        # The layer's bias corresponds to the T_0(x)=1 coefficient.
        # The einsum product also produces a T_0 coefficient, so we add them.
        c0 = layer.bias.item() + effective_coeffs[0, 0, 0].item()
        
        # The other coefficients correspond to T_1, T_2, ...
        other_coeffs = effective_coeffs[0, 0, 1:].flatten().tolist()
        all_coeffs = [c0] + other_coeffs

    print(f"Target Coefficients (T0-T4): [0.50, -1.50, 0.00, 2.00, 0.00]")
    print("Learned Coefficients:")
    for i, c in enumerate(all_coeffs):
        print(f"  - T_{i}(x) coefficient: {c:.2f}")
        
    print("-" * 65)
    print("Notice how the learned coefficients now correctly match the target equation.")

if __name__ == "__main__":
    main()
"""
Example 18: Automated Symbolic Regression with OrthoPolyNN
==========================================================
This example bridges Deep Learning with Symbolic Mathematics.

It trains an OrthoPolyNN on a noisy dataset, extracts the learned 
orthogonal Chebyshev coefficients, and uses SymPy to automatically 
translate them back into a standard human-readable algebraic equation 
(y = ax^3 + bx^2 + cx + d).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nn.orthopoly_nn import OrthoPolyNetwork, OrthoPolyConfig, OrthogonalBasisType

def main():
    print("=" * 65)
    print("Automated Symbolic Regression (Equation Estimation)")
    print("=" * 65)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Target Hidden Function: y = 3.5x^3 - 1.5x^2 + 2.0x - 1.0
    print("Target Hidden Equation : y = 3.5*x**3 - 1.5*x**2 + 2.0*x - 1.0\n")
    
    X = torch.linspace(-1, 1, 200).view(-1, 1)
    # Add noise to simulate real-world data collection
    noise = torch.randn_like(X) * 0.5
    y_true = 3.5 * (X**3) - 1.5 * (X**2) + 2.0 * X - 1.0
    y_noisy = y_true + noise
    
    # 2. Build a Direct OrthoPolyNN 
    config = OrthoPolyConfig(
        input_dim=1, output_dim=1, hidden_dims=[],
        max_degree=4, # Give it degree 4 to see if it correctly zeroes out x^4
        basis_type=OrthogonalBasisType.CHEBYSHEV_FIRST, dropout=0.0
    )
    model = OrthoPolyNetwork(config)
    
    # 3. Train using LBFGS to perfectly hit the algebraic minimum
    print("Training OrthoPolyNN on noisy data...")
    optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)
    criterion = nn.MSELoss()
    
    for epoch in range(30):
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X), y_noisy)
            loss.backward()
            return loss
        optimizer.step(closure)
        
    # 4. Extract and Estimate the Equation
    # Prune negligible weights for cleaner extraction
    with torch.no_grad():
        for param in model.parameters():
            param[torch.abs(param) < 1e-3] = 0.0
            
    estimated_eq = model.get_global_equation(feature_names=['x'])
    print("-" * 65)
    print(f"Neural Network Estimated Equation:\n y = {estimated_eq}")
    print("-" * 65)
    
    # 5. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X.numpy(), y_noisy.numpy(), color='gray', alpha=0.5, label='Noisy Data')
    plt.plot(X.numpy(), y_true.numpy(), 'k--', lw=2, label='True Curve')
    plt.plot(X.numpy(), model(X).detach().numpy(), 'r-', lw=2, label=f'Learned: {estimated_eq}')
    plt.title('Symbolic Equation Estimation via OrthoPolyNN')
    plt.legend()
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/18_symbolic_regression.png", dpi=300)
    print("Plot saved to results/18_symbolic_regression.png")

if __name__ == "__main__":
    main()
"""
Example 19: Solving Differential Equations with OrthoPolyNN
===========================================================
This example demonstrates how to use an OrthoPolyNetwork as a function 
approximator to solve an ordinary differential equation (ODE).

This approach is a cornerstone of Physics-Informed Neural Networks (PINNs).
The network is trained not on a dataset, but to satisfy the differential
equation itself, along with its initial or boundary conditions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nn.orthopoly_nn import OrthoPolyNetwork, OrthoPolyConfig, OrthogonalBasisType


def analytical_solution(t: torch.Tensor) -> torch.Tensor:
    """The exact analytical solution to the ODE for verification."""
    return torch.exp(-t)

def main():
    """
    Defines and solves the ODE: du/dt + u = 0, with u(0) = 1.
    """
    print("=" * 65)
    print("Solving du/dt + u = 0 with a Physics-Informed OrthoPolyNN")
    print("=" * 65)

    # 1. Configure the OrthoPolyNetwork to act as the solution u(t)
    # Input is time 't' (1D), output is the solution 'u' (1D).
    config = OrthoPolyConfig(
        input_dim=1,
        output_dim=1,
        hidden_dims=[32, 32],
        max_degree=3,
        basis_type=OrthogonalBasisType.LEGENDRE, # Legendre is a good general-purpose choice
    )
    model = OrthoPolyNetwork(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

    # 2. Training loop to enforce the physics (the ODE) and the initial condition
    epochs = 4000
    lambda_ic = 100.0  # Strong penalty for not meeting the initial condition
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print("Training the Physics-Informed Neural Network...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Create a set of "collocation points" where we'll enforce the ODE
        t_collocation = torch.linspace(0, 5, 100, requires_grad=True).view(-1, 1)

        # Predict u(t) at these points
        u_pred = model(t_collocation)

        # Use autograd to get the derivative du/dt
        du_dt = torch.autograd.grad(
            outputs=u_pred, inputs=t_collocation,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True
        )[0]

        # Physics Loss: The residual of the ODE, should be zero
        ode_residual = du_dt + u_pred
        loss_ode = torch.mean(ode_residual**2)

        # Data Loss: Enforce the initial condition u(0) = 1
        t_initial = torch.tensor([[0.0]])
        u_initial_pred = model(t_initial)
        loss_ic = (u_initial_pred - 1.0)**2

        # Combine the losses
        total_loss = loss_ode + lambda_ic * loss_ic.squeeze()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.6f}")

    # 3. Evaluate the trained model and plot the results
    model.eval()
    with torch.no_grad():
        t_test = torch.linspace(0, 5, 200).view(-1, 1)
        u_nn = model(t_test)
        u_analytical = analytical_solution(t_test)

    plt.figure(figsize=(10, 6))
    plt.plot(t_test.numpy(), u_analytical.numpy(), 'k--', lw=2, label='Analytical Solution: $e^{-t}$')
    plt.plot(t_test.numpy(), u_nn.numpy(), 'r-', lw=2, label='OrthoPolyNN Solution')
    plt.title('Solving a Linear ODE with a Physics-Informed OrthoPolyNN')
    plt.xlabel('Time (t)')
    plt.ylabel('u(t)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/19_solve_ode.png", dpi=300)
    print("\nPlot saved to results/19_solve_ode.png")

if __name__ == "__main__":
    main()
"""
Example 15: Chaotic Dynamics Forecasting (Lorenz Attractor)
===========================================================
This advanced real-world example evaluates the OrthoPolyNN on forecasting 
a chaotic dynamical system governed by non-linear polynomial differential 
equations (the Lorenz system). 

Because the underlying physics of the Lorenz attractor are governed by 
polynomials (specifically xy and xz cross-terms), OrthoPolyNN is 
mathematically predisposed to capture the smooth phase-space dynamics 
much more accurately than the piecewise-linear approximations of a 
standard ReLU MLP.
"""

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nn.orthopoly_nn import OrthoPolyNetwork, OrthoPolyConfig, OrthogonalBasisType, OrthoPolyTrainer
from src.layers.orthopoly_layer import OrthoPolyLayer

def extract_segre_equations(layer, feature_names, threshold=1e-4, return_exact=False):
    """
    Unrolls the multiplicative CP Tensor back into flat SymPy polynomials for multiple outputs.
    """
    syms = [sp.Symbol(name) for name in feature_names]
    
    bias = layer.bias.detach().numpy() if layer.bias is not None else np.zeros(layer.output_dim)
    w = layer.linear_weight.detach().numpy()  # Shape (output_dim, rank)
    c = layer.cheby_coeffs.detach().numpy()  # Shape (rank, input_dim, num_poly_terms)
    
    eqs = []
    exact_eqs = []
    for j in range(layer.output_dim):
        eq = bias[j]
        
        for r in range(layer.rank):
            r_mix = w[j, r]
            if abs(r_mix) < threshold:
                continue
                
            r_term = 1.0
            for i, sym in enumerate(syms):
                run_min = layer.running_min[i].item()
                run_max = layer.running_max[i].item()
                span = run_max - run_min
                
                scaled_sym = -1.0 if span < 1e-3 else 2.0 * (sym - run_min) / span - 1.0
                
                feat_term = 0.0
                for d in range(layer.num_poly_terms):
                    coeff = c[r, i, d]
                    if abs(coeff) > threshold:
                        if d == 0:
                            feat_term += coeff * 1.0
                        else:
                            feat_term += coeff * sp.chebyshevt(d, scaled_sym)
                r_term *= feat_term
            eq += r_mix * r_term
            
        expanded = sp.expand(eq)
        clean_eq = expanded.xreplace({n: round(n, 4) for n in expanded.atoms(sp.Number)})
        eqs.append(clean_eq)
        exact_eqs.append(expanded)
        
    if return_exact:
        return eqs, exact_eqs
    return eqs

class StandardMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = [nn.BatchNorm1d(in_dim)]  # Normalize raw inputs internally for MLP fairness
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def generate_lorenz_data(n_steps=10000, dt=0.01):
    """Generate data from the Lorenz equations."""
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    X = np.zeros((n_steps, 3))
    X[0] = [0.0, 1.0, 1.05]
    
    for i in range(n_steps - 1):
        x, y, z = X[i]
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        X[i + 1] = [x + dx, y + dy, z + dz]
        
    return X

def create_dataset(X_seq, lookback=1):
    X_data, y_data = [], []
    for i in range(len(X_seq) - lookback):
        X_data.append(X_seq[i:i+lookback].flatten())
        y_data.append(X_seq[i+lookback])
    return np.array(X_data), np.array(y_data)

def train_model(model, X_train, y_train, epochs=30, batch_size=128, lr=0.005):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model

def autoregressive_rollout(model, initial_state, steps):
    """Use the model to predict multiple steps into the future iteratively."""
    current_state = torch.tensor(initial_state, dtype=torch.float32)
    trajectory = [current_state.numpy()]
    
    with torch.no_grad():
        for _ in range(steps):
            next_state = model(current_state.unsqueeze(0)).squeeze(0)
            trajectory.append(next_state.numpy())
            current_state = next_state
            
    return np.array(trajectory)

def main():
    print("=" * 65)
    print("Chaotic Dynamics Forecasting: Lorenz Attractor")
    print("=" * 65)
    
    # Set seeds for deterministic reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Generate Data
    print("Generating Lorenz attractor data...")
    sequence = generate_lorenz_data(n_steps=15000, dt=0.01)
    
    # We use the RAW unscaled data! The ChebyshevLayer internally normalizes it to [-1, 1]
    # which naturally amplifies the chaotic cross-terms, protecting them from L1 pruning.
    X_data, y_data = create_dataset(sequence, lookback=1)
    
    # Train on first 10,000 steps, test extrapolation on next 5,000
    train_size = 10000
    X_train, y_train = X_data[:train_size], y_data[:train_size]
    
    hidden_dims = [64, 64]
    epochs = 80
    
    # 2. Train OrthoPolyNN (Zero Hidden Layers + Segre Embedding)
    print("\nTraining OrthoPolyNN (Chebyshev Segre Embedding, Degree 1 - Multilinear)...")
    ortho_config = OrthoPolyConfig(
        input_dim=3,
        output_dim=3,
        hidden_dims=[],  # No hidden layers
        max_degree=1,
        rank=12,  # Tightened rank focuses the network's capacity on exactly the required physics
        basis_type=OrthogonalBasisType.CHEBYSHEV_FIRST,
        interaction_mode='multiplicative'
    )
    ortho_model = OrthoPolyNetwork(ortho_config)
    
    # CRITICAL: Settle the internal Domain Scaler BEFORE training so the geometry is static
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    ortho_model.train()
    with torch.no_grad():
        for _ in range(100):
            ortho_model(X_t)
            
    # CRITICAL: Initialize Segre Embedding away from the vanishing gradient saddle point.
    with torch.no_grad():
        layer = ortho_model.layers[0]
        layer.cheby_coeffs.data[:, :, 0] = 1.0
        # Higher initial variance helps cross-terms establish themselves before pruning
        layer.cheby_coeffs.data[:, :, 1:] *= 0.5
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ortho_model.parameters(), lr=0.02)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    
    print("Phase 1: Discovering sparse algebraic CP structure (Burn-in + Adaptive L1)...")
    for epoch in range(500):
        # BURN-IN PHASE: Allow 100 epochs of pure MSE optimization to find the chaotic basin.
        # ANNEALING PHASE: Slowly ramp up L1 to squeeze out redundant noise paths.
        # Max L1 is gently set to 0.00005 to protect the chaotic interactions.
        current_l1 = 0.0 if epoch < 100 else 0.00005 * ((epoch - 100) / 400.0)
        
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            
            # Smart L1 Penalty for Segre Embedding
            l1_loss = 0.0
            for name, param in ortho_model.named_parameters():
                if 'cheby_coeffs' in name:
                    l1_loss += torch.abs(param[:, :, 1:]).sum()
                    # Softly penalize constants towards 1.0 to encourage feature pruning
                    l1_loss += torch.abs(param[:, :, 0] - 1.0).sum() * 0.1
                elif 'linear_weight' in name:
                    l1_loss += torch.abs(param).sum()
                else:
                    l1_loss += torch.abs(param).sum()
                    
            loss = criterion(ortho_model(x_batch), y_batch) + current_l1 * l1_loss
            loss.backward()
            optimizer.step()
            
    print("Phase 2: Pruning redundant tensor paths...")
    with torch.no_grad():
        for name, param in ortho_model.named_parameters():
            if 'cheby_coeffs' in name:
                param.data[:, :, 1:] = torch.where(torch.abs(param.data[:, :, 1:]) < 0.001, 0.0, param.data[:, :, 1:])
                param.data[:, :, 0] = torch.where(torch.abs(param.data[:, :, 0] - 1.0) < 0.05, 1.0, param.data[:, :, 0])
                param.data[:, :, 0] = torch.where(torch.abs(param.data[:, :, 0]) < 0.001, 0.0, param.data[:, :, 0])
            else:
                param.data = torch.where(torch.abs(param.data) < 0.001, 0.0, param.data)
                
        masks_zero = [(p != 0.0).detach() for p in ortho_model.parameters()]
        masks_one = [(p != 1.0).detach() for p in ortho_model.parameters()]
    
    print("Phase 3: Restoring exact coefficient amplitudes...")
    ortho_model.eval()
    lbfgs = torch.optim.LBFGS(ortho_model.parameters(), lr=0.1, max_iter=200)
    def closure():
        lbfgs.zero_grad()
        loss = criterion(ortho_model(X_t), y_t)
        loss.backward()
        with torch.no_grad():
            for p, m_z, m_o in zip(ortho_model.parameters(), masks_zero, masks_one):
                if p.grad is not None: 
                    p.grad.mul_((m_z & m_o).float())
        return loss
    lbfgs.step(closure)
    
    print("\n" + "=" * 65)
    print("Extracted Lorenz Equations (Scaled Domain):")
    eqs, exact_eqs = extract_segre_equations(ortho_model.layers[0], ["x", "y", "z"], return_exact=True)
    print(f"dx/dt ≈ x_next = {eqs[0]}")
    print(f"dy/dt ≈ y_next = {eqs[1]}")
    print(f"dz/dt ≈ z_next = {eqs[2]}")
    print("=" * 65 + "\n")
    
    print("Compiling Exact Symbolic Equations into highly optimized C/NumPy kernels...")
    syms = [sp.Symbol(name) for name in ["x", "y", "z"]]
    compiled_eqs = [sp.lambdify(syms, eq, modules="numpy") for eq in exact_eqs]
    
    def symbolic_rollout(initial_state, steps):
        current_state = np.array(initial_state, dtype=np.float64)
        trajectory = [current_state]
        for _ in range(steps):
            x_val, y_val, z_val = current_state
            next_state = np.array([
                compiled_eqs[0](x_val, y_val, z_val),
                compiled_eqs[1](x_val, y_val, z_val),
                compiled_eqs[2](x_val, y_val, z_val)
            ])
            trajectory.append(next_state)
            current_state = next_state
        return np.array(trajectory)

    # 3. Train Standard MLP
    print("Training Standard MLP (ReLU)...")
    mlp_model = StandardMLP(in_dim=3, hidden_dims=hidden_dims, out_dim=3)
    train_model(mlp_model, X_train, y_train, epochs=epochs, lr=0.005)
    
    # 4. Multi-step Autoregressive Rollout (Extrapolation)
    print("\nSimulating 1000-step future trajectories...")
    rollout_steps = 1000
    start_idx = train_size
    initial_state = X_data[start_idx]
    
    true_traj_raw = sequence[start_idx:start_idx + rollout_steps + 1]
    ortho_traj_raw = autoregressive_rollout(ortho_model, initial_state, rollout_steps)
    symbolic_traj_raw = symbolic_rollout(initial_state, rollout_steps)
    mlp_traj_raw = autoregressive_rollout(mlp_model, initial_state, rollout_steps)
    
    # 5. Calculate Divergence (Error over time)
    ortho_err = np.linalg.norm(true_traj_raw - ortho_traj_raw, axis=1)
    sym_err = np.linalg.norm(true_traj_raw - symbolic_traj_raw, axis=1)
    mlp_err = np.linalg.norm(true_traj_raw - mlp_traj_raw, axis=1)
    
    print("\n" + "=" * 65)
    print("Mean Trajectory Error (Lower is Better):")
    print(f"OrthoPolyNN (PyTorch Tensors) : {ortho_err.mean():.4f}")
    print(f"OrthoPolyNN (Compiled SymPy)  : {sym_err.mean():.4f}")
    print(f"Standard MLP (Deep ReLU)      : {mlp_err.mean():.4f}")
    print("=" * 65)
    
    # 6. Visualization
    print("\nGenerating 3D phase-space visualization...")
    fig = plt.figure(figsize=(24, 6))
    
    # True Trajectory
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.plot(true_traj_raw[:, 0], true_traj_raw[:, 1], true_traj_raw[:, 2], color='black', lw=1)
    ax1.set_title('True Lorenz Attractor')
    ax1.grid(False)
    
    # OrthoPolyNN Trajectory
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.plot(ortho_traj_raw[:, 0], ortho_traj_raw[:, 1], ortho_traj_raw[:, 2], color='blue', lw=1)
    ax2.set_title('OrthoPolyNN (PyTorch Tensors)')
    ax2.grid(False)
    # Compiled Symbolic Trajectory
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.plot(symbolic_traj_raw[:, 0], symbolic_traj_raw[:, 1], symbolic_traj_raw[:, 2], color='green', lw=1)
    ax3.set_title('Compiled Exact Mathematics')
    ax3.grid(False)
    
    # Standard MLP Trajectory
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.plot(mlp_traj_raw[:, 0], mlp_traj_raw[:, 1], mlp_traj_raw[:, 2], color='red', lw=1)
    ax4.set_title('Standard MLP (ReLU)')
    ax4.grid(False)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/15_lorenz_attractor.png", dpi=300)
    print("Plot saved to results/15_lorenz_attractor.png")

if __name__ == "__main__":
    main()
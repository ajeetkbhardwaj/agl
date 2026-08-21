"""
Example 14: Real-World Regression with OrthoPolyNN
==================================================
This example demonstrates the OrthoPolyNN architecture on the California 
Housing dataset. It highlights how orthogonal polynomial bases (Chebyshev)
provide stable gradients and high expressivity for continuous regression tasks,
while completely avoiding the NaN explosions typical of standard polynomials.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# Ensure imports resolve correctly from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nn.orthopoly_nn import OrthoPolyNetwork, OrthoPolyConfig, OrthogonalBasisType, OrthoPolyTrainer


class StandardMLP(nn.Module):
    """A standard Multi-Layer Perceptron baseline."""
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


def train_standard_mlp(model, X_train, y_train, epochs=50, batch_size=128, lr=0.01):
    """Standard PyTorch training loop for the baseline MLP."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                             torch.tensor(y_train, dtype=torch.float32))
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


def main():
    print("=" * 65)
    print("California Housing Regression: OrthoPolyNN vs Standard MLP")
    print("=" * 65)
    
    # 1. Data Preparation
    print("Loading and scaling dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target.reshape(-1, 1)
    
    # Using the full dataset for maximum data density
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Explicitly map inputs to the perfect domain for Chebyshev polynomials [-1, 1]
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_dim = X.shape[1]
    hidden_dims = [128, 64]
    epochs = 100
    
    # 2. Train OrthoPolyNN
    print(f"\nTraining OrthoPolyNN (Chebyshev First Kind, Max Degree 2, Epochs {epochs})...")
    config = OrthoPolyConfig(
        input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims,
        max_degree=2, basis_type=OrthogonalBasisType.CHEBYSHEV_FIRST, dropout=0.0
    )
    ortho_model = OrthoPolyNetwork(config)
    
    # Use the advanced trainer featuring spectral regularization
    trainer = OrthoPolyTrainer(ortho_model, learning_rate=0.005, spectral_reg=1e-4)
    trainer.train(X_train, y_train, epochs=epochs, batch_size=128, verbose=False)
    ortho_preds = ortho_model.predict(X_test)
    
    # 3. Train Standard MLP
    print(f"Training Standard MLP (ReLU Activations, Epochs {epochs})...")
    mlp = StandardMLP(in_dim=input_dim, hidden_dims=hidden_dims, out_dim=1)
    train_standard_mlp(mlp, X_train, y_train, epochs=epochs, batch_size=128, lr=0.01)
    with torch.no_grad():
        mlp_preds = mlp(torch.tensor(X_test, dtype=torch.float32)).numpy()
    
    # 4. Evaluation
    print("\n" + "=" * 65)
    print("Results on Test Set (Scaled Targets)")
    print("=" * 65)
    print(f"OrthoPolyNN R2 Score: {r2_score(y_test, ortho_preds):.4f} (Higher is Better)")
    print(f"OrthoPolyNN MSE:      {mean_squared_error(y_test, ortho_preds):.4f} (Lower is Better)")
    print("-" * 65)
    print(f"Standard MLP R2 Score:{r2_score(y_test, mlp_preds):.4f}")
    print(f"Standard MLP MSE:     {mean_squared_error(y_test, mlp_preds):.4f}")
    print("=" * 65)
    
    # 5. Visualization
    print("\nGenerating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for OrthoPolyNN
    ax1.scatter(y_test, ortho_preds, alpha=0.3, color='blue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_title(f'OrthoPolyNN\n$R^2 = {r2_score(y_test, ortho_preds):.4f}$')
    ax1.set_xlabel('True Values (Scaled)')
    ax1.set_ylabel('Predictions (Scaled)')
    
    # Plot for Standard MLP
    ax2.scatter(y_test, mlp_preds, alpha=0.3, color='green')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_title(f'Standard MLP\n$R^2 = {r2_score(y_test, mlp_preds):.4f}$')
    ax2.set_xlabel('True Values (Scaled)')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/14_california_housing.png", dpi=300)
    print("Plot saved to results/14_california_housing.png")

if __name__ == "__main__":
    main()
"""
Orthogonal Polynomial Neural Network (OrthoPolyNN)
==================================================
Main architecture for spectral neural networks using orthogonal polynomial bases.
Solves the combinatorial parameter explosion and Lipschitz gradient instability 
by operating in bounded spectral spaces (Chebyshev).
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .mixins import AlgebraicExtractionMixin
from ..layers.orthopoly_layer import OrthoPolyLayer


class OrthogonalBasisType(Enum):
    """Supported Orthogonal Polynomial Bases"""
    CHEBYSHEV_FIRST = "chebyshev_first"
    CHEBYSHEV_SECOND = "chebyshev_second"
    LEGENDRE = "legendre"
    HERMITE = "hermite"
    LAGUERRE = "laguerre"


@dataclass
class OrthoPolyConfig:
    """Configuration for Orthogonal Polynomial Neural Network"""
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [32])
    max_degree: int = 3
    rank: int = 4
    basis_type: OrthogonalBasisType = OrthogonalBasisType.CHEBYSHEV_FIRST
    interaction_mode: str = 'additive'
    activation: str = 'tanh'  # Tanh keeps domain in [-1, 1] for orthogonal stability
    dropout: float = 0.0
    use_bias: bool = True
    learning_rate: float = 0.01


class OrthoPolyNetwork(nn.Module, AlgebraicExtractionMixin):
    """
    A neural network that uses Spectral/Orthogonal Polynomial expansions.
    
    Unlike standard PNNs, OrthoPolyNN avoids parameter explosion and gradient 
    crashes (NaNs) by factorizing multivariate features and bounded basis evaluations.
    """
    
    def __init__(self, config: Optional[OrthoPolyConfig] = None, **kwargs):
        """
        Initialize the Orthogonal Polynomial Neural Network.
        
        Args:
            config: OrthoPolyConfig configuration
            **kwargs: Configuration arguments if config is None
        """
        super().__init__()
        
        if config is None:
            # Parse basis type from string if passed directly in kwargs
            if 'basis_type' in kwargs and isinstance(kwargs['basis_type'], str):
                kwargs['basis_type'] = OrthogonalBasisType[kwargs['basis_type'].upper()]
            self.config = OrthoPolyConfig(**kwargs)
        else:
            self.config = config
            
        self.activation_fn = self._get_activation_fn(self.config.activation)
        self.layers = nn.ModuleList()
        self._build_network()
        
        # Training state
        self.best_loss = float('inf')
        self.best_weights: Optional[Dict] = None
        
    def _get_activation_fn(self, name: str) -> nn.Module:
        import torch.nn.functional as F
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'silu': nn.SiLU(),
            'none': nn.Identity(),
        }
        return activations.get(name.lower(), nn.Tanh())
    
    @property
    def input_dim(self) -> int:
        return self.config.input_dim
    
    @property
    def output_dim(self) -> int:
        return self.config.output_dim
    
    def _build_network(self) -> None:
        """Build the network architecture sequence of Orthogonal Layers"""
        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
        num_hidden_layers = len(self.config.hidden_dims)

        basis_name_map = {
            OrthogonalBasisType.CHEBYSHEV_FIRST: "chebyshev_T",
            OrthogonalBasisType.CHEBYSHEV_SECOND: "chebyshev_U",
            OrthogonalBasisType.LEGENDRE: "legendre",
            OrthogonalBasisType.HERMITE: "hermite",
            OrthogonalBasisType.LAGUERRE: "laguerre",
        }
        basis_str = basis_name_map.get(self.config.basis_type, "chebyshev_T")
        
        for i in range(num_hidden_layers + 1):
            is_last_layer = (i == num_hidden_layers)
            layer_input_dim = dims[i]
            layer_output_dim = dims[i+1]

            layer = OrthoPolyLayer(
                input_dim=layer_input_dim,
                output_dim=layer_output_dim,
                max_degree=self.config.max_degree,
                rank=self.config.rank,
                use_bias=self.config.use_bias,
                basis_type=basis_str,
                interaction_mode=self.config.interaction_mode,
            )
            self.layers.append(layer)
            
            if not is_last_layer:
                self.layers.append(self.activation_fn)
                if self.config.dropout > 0:
                    self.layers.append(nn.Dropout(self.config.dropout))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Perform a single training step."""
        self.train()
        optimizer.zero_grad()
        y_pred = self(X)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return loss.item()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            learning_rate: float = 0.001, batch_size: Optional[int] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Standard training loop mimicking the PNN architecture interface.
        """
        self.config.learning_rate = learning_rate
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        history = {'loss': [], 'epoch': []}
        n_samples = X.shape[0]
        batch_size = batch_size or n_samples
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        for epoch in range(epochs):
            indices = torch.randperm(n_samples)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                loss = self.train_step(X_batch, y_batch, optimizer)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)
            history['epoch'].append(epoch)
            
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_best_weights()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            predictions = self(X_tensor)
        return predictions.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model."""
        y_pred = self.predict(X)
        mse = np.mean((y_pred - y) ** 2)
        mae = np.mean(np.abs(y_pred - y))
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        return {'mse': float(mse), 'mae': float(mae), 'r2': float(r2)}
    
    def _save_best_weights(self) -> None:
        self.best_weights = {k: v.cpu().clone() for k, v in self.state_dict().items()}
    
    def restore_best_weights(self) -> None:
        if self.best_weights is not None:
            self.load_state_dict(self.best_weights)

    def summary(self) -> str:
        lines = ["OrthoPoly Neural Network Summary", "=" * 40]
        lines.append(f"Input dimension: {self.config.input_dim}")
        lines.append(f"Output dimension: {self.config.output_dim}")
        lines.append(f"Hidden layers: {self.config.hidden_dims}")
        lines.append(f"Max Polynomial degree: {self.config.max_degree}")
        lines.append(f"Low-rank Factorization: {self.config.rank}")
        lines.append(f"Basis Type: {self.config.basis_type.value}")
        lines.append("-" * 40)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"OrthoPolyNetwork(input={self.config.input_dim}, hidden={self.config.hidden_dims}, output={self.config.output_dim})"


class OrthoPolyTrainer:
    """
    Advanced Trainer for OrthoPoly networks handling Spectral Regularization.
    """
    def __init__(self, model: OrthoPolyNetwork, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, spectral_reg: float = 0.01):
        self.model = model
        self.spectral_reg = spectral_reg
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.history = defaultdict(list)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        mse_loss = nn.functional.mse_loss(y_pred, y)
        
        if self.spectral_reg > 0:
            spec_loss = 0.0
            for layer in self.model.layers:
                if isinstance(layer, OrthoPolyLayer):
                    # Weight coefficients by the inverse of the basis norm `h_n`.
                    # This correctly penalizes high-degree terms for families where h_n grows.
                    inv_h = 1.0 / layer.h_norms.view(1, 1, -1).clamp(min=1e-8)
                    spec_loss = spec_loss + torch.sum((layer.cheby_coeffs ** 2) * inv_h)
            total_loss = mse_loss + self.spectral_reg * spec_loss
        else:
            total_loss = mse_loss
            
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        return total_loss.item()

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
              batch_size: int = 256, verbose: bool = True) -> Dict[str, List[float]]:
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                loss = self.train_step(x_batch, y_batch)
                epoch_loss += loss
            avg_loss = epoch_loss / len(loader)
            self.history['loss'].append(avg_loss)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        return self.history
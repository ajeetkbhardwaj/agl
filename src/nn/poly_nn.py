"""
Polynomial Neural Network (PNN)
===============================
Main architecture for polynomial neural networks based on algebraic geometry.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..layers.polynomial_layer import PolynomialLayer
from ..layers.groebner_layer import GroebnerLayer
from ..layers.variety_layer import VarietyLayer
from .mixins import AlgebraicExtractionMixin


@dataclass
class PNNConfig:
    """Configuration for Polynomial Neural Network"""
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [32])
    polynomial_degree: int = 2
    use_groebner: bool = False
    use_variety: bool = False
    constraint_ideals: Optional[List] = None
    variety_representations: Optional[List] = None
    activation: str = 'polynomial'  # 'polynomial', 'groebner', 'variety'
    dropout: float = 0.0
    use_batchnorm: bool = False
    use_bias: bool = True
    learning_rate: float = 0.01


class _PolynomialActivation(nn.Module):
    """Polynomial activation: a(x) = x + x^2 / 2."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x ** 2) / 2.0


class PolynomialNeuralNetwork(nn.Module, AlgebraicExtractionMixin):
    """
    A neural network that uses polynomial layers instead of standard linear layers.
    
    The network can be configured to use:
    - Standard polynomial layers (PolynomialLayer)
    - Groebner-constrained layers (GroebnerLayer)
    - Variety-constrained layers (VarietyLayer)
    
    Mathematical Foundation:
    - Polynomial layers: f(x) = W @ x + W2 @ x^2 + ... + bias
    - Groebner layers: weights constrained to an ideal
    - Variety layers: activations constrained to an algebraic variety
    """
    
    def __init__(self, config: Optional[PNNConfig] = None, **kwargs):
        """
        Initialize the Polynomial Neural Network.
        
        Args:
            config: PNN configuration
            **kwargs: Configuration arguments if config is None
        """
        super().__init__()
        if config is None:
            self.custom_layers = kwargs.pop('custom_layers', None)
            self.config = PNNConfig(**kwargs)
        else:
            self.config = config
            self.custom_layers = None
            
        self.activation_fn = self._get_activation_fn(self.config.activation)
        
        self.layers = nn.ModuleList()
        
        if self.custom_layers:
            for layer in self.custom_layers:
                self.layers.append(layer)
        else:
            self._build_network()
        
        # Training state
        self.best_loss = float('inf')
        self.best_weights: Optional[Dict] = None
        
    def _get_activation_fn(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'polynomial': _PolynomialActivation(),
            'silu': nn.SiLU(),
            'none': nn.Identity(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    @property
    def input_dim(self) -> int:
        return self.config.input_dim
    
    @property
    def hidden_dims(self) -> List[int]:
        return self.config.hidden_dims
    
    @property
    def output_dim(self) -> int:
        return self.config.output_dim
    
    @property
    def polynomial_degree(self) -> int:
        return self.config.polynomial_degree
    
    def _build_network(self) -> None:
        """Build the network architecture based on config"""
        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
        
        for i in range(len(dims) - 1):
            layer_input_dim = dims[i]
            layer_output_dim = dims[i + 1]
            is_last_layer = i == (len(dims) - 2)
            
            if self.config.use_groebner and self.config.constraint_ideals:
                # Use Groebner-constrained layer
                constraint = self.config.constraint_ideals[i] if i < len(self.config.constraint_ideals) else []
                layer = GroebnerLayer(
                    input_dim=layer_input_dim,
                    output_dim=layer_output_dim,
                    constraint_ideal=constraint
                )
            elif self.config.use_variety and self.config.variety_representations:
                # Use Variety-constrained layer
                variety = self.config.variety_representations[i] if i < len(self.config.variety_representations) else None
                layer = VarietyLayer(
                    input_dim=layer_input_dim,
                    output_dim=layer_output_dim,
                    variety=variety
                )
            else:
                # Use standard polynomial layer
                dropout_rate = getattr(self.config, '_explicit_last_layer_dropout', 0.0) if is_last_layer else self.config.dropout
                use_bn = getattr(self.config, '_explicit_last_layer_batchnorm', False) if is_last_layer else self.config.use_batchnorm
                
                layer = PolynomialLayer(
                    input_dim=layer_input_dim,
                    output_dim=layer_output_dim,
                    max_degree=self.config.polynomial_degree,
                    polynomial_type=self.config.activation if self.config.activation in ['general', 'homogeneous', 'quadratic'] else 'general',
                    dropout_rate=dropout_rate,
                    use_batchnorm=use_bn
                )
            
            self.layers.append(layer)

            if not is_last_layer:
                self.layers.append(self.activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Output tensor
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        for layer in self.layers:
            x = layer(x)

        return x
    
    def train_step(self, X: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        Perform a single training step.
        
        Args:
            X: Input tensor
            y: Target tensor
            optimizer: PyTorch optimizer
        """
        self.train()
        optimizer.zero_grad()
        y_pred = self(X)
        loss = nn.MSELoss()(y_pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        
        # The Constraint Promise: Hard Topological Projections post-optimization
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, 'project_to_variety'):
                    layer.project_to_variety()
                elif hasattr(layer, 'project_weights'):
                    layer.project_weights()
                    
        return loss.item()
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            learning_rate: float = 0.001, batch_size: Optional[int] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the network.
        
        Args:
            X: Training inputs of shape (n_samples, input_dim)
            y: Training targets of shape (n_samples, output_dim)
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Mini-batch size (None for full batch)
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
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
            # Shuffle data
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
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def _save_best_weights(self) -> None:
        """Save current weights as best weights"""
        self.best_weights = {k: v.cpu().clone() for k, v in self.state_dict().items()}
    
    def restore_best_weights(self) -> None:
        """Restore best weights"""
        if self.best_weights is not None:
            self.load_state_dict(self.best_weights)
            
    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str) -> None:
        """Load model from disk."""
        self.load_state_dict(torch.load(path, weights_only=True))
    
    def get_layer_info(self) -> List[Dict]:
        """Get information about each layer"""
        info = []
        for i, layer in enumerate(self.layers):
            layer_info = {
                'layer': i,
                'type': type(layer).__name__,
                'input_dim': getattr(layer, 'input_dim', None),
                'output_dim': getattr(layer, 'output_dim', None),
                'max_degree': getattr(layer, 'max_degree', None)
            }
            info.append(layer_info)
        return info
    
    def summary(self) -> str:
        """Get a summary of the network architecture"""
        lines = ["Polynomial Neural Network Summary", "=" * 40]
        lines.append(f"Input dimension: {self.config.input_dim}")
        lines.append(f"Output dimension: {self.config.output_dim}")
        lines.append(f"Hidden layers: {self.config.hidden_dims}")
        lines.append(f"Polynomial degree: {self.config.polynomial_degree}")
        lines.append(f"Total layers: {len(self.layers)}")
        lines.append("")
        lines.append("Layer Details:")
        lines.append("-" * 40)
        for info in self.get_layer_info():
            lines.append(f"Layer {info['layer']}: {info['type']}")
            if info['input_dim'] is not None:
                lines.append(f"  Input: {info['input_dim']}, Output: {info['output_dim']}")
        lines.append("-" * 40)
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"PNN(input={self.config.input_dim}, hidden={self.config.hidden_dims}, output={self.config.output_dim})"


class DeepPolynomialNetwork(PolynomialNeuralNetwork):
    """
    A deeper variant of PNN with residual connections.
    """
    
    def __init__(self, config: PNNConfig, residual: bool = True):
        super().__init__(config)
        self.residual = residual


def create_pnn(input_dim: int, output_dim: int, 
               hidden_dims: List[int] = [64, 32], 
               polynomial_degree: int = 2, 
               **kwargs) -> PolynomialNeuralNetwork:
    """
    Factory function to create a standard PNN.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions
        polynomial_degree: Degree of polynomial expansion
        **kwargs: Additional arguments for PNNConfig
        
    Returns:
        PolynomialNeuralNetwork instance
    """
    config = PNNConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        polynomial_degree=polynomial_degree,
        **kwargs
    )
    return PolynomialNeuralNetwork(config)


def create_groebner_pnn(input_dim: int, output_dim: int, 
                        constraint_ideals: List,
                        hidden_dims: List[int] = [64, 32], 
                        **kwargs) -> PolynomialNeuralNetwork:
    """
    Factory function to create a Groebner-constrained PNN.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        constraint_ideals: List of constraint ideals for each layer
        hidden_dims: Hidden layer dimensions
        **kwargs: Additional arguments for PNNConfig
        
    Returns:
        PolynomialNeuralNetwork with Groebner layers
    """
    config = PNNConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        use_groebner=True,
        constraint_ideals=constraint_ideals,
        **kwargs
    )
    return PolynomialNeuralNetwork(config)
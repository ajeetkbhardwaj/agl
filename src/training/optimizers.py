"""
Optimizers Module
=================
Custom optimizers for Polynomial Neural Networks.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Callable
import numpy as np


class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
    def step(self, model) -> None:
        """Perform optimization step."""
        raise NotImplementedError
        
        
import torch
class PyTorchOptimizer:
    """Wrapper for custom optimizers to work with PyTorch models."""
    def __init__(self, base_optimizer_class, **kwargs):
        self.base_optimizer = base_optimizer_class(**kwargs)
        self.param_groups = []
        
    def add_param_group(self, params, **kwargs):
        self.param_groups.append({
            'params': list(params) if hasattr(params, '__iter__') else [params],
            **kwargs
        })
        
    def step(self) -> None:
        for group in self.param_groups:
            params = group['params']
            for param in params:
                if param.grad is None:
                    continue
                grad = param.grad.data.numpy()
                param_data = param.data.numpy()
                
                class DummyModel:
                    def __init__(self, data, g):
                        self.layers = [DummyLayer(data, g)]
                class DummyLayer:
                    def __init__(self, data, g):
                        self.weight_matrix = data
                        self.grad_weight = g
                
                model = DummyModel(param_data, grad)
                self.base_optimizer.step(model)
                param.data = torch.tensor(model.layers[0].weight_matrix, dtype=param.dtype, device=param.device)
                
    def zero_grad(self) -> None:
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0, 
                 nesterov: bool = False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity: Dict = {}
    
    def step(self, model) -> None:
        """Perform SGD step."""
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weight_matrix'):
                grad = getattr(layer, 'grad_weight', None)
                if grad is None:
                    continue
                
                key = f'layer_{i}_weight'
                
                if self.momentum > 0:
                    if key not in self.velocity:
                        self.velocity[key] = np.zeros_like(layer.weight_matrix)
                        
                    if self.nesterov:
                        # Nesterov: velocity update = momentum * velocity + grad
                        velocity_update = self.momentum * self.velocity[key] + grad
                        self.velocity[key] = velocity_update
                        layer.weight_matrix -= self.learning_rate * velocity_update
                    else:
                        self.velocity[key] = (self.momentum * self.velocity[key] + grad)
                        layer.weight_matrix -= self.learning_rate * self.velocity[key]
                else:
                    layer.weight_matrix -= self.learning_rate * grad
                    
                np.clip(layer.weight_matrix, -10, 10, out=layer.weight_matrix)
            
            if hasattr(layer, 'bias') and getattr(layer, 'bias', None) is not None:
                grad_bias = getattr(layer, 'grad_bias', None)
                if grad_bias is not None:
                    bias_key = f'layer_{i}_bias'
                    if self.momentum > 0:
                        if bias_key not in self.velocity:
                            self.velocity[bias_key] = np.zeros_like(layer.bias)
                        self.velocity[bias_key] = (self.momentum * self.velocity[bias_key] + grad_bias)
                        layer.bias -= self.learning_rate * self.velocity[bias_key]
                    else:
                        layer.bias -= self.learning_rate * grad_bias
                    np.clip(layer.bias, -10, 10, out=layer.bias)


class Adam(Optimizer):
    """Adam optimizer with algebraic structure awareness."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict = {}
        self.v: Dict = {}
        self.t = 0
    
    def step(self, model) -> None:
        """Perform Adam step."""
        self.t += 1
        
        beta1_power = self.beta1 ** self.t
        beta2_power = self.beta2 ** self.t
        
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weight_matrix'):
                grad = getattr(layer, 'grad_weight', None)
                if grad is None:
                    continue
                
                key = f'layer_{i}_weight'
                
                if key not in self.m:
                    self.m[key] = np.zeros_like(layer.weight_matrix)
                    self.v[key] = np.zeros_like(layer.weight_matrix)
                
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
                
                m_hat = self.m[key] / (1 - beta1_power)
                v_hat = self.v[key] / (1 - beta2_power)
                
                layer.weight_matrix -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                np.clip(layer.weight_matrix, -10, 10, out=layer.weight_matrix)
            
            if hasattr(layer, 'bias') and getattr(layer, 'bias', None) is not None:
                grad_bias = getattr(layer, 'grad_bias', None)
                if grad_bias is not None:
                    key = f'layer_{i}_bias'
                    
                    if key not in self.m:
                        self.m[key] = np.zeros_like(layer.bias)
                        self.v[key] = np.zeros_like(layer.bias)
                    
                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad_bias
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad_bias ** 2)
                    
                    m_hat = self.m[key] / (1 - beta1_power)
                    v_hat = self.v[key] / (1 - beta2_power)
                    
                    layer.bias -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    np.clip(layer.bias, -10, 10, out=layer.bias)


class RMSprop(Optimizer):
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, rho: float = 0.9, 
                 epsilon: float = 1e-8, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.momentum = momentum
        self.square_avg: Dict = {}
        self.accumulators: Dict = {}
    
    def step(self, model) -> None:
        """Perform RMSprop step."""
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weight_matrix'):
                grad = getattr(layer, 'grad_weight', None)
                if grad is None:
                    continue
                
                key = f'layer_{i}_weight'
                
                if key not in self.square_avg:
                    self.square_avg[key] = np.zeros_like(layer.weight_matrix)
                
                # Update running average of squared gradients
                self.square_avg[key] = (self.rho * self.square_avg[key] + 
                                        (1 - self.rho) * (grad ** 2))
                
                # Compute update
                avg = np.sqrt(self.square_avg[key] + self.epsilon)
                
                if self.momentum > 0:
                    if key not in self.accumulators:
                        self.accumulators[key] = np.zeros_like(layer.weight_matrix)
                    self.accumulators[key] = (self.momentum * self.accumulators[key] + 
                                              grad / avg)
                    layer.weight_matrix -= self.learning_rate * self.accumulators[key]
                else:
                    layer.weight_matrix -= self.learning_rate * grad / avg
                
                layer.weight_matrix = np.clip(layer.weight_matrix, -10, 10)


class AdaGrad(Optimizer):
    """AdaGrad optimizer with algebraic structure awareness."""
    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-10):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.grad_squared: Dict = {}
    
    def step(self, model) -> None:
        """Perform AdaGrad step."""
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weight_matrix'):
                grad = getattr(layer, 'grad_weight', None)
                if grad is None:
                    continue
                
                key = f'layer_{i}_weight'
                
                if key not in self.grad_squared:
                    self.grad_squared[key] = np.zeros_like(layer.weight_matrix)
                
                # Accumulate squared gradients
                self.grad_squared[key] += grad ** 2
                
                # Compute adaptive learning rate
                avg = np.sqrt(self.grad_squared[key] + self.epsilon)
                
                layer.weight_matrix -= self.learning_rate * grad / avg
                layer.weight_matrix = np.clip(layer.weight_matrix, -10, 10)


class PolynomialAwareOptimizer(Optimizer):
    """
    Optimizer that leverages polynomial structure for more efficient updates.
    Uses the algebraic variety structure to constrain gradient updates.
    """
    
    def __init__(self, learning_rate: float = 0.01, variety_projection: bool = True,
                 ideal_constraint: bool = True):
        super().__init__(learning_rate)
        self.variety_projection = variety_projection
        self.ideal_constraint = ideal_constraint
    
    def project_to_variety(self, weights: np.ndarray, variety_repr: Optional[Dict] = None) -> np.ndarray:
        """
        Project weights onto an algebraic variety using iterative gradient projection.
        """
        if variety_repr is None:
            return weights
            
        max_iterations = variety_repr.get('max_iterations', 100)
        tolerance = variety_repr.get('tolerance', 1e-6)
        generators = variety_repr.get('generators', [])
        
        if not generators:
            return weights
            
        projected = weights.copy()
        for iteration in range(max_iterations):
            constraint_violations = np.array([
                gen(projected) if callable(gen) or hasattr(gen, '__call__') else np.sum(gen * projected)
                for gen in generators
            ])
            
            if np.linalg.norm(constraint_violations) < tolerance:
                break
                
            # Compute constraint gradient via finite differences
            eps = 1e-7
            gradient = np.zeros_like(projected)
            for i, gen in enumerate(generators):
                for j in range(projected.size):
                    w_plus = projected.copy()
                    w_plus.flat[j] += eps
                    w_minus = projected.copy()
                    w_minus.flat[j] -= eps
                    
                    val_plus = gen(w_plus) if callable(gen) or hasattr(gen, '__call__') else np.sum(gen * w_plus)
                    val_minus = gen(w_minus) if callable(gen) or hasattr(gen, '__call__') else np.sum(gen * w_minus)
                    
                    gradient.flat[j] += ((val_plus - val_minus) / (2 * eps)) * constraint_violations[i]
            
            projected -= 0.1 * gradient
            # The step size 0.1 is fixed; consider making it a configurable parameter or adaptive.
            
        return projected
    
    def step(self, model) -> None:
        """Perform polynomial-aware optimization step."""
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weight_matrix'):
                grad = getattr(layer, 'grad_weight', None)
                if grad is None:
                    continue
                
                # Standard gradient update
                update = self.learning_rate * grad
                
                # Apply variety projection if available
                if self.variety_projection and hasattr(model, 'variety_representations'):
                    variety = model.variety_representations[i] if i < len(model.variety_representations) else None
                    if variety is not None:
                        updated_weights = layer.weight_matrix - update
                        layer.weight_matrix = self.project_to_variety(updated_weights, variety)
                    else:
                        layer.weight_matrix -= update
                else:
                    layer.weight_matrix -= update
                
                np.clip(layer.weight_matrix, -10, 10, out=layer.weight_matrix)


def create_optimizer(name: str, learning_rate: float = 0.01, **kwargs) -> Optimizer:
    """
    Factory function to create optimizers.
    
    Args:
        name: Optimizer name ('sgd', 'adam', 'rmsprop', 'adagrad', 'polynomial_aware')
        learning_rate: Learning rate
        **kwargs: Additional optimizer parameters
        
    Returns:
        Optimizer instance
    """
    optimizers = {
        'sgd': SGD,
        'adam': Adam,
        'rmsprop': RMSprop,
        'adagrad': AdaGrad,
        'polynomial_aware': PolynomialAwareOptimizer
    }
    
    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name.lower()](learning_rate=learning_rate, **kwargs)
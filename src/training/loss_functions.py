"""
Loss Functions Module
=====================
Algebraic loss functions for Polynomial Neural Networks.
"""

from __future__ import annotations
from typing import Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class LossFunction:
    """Base loss function class."""
    name: str = "base"
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss."""
        raise NotImplementedError
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient with respect to predictions."""
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss."""
    
    def __init__(self):
        super().__init__("mse")
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class MeanAbsoluteError(LossFunction):
    """Mean Absolute Error loss."""
    
    def __init__(self):
        super().__init__("mae")
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true) / y_true.size


class CrossEntropy(LossFunction):
    """Cross-entropy loss for classification."""
    
    def __init__(self, epsilon: float = 1e-15, multi_class: bool = False):
        super().__init__("cross_entropy")
        self.epsilon = epsilon
        self.multi_class = multi_class
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.multi_class:
            y_pred = self._softmax(y_pred)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        if self.multi_class:
            return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=-1)))
        else:
            return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n = y_true.size
        if self.multi_class:
            y_pred_soft = self._softmax(y_pred)
            return (y_pred_soft - y_true) / n
        else:
            y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
            return (y_pred - y_true) / n


class IdealMembershipLoss(LossFunction):
    """
    Algebraic loss based on ideal membership.
    Measures how well predictions satisfy polynomial ideal constraints.
    """
    
    def __init__(self, ideal_generators: Optional[list] = None, 
                 weight: float = 1.0):
        super().__init__("ideal_membership")
        self.ideal_generators = ideal_generators or []
        self.weight = weight
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute ideal membership loss.
        
        The loss measures how well the predictions satisfy the ideal constraints.
        For a polynomial ideal I, the loss is the sum of squared evaluations
        of ideal generators at the prediction points.
        """
        base_loss = float(np.mean((y_true - y_pred) ** 2))
        
        if not self.ideal_generators:
            return base_loss
        
        # Compute ideal constraint violation
        ideal_loss = 0.0
        for generator in self.ideal_generators:
            # Evaluate generator at predictions
            # This is a simplified version - in practice would use proper polynomial evaluation
            # If generator is callable, it's assumed to evaluate the polynomial.
            # If generator is an array, it's treated as coefficients for a linear form (dot product).
            if callable(generator):
                # Assumes generator(y_pred) returns a (batch_size,) array or scalar
                constraint_value = generator(y_pred)
            else:
                # Assume generator is a polynomial expression
                constraint_value = np.sum(generator * y_pred)
            ideal_loss += np.mean(constraint_value ** 2)
        
        return base_loss + self.weight * ideal_loss
    
    def _compute_ideal_violation(self, y_pred: np.ndarray) -> np.ndarray:
        """Compute vector of constraint violations."""
        violations = []
        for generator in self.ideal_generators:
            if callable(generator):
                # Assumes generator(y_pred) returns a (batch_size,) array or scalar
                violations.append(generator(y_pred))
            elif isinstance(generator, np.ndarray):
                violations.append(np.sum(generator * y_pred, axis=-1))
            else:
                violations.append(np.sum(np.array(generator) * y_pred, axis=-1))
        
        if violations:
            arr = np.array(violations)
            return arr.T if arr.ndim > 1 else arr
        return np.zeros_like(y_pred)
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient including ideal constraint terms."""
        n = y_true.size
        grad = 2 * (y_pred - y_true) / n
        
        if not self.ideal_generators:
            return grad
        
        violations = self._compute_ideal_violation(y_pred)
        
        # Approximate Jacobian via finite differences (vectorized)
        eps = 1e-5
        batch_size = y_pred.shape[0] if y_pred.ndim > 1 else 1
        n_constraints = len(self.ideal_generators)
        
        J = np.zeros((batch_size, n_constraints, y_pred.shape[-1])) if y_pred.ndim > 1 else np.zeros((n_constraints, y_pred.shape[-1]))
        
        for j in range(y_pred.shape[-1]):
            y_plus = y_pred.copy()
            if y_pred.ndim > 1:
                y_plus[..., j] += eps
            else:
                y_plus[j] += eps
                
            y_minus = y_pred.copy()
            if y_pred.ndim > 1:
                y_minus[..., j] -= eps
            else:
                y_minus[j] -= eps
                
            J[..., j] = (self._compute_ideal_violation(y_plus) - 
                         self._compute_ideal_violation(y_minus)) / (2 * eps)
        
        if y_pred.ndim > 1:
            ideal_grad = 2 * np.einsum('bij,bi->bj', J, violations) / batch_size
        else:
            ideal_grad = 2 * J.T @ violations
            
        return grad + self.weight * ideal_grad


class VarietyConstraintLoss(LossFunction):
    """
    Loss that enforces predictions to lie on an algebraic variety.
    Uses distance to variety as a penalty term.
    """
    
    def __init__(self, variety_repr: Optional[dict] = None, 
                 weight: float = 1.0):
        super().__init__("variety_constraint")
        self.variety_repr = variety_repr
        self.weight = weight
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss with variety constraint penalty."""
        base_loss = float(np.mean((y_true - y_pred) ** 2))
        
        if self.variety_repr is None:
            return base_loss
        
        # Compute distance to variety
        variety_loss = np.mean(self._compute_variety_distance(y_pred))
        
        return base_loss + self.weight * variety_loss
    
    def _compute_variety_distance(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute distance to algebraic variety for each sample.
        """
        if self.variety_repr is None:
            return np.zeros(y_pred.shape[0] if y_pred.ndim > 1 else 1)
        
        total_violation = np.zeros(y_pred.shape[0] if y_pred.ndim > 1 else 1)
        
        if 'generators' in self.variety_repr:
            for gen in self.variety_repr['generators']:
                if callable(gen) or hasattr(gen, '__call__'):
                    # Assumes gen(y_pred) returns a (batch_size,) array or scalar
                    violation = gen(y_pred)
                else:
                    violation = np.sum(gen * y_pred, axis=-1) if y_pred.ndim > 1 else np.dot(gen, y_pred)
                total_violation += violation ** 2
        
        return total_violation
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient with variety constraint."""
        grad = 2 * (y_pred - y_true) / y_true.size
        
        if self.variety_repr is None:
            return grad
        
        eps = 1e-5
        batch_size = y_pred.shape[0] if y_pred.ndim > 1 else 1
        n_features = y_pred.shape[-1]
        
        dist_orig = self._compute_variety_distance(y_pred)
        grad_variety = np.zeros_like(y_pred)
        
        for j in range(n_features):
            y_plus = y_pred.copy()
            if y_pred.ndim > 1:
                y_plus[..., j] += eps
            else:
                y_plus[j] += eps
                
            dist_plus = self._compute_variety_distance(y_plus)
            
            if y_pred.ndim > 1:
                grad_variety[..., j] = (dist_plus - dist_orig) / eps
            else:
                grad_variety[j] = (dist_plus - dist_orig) / eps
        
        return grad + self.weight * grad_variety / batch_size


class SyzygyLoss(LossFunction):
    """
    Loss based on syzygy constraints.
    Enforces polynomial relations between network outputs.
    """
    
    def __init__(self, syzygy_relations: Optional[list] = None,
                 weight: float = 1.0):
        super().__init__("syzygy")
        self.syzygy_relations = syzygy_relations or []
        self.weight = weight
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss with syzygy constraint penalty."""
        base_loss = float(np.mean((y_true - y_pred) ** 2))
        
        if not self.syzygy_relations:
            return base_loss
        
        # Compute syzygy violation
        syzygy_loss = 0.0
        for relation in self.syzygy_relations:
            # Evaluate syzygy relation
            if callable(relation):
                violation = relation(y_pred)
            else:
                violation = np.sum(relation * y_pred)
            syzygy_loss += np.mean(violation ** 2)
        
        return base_loss + self.weight * syzygy_loss
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient with syzygy constraints."""
        grad = 2 * (y_pred - y_true) / y_true.size
        
        if not self.syzygy_relations:
            return grad
        
        batch_size = y_pred.shape[0] if y_pred.ndim > 1 else 1
        n_features = y_pred.shape[-1] if y_pred.ndim > 0 else 1

        syzygy_grad_term = np.zeros_like(y_pred)
        eps = 1e-5

        for relation_idx, relation in enumerate(self.syzygy_relations):
            # Evaluate the relation at current y_pred.
            # Assumes relation(y_pred) returns a (batch_size,) array or scalar.
            # If relation is an array, it's treated as coefficients for a linear form.
            current_violations = relation(y_pred) if callable(relation) else \
                                 (np.sum(relation * y_pred, axis=-1) if y_pred.ndim > 1 else np.dot(relation, y_pred))
            
            # Compute Jacobian of this single relation w.r.t. y_pred for each sample in batch
            # J_relation will be (batch_size, n_features) if y_pred is batched, else (n_features,)
            J_relation = np.zeros((batch_size, n_features)) if y_pred.ndim > 1 else np.zeros(n_features)

            for j in range(n_features):
                y_plus = y_pred.copy()
                if y_pred.ndim > 1:
                    y_plus[..., j] += eps
                else:
                    y_plus[j] += eps

                y_minus = y_pred.copy()
                if y_pred.ndim > 1:
                    y_minus[..., j] -= eps
                else:
                    y_minus[j] -= eps
                
                val_plus = relation(y_plus) if callable(relation) else \
                           (np.sum(relation * y_plus, axis=-1) if y_pred.ndim > 1 else np.dot(relation, y_plus))
                val_minus = relation(y_minus) if callable(relation) else \
                            (np.sum(relation * y_minus, axis=-1) if y_pred.ndim > 1 else np.dot(relation, y_minus))
                
                if y_pred.ndim > 1:
                    J_relation[:, j] = (val_plus - val_minus) / (2 * eps)
                else:
                    J_relation[j] = (val_plus - val_minus) / (2 * eps)
            
            # Contribution to gradient for this relation:
            # For each sample 'b', it's 2 * current_violations[b] * J_relation[b, :]
            if y_pred.ndim > 1:
                # current_violations is (batch_size,)
                # J_relation is (batch_size, n_features)
                # Resulting term is (batch_size, n_features)
                syzygy_grad_term += current_violations[:, np.newaxis] * J_relation
            else: # Single sample case
                # current_violations is scalar
                # J_relation is (n_features,)
                # Resulting term is (n_features,)
                syzygy_grad_term += current_violations * J_relation
        
        # The total syzygy loss is sum_k (1/batch_size * sum_b (relation_k(y_pred_b)**2))
        # So the gradient is (1/batch_size) * sum_k (sum_b (2 * relation_k(y_pred_b) * grad(relation_k(y_pred_b))))
        return grad + self.weight * (2 * syzygy_grad_term / batch_size)


class HuberLoss(LossFunction):
    """Huber loss - combines MSE and MAE."""
    
    def __init__(self, delta: float = 1.0):
        super().__init__("huber")
        self.delta = delta
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_true - y_pred
        abs_error = np.abs(error)
        
        # Quadratic loss for small errors, linear for large errors
        quadratic = np.where(abs_error <= self.delta, 
                            0.5 * error ** 2, 
                            self.delta * (abs_error - 0.5 * self.delta))
        
        return float(np.mean(quadratic))
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        # Gradient is error for small errors, delta * sign(error) for large errors
        grad = np.where(abs_error <= self.delta, 
                       error, 
                       self.delta * np.sign(error))
        
        return grad / y_true.size


class HingeLoss(LossFunction):
    """Hinge loss for SVM-style classification."""
    
    def __init__(self, margin: float = 1.0):
        super().__init__("hinge")
        self.margin = margin
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # y_true should be in {-1, 1}
        return float(np.mean(np.maximum(0, self.margin - y_true * y_pred)))
    
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Gradient is -y_true where margin > y_true * y_pred
        grad = np.where(self.margin > y_true * y_pred, -y_true, 0.0)
        return grad / y_true.size


def create_loss(name: str, **kwargs) -> LossFunction:
    """
    Factory function to create loss functions.
    
    Args:
        name: Loss name ('mse', 'mae', 'cross_entropy', 'ideal_membership', 
               'variety_constraint', 'syzygy', 'huber', 'hinge')
        **kwargs: Additional loss parameters
        
    Returns:
        LossFunction instance
    """
    losses = {
        'mse': MeanSquaredError,
        'mae': MeanAbsoluteError,
        'cross_entropy': CrossEntropy,
        'ideal_membership': IdealMembershipLoss,
        'variety_constraint': VarietyConstraintLoss,
        'syzygy': SyzygyLoss,
        'huber': HuberLoss,
        'hinge': HingeLoss
    }
    
    if name.lower() not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    
    return losses[name.lower()](**kwargs)
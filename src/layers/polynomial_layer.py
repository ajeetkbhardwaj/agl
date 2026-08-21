"""
Polynomial Neural Network Layer
================================
A neural network layer where weights are polynomials.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from ..alggeom.polynomial import Polynomial, Monomial, Variable


class PolynomialLayer(nn.Module):
    """
    A neural network layer where each weight is a polynomial function.
    
    Forward pass: y_j = sum_i poly_{ij}(x) * x_i
    
    where poly_{ij}(x) is a polynomial in the input x.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 max_degree: int = 2,
                 polynomial_type: str = 'general',
                 use_batchnorm: bool = False,
                 dropout_rate: float = 0.0,
                 seed: Optional[int] = None):
        """
        Initialize polynomial layer.
        
        Args:
            input_dim: Dimension of input
            output_dim: Dimension of output
            max_degree: Maximum degree of polynomials
            polynomial_type: 'general', 'homogeneous', 'quadratic'
            use_batchnorm: Whether to apply batch normalization
            dropout_rate: Dropout probability (0.0 means no dropout)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_degree = max_degree
        self.polynomial_type = polynomial_type
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.variables: List[Variable] = [Variable(f'x{i}') for i in range(input_dim)]
        self.rng = np.random.default_rng(seed if seed is not None else np.random.randint(0, 2**31))
        
        if self.use_batchnorm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        
        # Initialize polynomial weights
        self.polynomial_weights: List[List[Polynomial]] = []
        self.poly_coeffs = nn.ParameterDict()
        self._initialize_weights()
        
        # Add learnable weight matrix for backprop compatibility
        limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
        self.weight_matrix = nn.Parameter(torch.Tensor(self.output_dim, self.input_dim).uniform_(-limit, limit))
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        
        # Cached computations
        self._is_synced: bool = False
        self.last_input: Optional[torch.Tensor] = None
        self.last_output: Optional[torch.Tensor] = None
    
    def _initialize_weights(self) -> None:
        """Initialize polynomial weights with random coefficients"""
        self.polynomial_weights = []
        self.poly_coeffs.clear()
        
        # Xavier/Kaiming style variance scaling for polynomial features
        std_dev = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        
        for j in range(self.output_dim):
            row = []
            for i in range(self.input_dim):
                poly = self._create_random_polynomial(std_dev)
                row.append(poly)
                
                # Register terms as parameters
                for m_idx, (monomial, coeff) in enumerate(poly.terms.items()):
                    key = f"w_{j}_{i}_{m_idx}"
                    val = float(np.real(coeff))
                    self.poly_coeffs[key] = nn.Parameter(torch.tensor(val, dtype=torch.float32))
                    
            self.polynomial_weights.append(row)
    
    def _create_random_polynomial(self, std_dev: float = 0.01) -> Polynomial:
        """Create a random polynomial of specified type"""
        if self.polynomial_type == 'quadratic':
            return self._create_quadratic_polynomial(std_dev)
        elif self.polynomial_type == 'homogeneous':
            return self._create_homogeneous_polynomial(std_dev)
        else:
            return self._create_general_polynomial(std_dev)
    
    def _create_general_polynomial(self, std_dev: float = 0.01) -> Polynomial:
        """Create a general polynomial with random coefficients"""
        terms = {}

        # Constant term
        if self.rng.random() > 0.3:
            terms[Monomial(())] = self.rng.standard_normal() * std_dev

        # Generate monomials up to max_degree
        for degree in range(1, self.max_degree + 1):
            if self.rng.random() > 0.5:
                monomial = self._random_monomial(self.variables, degree)
                # Scale higher degree terms down to prevent exploding activations
                terms[monomial] = self.rng.standard_normal() * (std_dev / degree)

        return Polynomial(terms)

    def _create_quadratic_polynomial(self, std_dev: float = 0.01) -> Polynomial:
        """Create a quadratic polynomial"""
        terms = {}

        # Constant
        if self.rng.random() > 0.3:
            terms[Monomial(())] = self.rng.standard_normal() * std_dev

        # Linear terms
        for v in self.variables:
            if self.rng.random() > 0.4:
                monomial = Monomial(((v, 1),))
                terms[monomial] = self.rng.standard_normal() * std_dev

        # Quadratic terms
        for i, v1 in enumerate(self.variables):
            for v2 in self.variables[i:]:
                if self.rng.random() > 0.6:
                    if v1 != v2:
                        monomial = Monomial(tuple(sorted(((v1, 1), (v2, 1)), key=lambda t: t[0].name)))
                    else:
                        monomial = Monomial(((v1, 2),))
                    terms[monomial] = self.rng.standard_normal() * (std_dev / 2.0)

        return Polynomial(terms)

    def _create_homogeneous_polynomial(self, std_dev: float = 0.01) -> Polynomial:
        """Create a homogeneous polynomial (all terms same degree)"""
        terms = {}

        monomial = self._random_monomial(self.variables, self.max_degree)
        terms[monomial] = self.rng.standard_normal() * (std_dev / self.max_degree)

        return Polynomial(terms)

    def _random_monomial(self, variables: List[Variable], degree: int) -> Monomial:
        """Generate a random monomial of given degree"""
        monomial_dict = {}
        remaining_degree = degree

        for v in variables:
            if remaining_degree == 0:
                break
            max_exp = min(remaining_degree, self.rng.integers(0, 4))
            if max_exp > 0:
                monomial_dict[v] = max_exp
                remaining_degree -= max_exp

        # Distribute remaining degree
        while remaining_degree > 0:
            v = variables[self.rng.integers(len(variables))]
            monomial_dict[v] = monomial_dict.get(v, 0) + 1
            remaining_degree -= 1

        return Monomial(tuple(sorted(monomial_dict.items(), key=lambda item: item[0].name)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through polynomial layer using PyTorch Autograd.
        
        Args:
            x: Input tensor of shape (input_dim,) or (batch, input_dim)
        
        Returns:
            Output tensor of shape (output_dim,) or (batch, output_dim)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        is_batch = x.dim() == 2
        
        if self.training:
            self._is_synced = False
        if not is_batch:
            x = x.unsqueeze(0)
            
        batch_size = x.shape[0]
        
        # Base output from learnable linear weights
        output = x @ self.weight_matrix.T + self.bias
            
        # Pre-compute all monomial powers for efficiency
        x_powers = {degree: torch.pow(x, degree) for degree in range(1, self.max_degree + 1)}
        
        # Polynomial output evaluated fully via tensor operations to preserve gradient tracking
        poly_outputs = []
        for j in range(self.output_dim):
            j_out = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
            for i in range(self.input_dim):
                poly = self.polynomial_weights[j][i]
                poly_val = torch.zeros(batch_size, device=x.device, dtype=x.dtype)
                
                for m_idx, (monomial, _) in enumerate(poly.terms.items()):
                    key = f"w_{j}_{i}_{m_idx}"
                    coeff = self.poly_coeffs[key]

                    if not monomial.variables:
                        poly_val = poly_val + coeff
                    else:
                        term_val = torch.ones(batch_size, device=x.device, dtype=x.dtype)
                        for var, exp in monomial.variables:
                            var_idx = int(var.name[1:]) if var.name.startswith('x') else 0
                            if exp in x_powers:
                                term_val = term_val * x_powers[exp][:, var_idx]
                            else:
                                term_val = term_val * (x[:, var_idx] ** exp)
                        poly_val = poly_val + coeff * term_val
                        
                j_out = j_out + poly_val * x[:, i]
            poly_outputs.append(j_out)
            
        poly_tensor = torch.stack(poly_outputs, dim=1)
        output = output + poly_tensor
        
        if self.use_batchnorm:
            output = self.batch_norm(output)
            
        if self.dropout_rate > 0:
            output = self.dropout(output)
        
        self.last_input = x.detach().clone()
        self.last_output = output.detach().clone()
        
        return output if is_batch else output[0]

    def clip_gradients(self, max_norm: float = 1.0) -> None:
        """
        Clip gradients of the layer's parameters to prevent exploding gradients.
        """
        nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def check_gradients(self, x: Optional[torch.Tensor] = None) -> bool:
        """
        Verify gradients using PyTorch's built-in gradcheck.
        """
        original_dtype = next(self.parameters()).dtype
        if x is None:
            x = torch.randn(2, self.input_dim, dtype=torch.float64, requires_grad=True)
        else:
            x = x.to(torch.float64).requires_grad_(True)
            
        # Temporarily cast layer to float64 for precise finite differences
        self.double()
        try:
            test = torch.autograd.gradcheck(self.forward, (x,), eps=1e-6, atol=1e-4)
            return test
        finally:
            self.to(original_dtype)
            x = x.to(original_dtype)

    def _sync_polynomials(self) -> None:
        """Syncs learned PyTorch parameters back to the symbolic Polynomial objects."""
        if self._is_synced:
            return
        for j in range(self.output_dim):
            for i in range(self.input_dim):
                poly = self.polynomial_weights[j][i]
                for m_idx, monomial in enumerate(list(poly.terms.keys())):
                    key = f"w_{j}_{i}_{m_idx}"
                    poly.terms[monomial] = self.poly_coeffs[key].item()
        self._is_synced = True

    def get_polynomial_weights(self) -> List[List[Polynomial]]:
        """Get the polynomial weights"""
        self._sync_polynomials()
        return self.polynomial_weights
    
    def set_polynomial_weights(self, weights: List[List[Polynomial]]) -> None:
        """Set polynomial weights and update PyTorch parameters"""
        self.polynomial_weights = weights
        self.poly_coeffs.clear()
        for j in range(self.output_dim):
            for i in range(self.input_dim):
                poly = self.polynomial_weights[j][i]
                for m_idx, (monomial, coeff) in enumerate(poly.terms.items()):
                    key = f"w_{j}_{i}_{m_idx}"
                    val = float(np.real(coeff))
                    self.poly_coeffs[key] = nn.Parameter(torch.tensor(val, dtype=torch.float32))
    
    def get_weight_polynomials(self, i: int, j: int) -> Polynomial:
        """Get polynomial weight for connection from input i to output j"""
        self._sync_polynomials()
        return self.polynomial_weights[j][i]
    
    def symbolic_forward(self) -> List[Polynomial]:
        """
        Get symbolic representation of output as polynomials.

        Returns:
            List of output polynomials (one per output dimension)
        """
        self._sync_polynomials()
        output_polys = []

        for j in range(self.output_dim):
            result = Polynomial({})
            for i in range(self.input_dim):
                # Multiply polynomial weight by input variable
                x_poly = Polynomial.variable(self.variables[i])
                product = self.polynomial_weights[j][i] * x_poly
                result = result + product
            output_polys.append(result)

        return output_polys

    def to_dense_weights(self) -> np.ndarray:
        """
        Convert to dense weight matrix (evaluating polynomials at x=0).
        """
        self._sync_polynomials()
        weights = np.zeros((self.output_dim, self.input_dim))

        for j in range(self.output_dim):
            for i in range(self.input_dim):
                context = {v: 0.0 for v in self.variables}
                weights[j, i] = float(self.polynomial_weights[j][i].evaluate(context))

        return weights
    
    def get_polynomial_info(self) -> Dict:
        """Get information about the polynomials"""
        self._sync_polynomials()
        info = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'max_degree': self.max_degree,
            'polynomial_type': self.polynomial_type,
            'total_polynomials': self.input_dim * self.output_dim,
            'polynomial_details': []
        }
        
        for j in range(self.output_dim):
            for i in range(self.input_dim):
                poly = self.polynomial_weights[j][i]
                info['polynomial_details'].append({
                    'from': i,
                    'to': j,
                    'terms': len(poly.terms),
                    'degree': poly.degree(),
                    'string': str(poly)
                })
        
        return info
    
    def __repr__(self) -> str:
        return (f"PolynomialLayer(in={self.input_dim}, out={self.output_dim}, "
                f"degree={self.max_degree}, type={self.polynomial_type})")


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """
        Args:
            patience: How many epochs to wait after last time validation loss improved.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return self.early_stop
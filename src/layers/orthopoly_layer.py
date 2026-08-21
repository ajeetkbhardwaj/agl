"""
OrthoPolyLayer
Generalized orthogonal polynomial layer supporting multiple families.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn

from ..alggeom.polynomial import Polynomial, Monomial, Variable


@dataclass
class OrthoPolyFamily:
    name: str
    interval: Tuple[float, float]
    is_bounded: bool
    A: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    h_norms: torch.Tensor


class OrthoPolyLayer(nn.Module):
    """Orthonormal polynomial layer.

    - Evaluates polynomials up to degree `max_degree` per input feature.
    - Parameterization: low-rank factorization via `edge_weights` and
      `cheby_coeffs` (named for historical reasons).

    Shapes:
    - input: (batch, input_dim)
    - cheby_coeffs: (rank, input_dim, D+1)
    - edge_weights: (output_dim, input_dim, rank)
    - output: (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_degree: int = 5,
        rank: int = 4,
        use_bias: bool = True,
        basis_type: str = "chebyshev_T",
        interaction_mode: str = "additive",
        device=None,
        dtype=torch.float32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_degree = max_degree
        self.rank = rank
        self.use_bias = use_bias
        self.interaction_mode = interaction_mode

        # Build family and buffers
        self.family = self._build_family(basis_type, self.max_degree)
        self.register_buffer("A", self.family.A)
        self.register_buffer("B", self.family.B)
        self.register_buffer("C", self.family.C)
        self.register_buffer("h_norms", self.family.h_norms)
        self.register_buffer("interval", torch.tensor(self.family.interval))

        # Buffers for adaptive domain normalization.
        self.register_buffer("running_min", torch.zeros(input_dim))
        self.register_buffer("running_max", torch.zeros(input_dim))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.momentum = 0.1

        # Basis transform matrix computed from recurrence.
        # Rows correspond to polynomial coefficients in the standard monomial basis.
        M = self._compute_basis_transform()
        self.register_buffer("basis_transform", M)

        # In multiplicative mode, we MUST include P_0(x) = 1
        self.num_poly_terms = self.max_degree + 1

        # Parameters: low-rank factorization
        # cheby_coeffs: (rank, input_dim, num_poly_terms)
        self.cheby_coeffs = nn.Parameter(
            torch.randn(self.rank, self.input_dim, self.num_poly_terms, dtype=dtype, device=device)
        )

        if self.interaction_mode == 'additive':
            # linear_weight: (output_dim, input_dim, rank)
            self.linear_weight = nn.Parameter(
                torch.randn(self.output_dim, self.input_dim, self.rank, dtype=dtype, device=device)
            )
        else:  # 'multiplicative'
            # Segre Embedding / CP Decomposition: mixes the rank-1 cross terms globally
            # Shape: (output_dim, rank)
            self.linear_weight = nn.Parameter(
                torch.randn(self.output_dim, self.rank, dtype=dtype, device=device)
            )

        # bias per output
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.output_dim, dtype=dtype, device=device))
        else:
            self.register_parameter('bias', None)

        self._initialize_weights()

    def _initialize_weights(self):
        with torch.no_grad():
            for d_idx in range(self.num_poly_terms):
                std_d = 1.0 / (d_idx + 1.0)
                nn.init.normal_(self.cheby_coeffs[:, :, d_idx], mean=0.0, std=std_d)
        nn.init.xavier_uniform_(self.linear_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _build_family(self, name: str, D: int) -> OrthoPolyFamily:
        """Create OrthoPolyFamily for supported families using vectorized arange."""
        n = torch.arange(D + 1, dtype=torch.float32)
        if name == "chebyshev_T":
            A = torch.full((D + 1,), 2.0)
            B = torch.zeros(D + 1)
            C = torch.ones(D + 1)
            h = torch.ones(D + 1) * (torch.pi / 2.0)
            h[0] = torch.pi
            interval = (-1.0, 1.0)
            bounded = True
        elif name == "chebyshev_U":
            A = torch.full((D + 1,), 2.0)
            B = torch.zeros(D + 1)
            C = torch.ones(D + 1)
            h = torch.ones(D + 1) * (torch.pi / 2.0)
            interval = (-1.0, 1.0)
            bounded = True
        elif name == "legendre":
            # A_n = (2n+1)/(n+1), C_n = n/(n+1)
            A = (2.0 * n + 1.0) / (n + 1.0)
            B = torch.zeros(D + 1)
            C = n / (n + 1.0)
            h = 2.0 / (2.0 * n + 1.0)
            interval = (-1.0, 1.0)
            bounded = True
        elif name == "hermite":
            A = torch.full((D + 1,), torch.sqrt(torch.tensor(2.0)))
            B = torch.zeros(D + 1)
            C = 2.0 * n
            # h_n = 2^n n! sqrt(pi)
            h = torch.exp(n * torch.log(torch.tensor(2.0)) + torch.lgamma(n + 1.0) + 0.5 * torch.log(torch.tensor(torch.pi)))
            interval = (float('-inf'), float('inf'))
            bounded = False
        elif name == "laguerre":
            A = -1.0 / (n + 1.0)
            B = (2.0 * n + 1.0) / (n + 1.0)
            C = n / (n + 1.0)
            h = torch.ones(D + 1)
            interval = (0.0, float('inf'))
            bounded = False
        else:
            raise ValueError(f"Unsupported family: {name}")

        # Ensure initial P1 uses canonical x (A0=1, B0=0) unless family explicitly needs otherwise
        A = A.clone()
        B = B.clone()
        A[0] = 1.0
        B[0] = 0.0

        return OrthoPolyFamily(name=name, interval=interval, is_bounded=bounded, A=A, B=B, C=C, h_norms=h)

    def _compute_basis_transform(self) -> torch.Tensor:
        """Compute basis transform matrix M (D+1 x D+1) from recurrence.

        Note: we store M and avoid storing inv(M.t()) for numerical stability.
        """
        D = self.max_degree
        M = torch.zeros((D + 1, D + 1), dtype=self.family.A.dtype)
        M[0, 0] = 1.0
        if D >= 1:
            # First-order polynomial P1(x) should be x (coefficient 1) by construction
            M[1, 1] = 1.0
            M[1, 0] = 0.0
        for d in range(1, D):
            x_shift = torch.roll(M[d], shifts=1, dims=0)
            x_shift[0] = 0.0
            M[d + 1] = self.family.A[d] * x_shift + self.family.B[d] * M[d] - self.family.C[d] * M[d - 1]
        return M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the polynomial layer on inputs.

        x: shape (batch, input_dim)
        returns: (batch, output_dim)
        """
        D = self.max_degree

        # Adaptive domain normalization
        if self.training:
            if self.num_batches_tracked == 0:
                self.running_min.copy_(torch.min(x, dim=0)[0].data)
                self.running_max.copy_(torch.max(x, dim=0)[0].data)
            else:
                self.running_min.mul_(1 - self.momentum).add_(self.momentum * torch.min(x, dim=0)[0].data)
                self.running_max.mul_(1 - self.momentum).add_(self.momentum * torch.max(x, dim=0)[0].data)
            self.num_batches_tracked += 1

        span = self.running_max - self.running_min
        span = torch.where(span < 1e-6, torch.ones_like(span), span)
        x_norm = 2.0 * (x - self.running_min) / span - 1.0
        x_norm = x_norm.clamp(min=-1.0, max=1.0)

                # Build recurrence P_d(x) for d=0..D as a list to avoid inplace autograd conflicts.
        P_list = [torch.ones((x.shape[0], self.input_dim), dtype=x.dtype, device=x.device)]
        if D >= 1:
            P_list.append((self.A[0] * x_norm) + self.B[0])
        for n in range(1, D):
            coeff_ax = (self.A[n] * x_norm) + self.B[n]
            P_next = coeff_ax * P_list[n] - self.C[n] * P_list[n - 1]
            P_list.append(P_next)
        P = torch.stack(P_list, dim=-1)  # Shape (batch, input_dim, num_poly_terms)

        if self.interaction_mode == 'additive':
            # edge_cheby: (output_dim, input_dim, D+1)
            # einsum pattern: 'o i r, r i d -> o i d'
            edge_cheby = torch.einsum('o i r, r i d -> o i d', self.linear_weight, self.cheby_coeffs)
            # Evaluate: 'b i d, o i d -> b o'
            out = torch.einsum('b i d, o i d -> b o', P, edge_cheby)
        else:  # 'multiplicative'
            # b=batch, i=input, d=degree, r=rank
            poly_features = torch.einsum('bid,rid->bri', P, self.cheby_coeffs)
            # THE SEGRE EMBEDDING: Multiply features across `input_dim`
            cross_terms = torch.prod(poly_features, dim=2)  # [batch, rank]
            # o=output
            out = torch.einsum('br,or->bo', cross_terms, self.linear_weight)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out

    
    def get_polynomial_weights(self):
        """Return list of list of Polynomial objects of shape (output_dim x input_dim).
        
        Each polynomial constructed from the basis expansion.
        """
        D = self.max_degree
        if self.interaction_mode == 'multiplicative':
            raise NotImplementedError("Symbolic extraction for multiplicative mode (Segre embedding) is not supported yet.")

        # compute collapsed coefficients: (output_dim, input_dim, D+1)
        coeffs = torch.einsum('o i r, r i d -> o i d', self.linear_weight, self.cheby_coeffs).detach().cpu()
        polys = []
        for o in range(self.output_dim):
            row = []
            for i in range(self.input_dim):
                var = Variable(f'x{i}')
                terms = {}
                # Convert from family basis -> standard monomial basis using basis_transform.
                for d in range(0, D + 1):

                    c = float(coeffs[o, i, d])
                    if abs(c) < 1e-12:
                        continue
                    row_std = (c * self.basis_transform[d]).tolist()
                    for k, val in enumerate(row_std):
                        if abs(val) < 1e-12:
                            continue
                        monom = Monomial(()) if k == 0 else Monomial(((var, k),))
                        terms[monom] = terms.get(monom, 0.0) + float(val)
                if i == 0 and self.bias is not None:
                    bias_val = float(self.bias[o].item())
                    if abs(bias_val) > 1e-12:
                        terms[Monomial(())] = terms.get(Monomial(()), 0.0) + bias_val
                poly = Polynomial(terms)
                row.append(poly)
            polys.append(row)
        return polys

    def set_polynomial_weights(self, polys):
        """Set layer parameters from provided polynomial objects.

        Solve for chebyshev coefficients using basis_transform.t() via `torch.linalg.solve`
        and perform low-rank SVD factorization with shapes adjusted as required.
        """
        if self.interaction_mode == 'multiplicative':
            raise NotImplementedError("Symbolic setting for multiplicative mode (Segre embedding) is not supported yet.")

        D = self.max_degree
        # Build target coefficient tensor c_std: (output_dim, input_dim, D+1)
        c_std = torch.zeros((self.output_dim, self.input_dim, D + 1), dtype=self.cheby_coeffs.dtype, device=self.cheby_coeffs.device)
        for o in range(self.output_dim):
            for i in range(self.input_dim):
                p = polys[o][i]
                var = Variable(f'x{i}')
                for monomial, coeff in p.terms.items():
                    # monomial variables like 'x0': k
                    if not monomial.variables:
                        deg = 0
                    else:
                        # Expect single-variable monomials for this layer
                        deg = monomial.variables[0][1]
                    c_std[o, i, deg] = coeff

        # Solve for family-domain coefficients: a_std = c_cheb @ M



        # Solve for family-domain coefficients: a_std = c_cheb @ M
        # => c_cheb = a_std @ M^{-1}. Use solve with M.T for correct batched orientation.
        M_t = self.basis_transform.t().to(self.cheby_coeffs.dtype)
        flat = c_std.reshape(-1, D + 1)
        c_cheb_flat = torch.linalg.solve(M_t, flat.t()).t()

        c_cheb = c_cheb_flat.reshape(self.output_dim, self.input_dim, D + 1)

        # Build batched matrix A[i] = c_cheb[:, i, :] for each input feature i
        A = torch.zeros((self.input_dim, self.output_dim, D + 1), dtype=self.cheby_coeffs.dtype, device=self.cheby_coeffs.device)
        for o in range(self.output_dim):
            for i in range(self.input_dim):
                A[i, o, :] = c_cheb[o, i, :]

        # Batched SVD over input features to match low-rank factorization.
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        actual_rank = min(self.rank, S.shape[1])
        scale = torch.sqrt(torch.clamp(S[:, :actual_rank], min=0.0))

        with torch.no_grad():
            self.linear_weight.data.zero_()
            self.cheby_coeffs.data.zero_()
            if self.bias is not None:
                self.bias.data.zero_()

            self.linear_weight.data[:, :, :actual_rank] = (U[:, :, :actual_rank] * scale.unsqueeze(1)).permute(1, 0, 2)
            cc = (Vh[:, :actual_rank, :] * scale.unsqueeze(-1)).permute(1, 0, 2)
            self.cheby_coeffs.data[:actual_rank, :, :] = cc.to(self.cheby_coeffs.device)

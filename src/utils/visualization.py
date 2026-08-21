"""
Visualization Utilities for Polynomial Neural Networks
=======================================================
Tools for visualizing algebraic varieties and training progress.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _evaluate_xy(polynomial, xv: float, yv: float,
                 variables: Optional[List[Any]] = None) -> float:
    """Evaluate a polynomial at (x, y).

    Routing order:
    1. Objects exposing a ``variables()`` method (alggeom.Polynomial) are
       evaluated with proper {Variable: value} keys.
    2. Other objects with ``evaluate`` get string keys {'x':.., 'y':..}.
    3. Anything else is treated as a callable f({'x':.., 'y':..}).
    """
    if callable(getattr(polynomial, 'variables', None)):
        vars_list = variables if variables is not None else polynomial.variables()
        val = polynomial.evaluate(dict(zip(vars_list, (xv, yv))))
    elif hasattr(polynomial, 'evaluate'):
        val = polynomial.evaluate({'x': xv, 'y': yv})
    else:
        val = polynomial({'x': xv, 'y': yv})
    return val.real if isinstance(val, complex) else val


def _eval_on_grid(polynomial, X: np.ndarray, Y: np.ndarray,
                  variables: Optional[List[Any]] = None) -> np.ndarray:
    """Evaluate a polynomial on a meshgrid; failures become NaN."""
    Z = np.full_like(X, np.nan, dtype=float)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = _evaluate_xy(polynomial, X[i, j], Y[i, j], variables)
            except Exception:
                Z[i, j] = np.nan
    return Z


class VarietyVisualizer:
    """Visualize algebraic varieties in 2D and 3D."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
    
    def plot_variety_2d(
        self,
        polynomial,
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
        resolution: int = 100,
        ax: Optional[plt.Axes] = None,
        variables: Optional[List[Any]] = None
    ) -> plt.Figure:
        """Plot a 2D algebraic variety (curve)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = _eval_on_grid(polynomial, X, Y, variables)

        # Plot contour where polynomial = 0
        ax.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Algebraic Variety (2D)')
        ax.grid(True, alpha=0.3)

        return fig

    def plot_variety_3d(
        self,
        polynomial,
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2),
        z_range: Tuple[float, float] = (-2, 2),
        resolution: int = 50,
        ax: Optional[Axes3D] = None,
        variables: Optional[List[Any]] = None
    ) -> plt.Figure:
        """Plot the surface z = f(x, y) spanned by a polynomial."""
        fig = plt.figure(figsize=self.figsize)

        if ax is None:
            ax = fig.add_subplot(111, projection='3d')

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = _eval_on_grid(polynomial, X, Y, variables)

        # Plot surface z = f(x, y); its intersection with z = 0 is the variety
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
        ax.contour3D(X, Y, Z, levels=[0], colors='red', linewidths=3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('Surface z = f(x, y); red curve: f = 0')

        return fig
    
    def plot_point_on_variety(
        self,
        polynomial,
        point: np.ndarray,
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
        ax: Optional[plt.Axes] = None,
        variables: Optional[List[Any]] = None
    ) -> plt.Figure:
        """Plot variety with a specific point highlighted."""
        fig = self.plot_variety_2d(polynomial, x_range, y_range, ax=ax,
                                   variables=variables)
        ax = fig.axes[0]

        # Check if point is on variety
        try:
            value = _evaluate_xy(polynomial, point[0], point[1], variables)
            on_variety = abs(value) < 0.1
        except Exception:
            on_variety = False

        color = 'red' if on_variety else 'gray'
        ax.scatter(point[0], point[1], c=color, s=100, zorder=5)

        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color='blue', lw=2, label='Variety'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=10, label=f'Point (on variety: {on_variety})'),
        ]
        ax.legend(handles=handles)

        return fig


class TrainingVisualizer:
    """Visualize training progress.

    Histories may be objects with attribute-style fields
    (``train_losses``, ``val_losses``, ...) or plain dictionaries as
    returned by ``OrthoPolyNetwork.fit`` (``{'loss': [...], ...}``).
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 4)):
        self.figsize = figsize

    @staticmethod
    def _get(history: Any, name: str):
        if isinstance(history, dict):
            return history.get(name)
        return getattr(history, name, None)

    @staticmethod
    def _series(history: Any, *names: str) -> Optional[List[float]]:
        for name in names:
            data = TrainingVisualizer._get(history, name)
            if data:
                return data
        return None

    def plot_training_history(
        self,
        history: Any,
        metrics: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot training history."""
        if metrics is None:
            metrics = ['loss']
            if self._series(history, 'train_accuracies', 'val_accuracies',
                            'accuracy') is not None:
                metrics.append('accuracy')

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics,
                                 figsize=(self.figsize[0] * n_metrics / 2,
                                          self.figsize[1]))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric == 'loss':
                train = self._series(history, 'train_losses', 'loss')
                val = self._series(history, 'val_losses', 'val_loss')
                if train is not None:
                    ax.plot(range(len(train)), train, label='Train Loss')
                if val is not None:
                    ax.plot(range(len(val)), val, label='Val Loss')
                ax.set_ylabel('Loss')
            elif metric == 'accuracy':
                train = self._series(history, 'train_accuracies', 'accuracy')
                val = self._series(history, 'val_accuracies', 'val_accuracy')
                if train is not None:
                    ax.plot(range(len(train)), train, label='Train Acc')
                if val is not None:
                    ax.plot(range(len(val)), val, label='Val Acc')
                ax.set_ylabel('Accuracy')

            ax.set_xlabel('Epoch')
            ax.set_title(f'{metric.capitalize()} over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_learning_rate_schedule(
        self,
        history: Any
    ) -> plt.Figure:
        """Plot learning rate schedule."""
        fig, ax = plt.subplots(figsize=self.figsize)

        lrs = self._series(history, 'learning_rates', 'lr')
        if lrs is not None:
            ax.plot(range(len(lrs)), lrs, color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        return fig


class PolynomialVisualizer:
    """Visualize polynomial functions and their properties."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
    
    def plot_polynomial_1d(
        self,
        polynomial,
        x_range: Tuple[float, float] = (-5, 5),
        resolution: int = 200,
        ax: Optional[plt.Axes] = None,
        variables: Optional[List[Any]] = None
    ) -> plt.Figure:
        """Plot a 1D polynomial."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = []
        for xi in x:
            try:
                if callable(getattr(polynomial, 'variables', None)):
                    vars_list = variables if variables is not None else polynomial.variables()
                    val = polynomial.evaluate({vars_list[0]: xi})
                elif hasattr(polynomial, 'evaluate'):
                    val = polynomial.evaluate({'x': xi})
                else:
                    val = polynomial({'x': xi})
                y.append(val.real if isinstance(val, complex) else val)
            except Exception:
                y.append(np.nan)
        y = np.array(y)
        
        ax.plot(x, y, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('p(x)')
        ax.set_title(f'Polynomial: {polynomial}')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_polynomial_2d_heatmap(
        self,
        polynomial,
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2),
        resolution: int = 100,
        ax: Optional[plt.Axes] = None,
        variables: Optional[List[Any]] = None
    ) -> plt.Figure:
        """Plot 2D polynomial as heatmap."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.get_figure()

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = _eval_on_grid(polynomial, X, Y, variables)
        
        im = ax.imshow(Z, extent=[*x_range, *y_range], origin='lower', cmap='RdBu_r')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Polynomial Heatmap')
        plt.colorbar(im, ax=ax)
        
        return fig


def plot_decision_boundary(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 100,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot decision boundary for a classifier."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(mesh_points)
    if hasattr(Z, 'detach'):            # torch.Tensor -> numpy
        Z = Z.detach().numpy()
    
    # Safely handle outputs
    if Z.ndim > 1:
        if Z.shape[1] > 1:
            Z = Z[:, 1]  # For binary classification with 2 outputs
        elif Z.shape[1] == 1:
            Z = Z.flatten()  # For single probability output
    Z = Z.reshape(xx.shape)
    
    # Plot contour
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu_r', alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)
    
    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu_r', edgecolors='k', s=50)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Boundary')
    
    return fig


def plot_groebner_basis_convergence(
    basis_sizes: List[int],
    computation_times: List[float],
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """Plot Gröbner basis computation statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Basis size over iterations
    ax1.plot(basis_sizes, 'b-o')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Basis Size')
    ax1.set_title('Gröbner Basis Size')
    ax1.grid(True, alpha=0.3)
    
    # Computation time
    ax2.plot(computation_times, 'r-o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Computation Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
import numpy as np
import torch
import sympy as sp

from src.nn.poly_nn import PolynomialNeuralNetwork, create_groebner_pnn
from src.nn.orthopoly_nn import OrthoPolyNetwork


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def test_pnn_fit_and_exact_distillation():
    torch.manual_seed(7); np.random.seed(7)
    X = np.random.randn(240, 2).astype(np.float64)
    y = (X[:, :1] ** 2).astype(np.float64)
    pnn = PolynomialNeuralNetwork(input_dim=2, output_dim=1,
                                  hidden_dims=[16], polynomial_degree=3,
                                  activation='none')
    history = pnn.fit(X, y, epochs=150, learning_rate=0.01, verbose=False)
    assert isinstance(history, dict)

    pred = pnn.predict(X)
    assert pred.shape == (240, 1)
    assert _r2(y, pred) > 0.90

    report = pnn.evaluate(X, y)
    assert isinstance(report, dict) and "r2" in report

    eq = pnn.get_global_equation(["a", "b"], threshold=1e-9, round_to=None)
    f_eq = sp.lambdify(sp.symbols("a b"), eq, "numpy")
    sym_vals = np.asarray(f_eq(X[:, 0], X[:, 1]), dtype=float).reshape(-1, 1)
    # network params are float32; exact distillation agrees to float32 noise
    assert np.abs(sym_vals - pred).max() < 5e-3
    assert np.median(np.abs(sym_vals - pred)) < 1e-5


def test_orthopoly_network_additive_regression():
    torch.manual_seed(7); np.random.seed(7)
    u = np.random.uniform(-0.95, 0.95, size=(300, 1)).astype(np.float64)
    y = (u ** 2).astype(np.float64)
    net = OrthoPolyNetwork(input_dim=1, output_dim=1, hidden_dims=[16],
                           max_degree=4, rank=3, basis_type="chebyshev_first",
                           interaction_mode="additive")
    history = net.fit(u, y, epochs=200, learning_rate=0.01, verbose=False)
    assert isinstance(history, dict) and len(history.get("loss", [])) > 0

    pred = net.predict(u)
    assert pred.shape == y.shape
    assert _r2(y, pred) > 0.90


def test_orthopoly_network_multiplicative_smoke():
    torch.manual_seed(7); np.random.seed(7)
    u = np.random.uniform(-0.95, 0.95, size=(120, 2)).astype(np.float64)
    y = (u[:, :1] * u[:, 1:]).astype(np.float64)
    net = OrthoPolyNetwork(input_dim=2, output_dim=1, hidden_dims=[8],
                           max_degree=3, rank=3, basis_type="chebyshev_first",
                           interaction_mode="multiplicative")
    net.fit(u, y, epochs=30, learning_rate=0.01, verbose=False)
    assert net.predict(u).shape == y.shape


def test_groebner_pnn_train_step_projects_to_normal_form():
    torch.manual_seed(7); np.random.seed(7)
    gp = create_groebner_pnn(
        input_dim=2, output_dim=1, constraint_ideals=[["x0 - x1"], []],
        hidden_dims=[6], polynomial_degree=2, activation='tanh')
    opt = torch.optim.Adam(gp.parameters(), lr=1e-3)
    X_t = torch.randn(32, 2)
    y_t = torch.randn(32, 1)
    loss = gp.train_step(X_t, y_t, opt)
    assert np.isfinite(loss)

    # projection after the optimizer step forces w[0, 0] to exactly zero:
    # the normal form of any weight vector modulo <x0 - x1> has no x0 term
    flat = gp.layers[0].weight_matrix.detach().numpy().ravel()
    assert flat[0] == 0.0

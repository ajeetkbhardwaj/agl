import numpy as np
import pytest

from src.utils.symbolic import (
    Monomial, Polynomial, Variable,
    create_polynomial_from_coefficients,
    parse_polynomial_string, polynomial_gcd, polynomial_lcm,
    polynomial_to_array, SymbolicPolynomialSystem,
)
from src.utils.visualization import TrainingVisualizer, VarietyVisualizer


@pytest.fixture(scope="module")
def xy():
    return Variable("x"), Variable("y")


def test_variable_and_monomial_basics(xy):
    x, y = xy
    assert hash(x) == hash(Variable("x"))
    m_xx = Monomial(((x, 2),))
    m_y = Monomial(((y, 1),))
    prod = m_xx * m_y
    assert prod.degree() == 3
    assert m_xx.divides(prod)
    assert not m_xx.divides(m_y)
    # div(self, other) returns other / self
    assert m_xx.div(prod) == m_y


def test_polynomial_arithmetic(xy):
    x, y = xy
    a = parse_polynomial_string("x^2 + y", [x, y])
    b = parse_polynomial_string("x - y", [x, y])
    prod = a * b
    for px, py in [(2.0, 1.0), (-1.5, 0.4)]:
        expected = (px**2 + py) * (px - py)
        assert prod.evaluate({x: px, y: py}) == pytest.approx(expected)
    diff = (a + b) - b
    for px, py in [(1.0, 2.0), (3.3, -2.1)]:
        assert diff.evaluate({x: px, y: py}) == pytest.approx(
            a.evaluate({x: px, y: py}), abs=1e-9)


def test_divide_by_exact(xy):
    x, _ = xy
    num = parse_polynomial_string("x^2 - 1", [x])
    den = parse_polynomial_string("x - 1", [x])
    quotient, remainder = num.divide_by(den)
    assert remainder.is_zero()
    assert quotient.evaluate({x: 4.0}) == pytest.approx(5.0)


def test_leading_monomial_and_coefficient(xy):
    x, y = xy
    f = parse_polynomial_string("3*x^2 + 2*x*y + 5*y^3", [x, y])
    lm = f.leading_monomial(order="grevlex")
    assert lm.degree() == 3
    assert f.leading_coefficient(order="grevlex") == pytest.approx(5.0)


def test_is_one_is_zero_copy_variables(xy):
    x, _ = xy
    one = Polynomial.constant(1.0)
    zero = Polynomial({})
    p = parse_polynomial_string("x", [x])
    assert one.is_one()
    assert zero.is_zero()
    assert not p.is_zero()
    copied = one.copy()
    assert copied is not one and copied.is_one()
    assert x in p.variables()


def test_univariate_gcd_lcm(xy):
    x, _ = xy
    f = parse_polynomial_string("x^2 - 1", [x])
    g = parse_polynomial_string("x - 1", [x])
    gcd = polynomial_gcd(f, g)
    lcm = polynomial_lcm(f, g)
    assert gcd.degree() == 1
    assert lcm.degree() == 2
    assert gcd.evaluate({x: 1.0}) == pytest.approx(0.0, abs=1e-7)
    # every root of f must also be a root of the lcm
    for v in [1.0, -1.0]:
        assert lcm.evaluate({x: v}) == pytest.approx(0.0, abs=1e-7)


def test_multivariate_gcd_rejected(xy):
    x, y = xy
    f = parse_polynomial_string("x*y - x", [x, y])
    with pytest.raises(NotImplementedError):
        polynomial_gcd(f, f)


def test_coefficient_roundtrip(xy):
    x, y = xy
    coeffs = np.array([[0.0, 2.0], [3.0, 0.0]])   # 2y + 3x
    poly = create_polynomial_from_coefficients(coeffs, [x, y])
    back = polynomial_to_array(poly, [x, y], max_degree=1)
    assert np.allclose(back, coeffs)


def test_symbolic_system_evaluate_and_jacobian(xy):
    x, y = xy
    system = SymbolicPolynomialSystem([x, y])
    system.add_polynomial(parse_polynomial_string("x^2 + y", [x, y]))
    point = np.array([2.0, 1.0])
    vals = system.evaluate(point)
    assert np.allclose(vals, [5.0])
    jac = system.jacobian(point)
    assert np.allclose(jac, [[4.0, 1.0]])


def test_variety_visualizer_smoke(xy):
    x, y = xy
    circle = parse_polynomial_string("x^2 + y^2 - 1", [x, y])
    fig = VarietyVisualizer(figsize=(6, 6)).plot_variety_2d(
        circle, x_range=(-2, 2), y_range=(-2, 2), resolution=60,
        variables=[x, y])
    assert fig.axes[0].get_title() != ""
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_training_visualizer_dict_history():
    history = {"loss": [3.0, 2.0, 1.5, 1.2],
               "val_loss": [3.2, 2.4, 2.0, 1.9]}
    fig = TrainingVisualizer().plot_training_history(history)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)

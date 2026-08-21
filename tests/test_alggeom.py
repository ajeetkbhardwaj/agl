import numpy as np
import pytest

from src.alggeom.polynomial import Monomial, Polynomial, Variable
from src.alggeom.groebnerbasis import GroebnerBasis
from src.alggeom.ideal import Ideal
from src.alggeom.algvariety import AffineSpace, Hypersurface


@pytest.fixture(scope="module")
def xy():
    return Variable("x"), Variable("y")


def test_variable_equality_and_hash(xy):
    x, y = xy
    assert x == Variable("x")
    assert hash(x) == hash(Variable("x"))
    assert x != y
    assert str(x) == "x"


def test_monomial_arithmetic(xy):
    x, y = xy
    m_xx = Monomial(((x, 2),))
    m_y = Monomial(((y, 1),))
    prod = m_xx * m_y
    assert prod.degree() == 3
    assert m_xx.divides(prod)
    assert not m_xx.divides(m_y)
    # div(self, other) returns other / self
    assert m_xx.div(prod) == m_y
    assert m_xx.lcm(m_y) == prod


def test_polynomial_from_string_and_evaluate(xy):
    x, y = xy
    p = Polynomial.from_string("x^2*y - 3", [x, y])
    assert p.evaluate({x: 2.0, y: 1.0}) == pytest.approx(1.0)


def test_polynomial_ring_operations(xy):
    x, y = xy
    a = Polynomial.variable(x) + Polynomial.variable(y)
    b = Polynomial.variable(x) - Polynomial.variable(y)
    diff_sq = a * b
    for px, py in [(1.7, -0.3), (2.0, 2.0), (-4.0, 0.5)]:
        expected = px**2 - py**2
        assert diff_sq.evaluate({x: px, y: py}) == pytest.approx(expected)
    sq = a**2
    assert sq.evaluate({x: 2.0, y: 3.0}) == pytest.approx(25.0)
    scaled = 3.0 * Polynomial.variable(x)
    assert scaled.evaluate({x: 2.0}) == pytest.approx(6.0)


def test_differentiate(xy):
    x, y = xy
    f = Polynomial.from_string("x^2*y", [x, y])
    dfdx = f.differentiate(x)
    assert dfdx.evaluate({x: 3.0, y: 2.0}) == pytest.approx(12.0)


def test_divide_by_exact(xy):
    x, _ = xy
    num = Polynomial.from_string("x^2 - 1", [x])
    den = Polynomial.from_string("x - 1", [x])
    quotient, remainder = num.divide_by(den)
    assert remainder.is_zero()
    recombined = (den * quotient).evaluate({x: 7.0})
    assert recombined == pytest.approx(num.evaluate({x: 7.0}))


def test_leading_term_grevlex(xy):
    x, y = xy
    f = Polynomial.from_string("3*x^2 + 2*x*y + 5*y^3", [x, y])
    lm = f.leading_monomial(order="grevlex")
    assert lm.degree() == 3
    assert f.leading_coefficient(order="grevlex") == pytest.approx(5.0)


def test_zero_one_helpers(xy):
    one = Polynomial.constant(1.0)
    zero = Polynomial({})
    assert zero.is_zero()
    assert one.is_constant()
    assert zero.is_constant()   # the zero polynomial is the constant 0
    assert one.copy() is not one


def test_groebner_basis_compute_membership(xy):
    x, y = xy
    f1 = Polynomial.from_string("x^2 - y", [x, y])
    f2 = Polynomial.from_string("y^2 - x", [x, y])
    gb = GroebnerBasis([f1, f2])
    gb.compute()
    basis = gb.basis
    assert len(basis) > 0
    assert gb.is_groebner_basis()
    in_ideal, _ = gb.ideal_membership(f1 * f2)
    assert in_ideal is True or in_ideal is True  # truthy check below
    assert bool(in_ideal)
    one_in, _ = gb.ideal_membership(Polynomial.constant(1.0))
    assert not bool(one_in)


def test_ideal_contains(xy):
    x, y = xy
    gens = [Polynomial.from_string("x", [x, y]),
            Polynomial.from_string("y", [x, y])]
    I = Ideal(gens)
    member = gens[0] + gens[1]
    assert I.contains(member)
    assert not I.contains(Polynomial.constant(1.0))
    product = I.product(I)
    assert isinstance(product, Ideal)


def test_hypersurface_constraint_violation(xy):
    x, y = xy
    circle = Hypersurface(Polynomial.from_string("x^2 + y^2 - 1", [x, y]))
    on_curve = circle.constraint_violations(np.array([1.0, 0.0]))
    off_curve = circle.constraint_violations(np.array([2.0, 2.0]))
    assert abs(float(np.ravel(on_curve)[0])) < 1e-12
    assert float(np.ravel(off_curve)[0]) > 1.0


def test_affine_space_dimension():
    space = AffineSpace(3, ["a", "b", "c"])
    assert space.dimension() == 3

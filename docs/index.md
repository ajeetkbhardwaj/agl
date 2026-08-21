# Algebraic Geometric Learning

A library that fuses Algebraic Geometry with Neural Networks i.e [**commutative algebra]** with **polynomial neural
networks**: Groebner bases, ideals and algebraic varieties act as hard
structural constraints on learning, while orthogonal polynomial layers
(Chebyshev, Legendre, Hermite, Laguerre) provide stable, exactly
symbolic function approximation.

## What is inside

| Package          | Contents                                                                                                                      |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `src.alggeom`  | Polynomials, monomial orders, Groebner bases, ideals, algebraic varieties, system solving                                     |
| `src.layers`   | `PolynomialLayer`, `OrthoPolyLayer` (additive/multiplicative), `RationalPolyLayer`, `GroebnerLayer`, `VarietyLayer` |
| `src.nn`       | `PolynomialNeuralNetwork` with exact symbolic distillation, `OrthoPolyNetwork`                                            |
| `src.training` | Loss functions (ideal membership, variety constraint), polynomial-aware optimizers                                            |
| `src.utils`    | Symbolic utilities and visualization                                                                                          |

## Documentation series

The **Algebraic Geometry**, **Neural Networks** and **Utilities** parts
develop the mathematics from definitions to working code. The six
**Tutorials** apply the library to real datasets (NASA exoplanets,
Mauna Loa CO2, SILSO sunspots, UCI concrete / power plant / airfoil)
with every result reproduced by a runnable script under
[`experiments/`](https://github.com).

## Getting started

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest          # run the test suite
```

All code blocks in the docs were executed against the real library;
the test suite keeps them honest.

from __future__ import division, print_function

from sympy import (
    Add,
    Derivative,
    Expr,
    Function,
    Mul,
    Pow,
    Rational,
    S,
    simplify,
    symbols,
)
from sympy.core.decorators import _sympifyit, call_highest_priority
from sympy.core.function import UndefinedFunction
from sympy.physics.quantum import Operator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator
from sympy.printing.latex import print_latex
from sympy.printing.pretty.stringpict import prettyForm

from acchamiltoniansandmatrices.LieMaps.Poisson import PoissonBracket

A = Function("A", commutative=False)
B = Function("B", commutative=False)
C = Function("C", commutative=False)
D = Function("D", commutative=False)

x, px, y, py = symbols("x px y py")

half = Rational(1, 2)

import pytest

single_pb_fail = [
    ((1, 2), 0),
    ((1, half), 0),
]

single_pb_pass = [
    ((S.One, half), {}, 0),
    ((A, half), {}, 0),
    ((A(x), half), {}, 0),
    ((A(x, px), half), {}, 0),
    ((S.One, half), {"coords": [x]}, 0),
    ((A, half), {"coords": [x]}, 0),
    ((A(x), half), {"coords": [x]}, 0),
    ((A(x, px), half), {"coords": [x]}, 0),
    (reversed((S.One, half)), {}, 0),
    (reversed((A, half)), {}, 0),
    (reversed((A(x), half)), {}, 0),
    (reversed((A(x, px), half)), {}, 0),
    (reversed((S.One, half)), {"coords": [x]}, 0),
    (reversed((A, half)), {"coords": [x]}, 0),
    (reversed((A(x), half)), {"coords": [x]}, 0),
    (reversed((A(x, px), half)), {"coords": [x]}, 0),
]

single_pb_doit_fail = []

single_pb_doit_pass = [
    # Brackets doit returns self - no indep coords/mom to take der.
    ((A, B), {}, PoissonBracket(A, B)),
    ((A(x, px), B), {}, PoissonBracket(A(x, px), B)),
    ((A, B(x, px)), {}, PoissonBracket(A, B(x, px))),
    ((A(x, px), B(x, px)), {}, PoissonBracket(A(x, px), B(x, px))),
    ((x ** 2, x + px), {}, PoissonBracket(x ** 2, x + px)),
    (reversed((A, B)), {}, PoissonBracket(B, A)),
    (reversed((A(x, px), B)), {}, PoissonBracket(B, A(x, px))),
    (reversed((A, B(x, px))), {}, PoissonBracket(B(x, px), A)),
    (reversed((A(x, px), B(x, px))), {}, PoissonBracket(B(x, px), A(x, px))),
    # Brackets actually performing the doit
    ((A, B), {"coords": [x]}, PoissonBracket(A, B)),
    ((A(x, px), B), {"coords": [x]}, PoissonBracket(A(x, px), B)),
    ((A, B(x, px)), {"coords": [x]}, PoissonBracket(A, B(x, px))),
    ((A(x, px), B(x, px)), {"coords": [x]}, PoissonBracket(A(x, px), B(x, px))),
    ((x ** 2, x + px), {"coords": [x], "mom": [px]}, 2 * x),
    (
        (A(x, px), B(x, y, py)),
        {"coords": [x], "mom": [px]},
        -Derivative(A(x, px), px) * Derivative(B(x, y, py), x),
    ),
    (
        (A(x, y, px), B(x, y, py)),
        {"coords": [x], "mom": [px]},
        -Derivative(A(x, y, px), px) * Derivative(B(x, y, py), x),
    ),
    (
        (A(x, px), B(x, y, py)),
        {"coords": [x, y], "mom": [px]},
        -Derivative(A(x, px), px) * Derivative(B(x, y, py), x),
    ),
    (
        (A(x, px), B(x, y, py)),
        {"coords": [x, y], "mom": [px, py]},
        -Derivative(A(x, px), px) * Derivative(B(x, y, py), x),
    ),
    (
        (A(x, y, px), B(x, y, py)),
        {"coords": [x, y], "mom": [px, py]},
        -Derivative(A(x, y, px), px) * Derivative(B(x, y, py), x)
        + Derivative(A(x, y, px), y) * Derivative(B(x, y, py), py),
    ),
    (
        (A(x, px) + B(x, y, py), C(x, px)),
        {"coords": [x, y], "mom": [px, py]},
        (Derivative(A(x, px), x) + Derivative(B(x, y, py), x)) * Derivative(C(x, px), px)
        - Derivative(A(x, px), px) * Derivative(C(x, px), x),
    ),
    (
        (A(x, px) * B(x, y, py), C(x, px)),
        {"coords": [x, y], "mom": [px, py]},
        (A(x, px) * Derivative(B(x, y, py), x) + Derivative(A(x, px), x) * B(x, y, py))
        * Derivative(C(x, px), px)
        - Derivative(A(x, px), px) * B(x, y, py) * Derivative(C(x, px), x),
    ),
    (
        (A(x, px), B(x, y, py) * C(x, px)),
        {"coords": [x, y], "mom": [px, py]},
        S.NegativeOne
        * (
            (B(x, y, py) * Derivative(C(x, px), x) + Derivative(B(x, y, py), x) * C(x, px))
            * Derivative(A(x, px), px)
            - B(x, y, py) * Derivative(C(x, px), px) * Derivative(A(x, px), x)
        ),
    ),
    (
        (A(x, px) ** 3, B(x, y, py)),
        {"coords": [x, y], "mom": [px, py]},
        -3
        * A(x, px) ** 3
        * Derivative(A(x, px), px)
        * A(x, px) ** (-1)
        * Derivative(B(x, y, py), x),
    ),
]


@pytest.mark.parametrize("input,expected", single_pb_fail)
def test_single_pb_parametrized_fail(input, expected):
    with pytest.raises(AttributeError):
        test = PoissonBracket(*input)


@pytest.mark.parametrize("input,kwargs,expected", single_pb_pass)
def test_single_pb_parametrized_pass(input, kwargs, expected):
    test = PoissonBracket(*input, **kwargs)
    assert test == expected


@pytest.mark.parametrize("input,kwargs,expected", single_pb_doit_pass)
def test_single_pb_doit_parametrized_pass(input, kwargs, expected):
    test = PoissonBracket(*input, **kwargs)
    assert test.doit() == expected

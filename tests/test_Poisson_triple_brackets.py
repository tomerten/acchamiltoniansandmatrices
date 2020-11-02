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

triple_pb_doit_fail = []

triple_pb_doit_pass = [
    (
        (A(x, px), B(x, y, py)),
        (C(x, px),),
        (D(x, y),),
        {"coords": [x, y], "mom": [px, py]},
        {"coords": [x, y], "mom": [px, py]},
        {"coords": [x, y], "mom": [px, py]},
        (
            -Derivative(C(x, px), px)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x, y, py), py, (x, 2))
                - Derivative(A(x, px), px, x) * Derivative(B(x, y, py), py, x)
            )
            - Derivative(C(x, px), x)
            * Derivative(A(x, px), (px, 2))
            * Derivative(B(x, y, py), py, x)
        )
        * Derivative(D(x, y), y)
        + (
            -Derivative(C(x, px), px)
            * (
                -Derivative(A(x, px), (px, 2)) * Derivative(B(x, y, py), (x, 2))
                - Derivative(A(x, px), (px, 2), x) * Derivative(B(x, y, py), x)
            )
            - Derivative(C(x, px), (px, 2))
            * (
                -Derivative(A(x, px), px) * Derivative(B(x, y, py), (x, 2))
                - Derivative(A(x, px), px, x) * Derivative(B(x, y, py), x)
            )
            - Derivative(C(x, px), x) * Derivative(A(x, px), (px, 3)) * Derivative(B(x, y, py), x)
            - Derivative(C(x, px), px, x)
            * Derivative(A(x, px), (px, 2))
            * Derivative(B(x, y, py), x)
        )
        * Derivative(D(x, y), x),
    ),
]


@pytest.mark.parametrize(
    "input1,input2,input3,kwargs1,kwargs2,kwargs3,expected", triple_pb_doit_pass
)
def test_double_pb_doit_parametrized_pass(
    input1, input2, input3, kwargs1, kwargs2, kwargs3, expected
):
    pb1 = PoissonBracket(*input1, **kwargs1)
    pb2 = PoissonBracket(pb1, *input2, **kwargs2)
    pb3 = PoissonBracket(pb2, *input3, **kwargs3)
    assert pb3.doit() == expected

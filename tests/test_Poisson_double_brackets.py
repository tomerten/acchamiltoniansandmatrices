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

double_pb_fail = []

double_pb_pass = [
    ((S.One, half), (S.One,), {}, {}, 0),
    ((A, half), (S.One,), {}, {}, 0),
    ((A(x), half), (S.One,), {}, {}, 0),
    ((A(x, px), half), (S.One,), {}, {}, 0),
    ((S.One, half), (S.One,), {"coords": [x]}, {}, 0),
    ((A, half), (S.One,), {"coords": [x]}, {}, 0),
    ((A(x), half), (S.One,), {"coords": [x]}, {}, 0),
    ((A(x, px), half), (S.One,), {"coords": [x]}, {}, 0),
    ((S.One, half), (A,), {}, {}, 0),
    ((A, half), (A,), {}, {}, 0),
    ((A(x), half), (A,), {}, {}, 0),
    ((A(x, px), half), (A,), {}, {}, 0),
    ((S.One, half), (A,), {"coords": [x]}, {}, 0),
    ((A, half), (A,), {"coords": [x]}, {}, 0),
    ((A(x), half), (A,), {"coords": [x]}, {}, 0),
    ((A(x, px), half), (A,), {"coords": [x]}, {}, 0),
    ((S.One, half), (A(x),), {}, {}, 0),
    ((A, half), (A(x),), {}, {}, 0),
    ((A(x), half), (A(x),), {}, {}, 0),
    ((A(x, px), half), (A(x),), {}, {}, 0),
    ((S.One, half), (A(x),), {"coords": [x]}, {}, 0),
    ((A, half), (A(x),), {"coords": [x]}, {}, 0),
    ((A(x), half), (A(x),), {"coords": [x]}, {}, 0),
    ((A(x, px), half), (A(x),), {"coords": [x]}, {}, 0),
    ((S.One, half), (A(x, px),), {}, {}, 0),
    ((A, half), (A(x, px),), {}, {}, 0),
    ((A(x), half), (A(x, px),), {}, {}, 0),
    ((A(x, px), half), (A(x, px),), {}, {}, 0),
    ((S.One, half), (A(x, px),), {"coords": [x]}, {}, 0),
    ((A, half), (A(x, px),), {"coords": [x]}, {}, 0),
    ((A(x), half), (A(x, px),), {"coords": [x]}, {}, 0),
    ((A(x, px), half), (A(x, px),), {"coords": [x]}, {}, 0),
    (reversed((S.One, half)), (S.One,), {}, {}, 0),
    (reversed((A, half)), (S.One,), {}, {}, 0),
    (reversed((A(x), half)), (S.One,), {}, {}, 0),
    (reversed((A(x, px), half)), (S.One,), {}, {}, 0),
    (reversed((S.One, half)), (S.One,), {"coords": [x]}, {}, 0),
    (reversed((A, half)), (S.One,), {"coords": [x]}, {}, 0),
    (reversed((A(x), half)), (S.One,), {"coords": [x]}, {}, 0),
    (reversed((A(x, px), half)), (S.One,), {"coords": [x]}, {}, 0),
    (reversed((S.One, half)), (A,), {}, {}, 0),
    (reversed((A, half)), (A,), {}, {}, 0),
    (reversed((A(x), half)), (A,), {}, {}, 0),
    (reversed((A(x, px), half)), (A,), {}, {}, 0),
    (reversed((S.One, half)), (A,), {"coords": [x]}, {}, 0),
    (reversed((A, half)), (A,), {"coords": [x]}, {}, 0),
    (reversed((A(x), half)), (A,), {"coords": [x]}, {}, 0),
    (reversed((A(x, px), half)), (A,), {"coords": [x]}, {}, 0),
    (reversed((S.One, half)), (A(x),), {}, {}, 0),
    (reversed((A, half)), (A(x),), {}, {}, 0),
    (reversed((A(x), half)), (A(x),), {}, {}, 0),
    (reversed((A(x, px), half)), (A(x),), {}, {}, 0),
    (reversed((S.One, half)), (A(x),), {"coords": [x]}, {}, 0),
    (reversed((A, half)), (A(x),), {"coords": [x]}, {}, 0),
    (reversed((A(x), half)), (A(x),), {"coords": [x]}, {}, 0),
    (reversed((A(x, px), half)), (A(x),), {"coords": [x]}, {}, 0),
    (reversed((S.One, half)), (A(x, px),), {}, {}, 0),
    (reversed((A, half)), (A(x, px),), {}, {}, 0),
    (reversed((A(x), half)), (A(x, px),), {}, {}, 0),
    (reversed((A(x, px), half)), (A(x, px),), {}, {}, 0),
    (reversed((S.One, half)), (A(x, px),), {"coords": [x]}, {}, 0),
    (reversed((A, half)), (A(x, px),), {"coords": [x]}, {}, 0),
    (reversed((A(x), half)), (A(x, px),), {"coords": [x]}, {}, 0),
    (reversed((A(x, px), half)), (A(x, px),), {"coords": [x]}, {}, 0),
    ((A, B), (S.One,), {}, {}, 0),
]


@pytest.mark.parametrize("input1,input2,kwargs1,kwargs2,expected", double_pb_pass)
def test_double_pb_parametrized_pass(input1, input2, kwargs1, kwargs2, expected):
    pb1 = PoissonBracket(*input1, **kwargs1)
    pb2 = PoissonBracket(pb1, *input2, **kwargs2)
    assert pb2 == expected


double_pb_doit_fail = []

double_pb_doit_pass = [
    ((A, B), (S.One,), {}, {}, 0),
    ((A, B), (C,), {}, {}, PoissonBracket(PoissonBracket(A, B), C)),
    ((A(x, px), B(x, px)), (C,), {}, {}, PoissonBracket(PoissonBracket(A(x, px), B(x, px)), C)),
    ((x ** 2, x + px), (C,), {}, {}, PoissonBracket(PoissonBracket(x ** 2, x + px), C)),
    ((A, B), (C,), {}, {}, PoissonBracket(PoissonBracket(A, B), C)),
    ((A(x), B), (C,), {}, {}, PoissonBracket(PoissonBracket(A(x), B), C)),
    ((A, B(x)), (C,), {}, {}, PoissonBracket(PoissonBracket(A, B(x)), C)),
    ((A, B), (C(x),), {}, {}, S.NegativeOne * PoissonBracket(C(x), PoissonBracket(A, B))),
    ((A, B), (C,), {"coords": [x]}, {}, PoissonBracket(PoissonBracket(A, B), C)),
    ((A(x), B), (C,), {"coords": [x]}, {}, PoissonBracket(PoissonBracket(A(x), B), C)),
    ((A, B(x)), (C,), {"coords": [x]}, {}, PoissonBracket(PoissonBracket(A, B(x)), C)),
    (
        (A, B),
        (C(x),),
        {"coords": [x]},
        {},
        S.NegativeOne * PoissonBracket(C(x), PoissonBracket(A, B)),
    ),
    ((A(x), B), (C,), {}, {"coords": [x]}, PoissonBracket(PoissonBracket(A(x), B), C)),
    ((A, B(x)), (C,), {}, {"coords": [x]}, PoissonBracket(PoissonBracket(A, B(x)), C)),
    (
        (A, B),
        (C(x),),
        {},
        {"coords": [x]},
        S.NegativeOne * PoissonBracket(C(x), PoissonBracket(A, B)),
    ),
    ((A, B), (C,), {}, {"coords": [x]}, PoissonBracket(PoissonBracket(A, B), C)),
    ((A(x), B(x)), (C,), {"coords": [x]}, {}, PoissonBracket(PoissonBracket(A(x), B(x)), C)),
    (
        (A(x), B(x)),
        (C(x),),
        {"coords": [x]},
        {},
        S.NegativeOne * PoissonBracket(C(x), PoissonBracket(A(x), B(x))),
    ),
    (
        (A(x), B(x)),
        (C(x),),
        {},
        {"coords": [x]},
        -PoissonBracket(C(x), PoissonBracket(A(x), B(x))),
    ),
    (
        (A(x, px), B(x)),
        (C(x),),
        {"coords": [x]},
        {},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x))),
    ),
    ((A(x, px), B(x)), (C(x),), {}, {}, -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x)))),
    ((A(x), B(x, px)), (C(x),), {}, {}, -PoissonBracket(C(x), PoissonBracket(A(x), B(x, px)))),
    ((A(x), B(x)), (C(x, px),), {}, {}, -PoissonBracket(C(x, px), PoissonBracket(A(x), B(x)))),
    (
        (A(x, px), B(x, px)),
        (C(x),),
        {},
        {},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x, px),),
        {},
        {},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {},
        {},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x),),
        {"coords": [x]},
        {},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x), B(x, px)),
        (C(x),),
        {"coords": [x]},
        {},
        -PoissonBracket(C(x), PoissonBracket(A(x), B(x, px))),
    ),
    (
        (A(x), B(x)),
        (C(x, px),),
        {"coords": [x]},
        {},
        -PoissonBracket(C(x, px), PoissonBracket(A(x), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x),),
        {"coords": [x]},
        {},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x, px),),
        {"coords": [x]},
        {},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {"coords": [x]},
        {},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x),),
        {},
        {"coords": [x]},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x), B(x, px)),
        (C(x),),
        {},
        {"coords": [x]},
        -PoissonBracket(C(x), PoissonBracket(A(x), B(x, px))),
    ),
    (
        (A(x), B(x)),
        (C(x, px),),
        {},
        {"coords": [x]},
        -PoissonBracket(C(x, px), PoissonBracket(A(x), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x),),
        {},
        {"coords": [x]},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x, px),),
        {},
        {"coords": [x]},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {},
        {"coords": [x]},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x), B(x, px)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {},
        -PoissonBracket(C(x), PoissonBracket(A(x), B(x, px))),
    ),
    (
        (A(x), B(x)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {},
        -PoissonBracket(C(x, px), PoissonBracket(A(x), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x),),
        {},
        {"coords": [x], "mom": [px]},
        Derivative(C(x), x) * Derivative(A(x, px), (px, 2)) * Derivative(B(x), x),
    ),
    (
        (A(x), B(x, px)),
        (C(x),),
        {},
        {"coords": [x], "mom": [px]},
        -Derivative(C(x), x) * Derivative(A(x), x) * Derivative(B(x, px), (px, 2)),
    ),
    ((A(x), B(x)), (C(x, px),), {}, {"coords": [x], "mom": [px]}, 0),
    (
        (A(x, px), B(x, px)),
        (C(x),),
        {},
        {"coords": [x], "mom": [px]},
        -Derivative(C(x), x)
        * (
            -Derivative(A(x, px), px) * Derivative(B(x, px), px, x)
            - Derivative(A(x, px), (px, 2)) * Derivative(B(x, px), x)
            + Derivative(A(x, px), x) * Derivative(B(x, px), (px, 2))
            + Derivative(A(x, px), px, x) * Derivative(B(x, px), px)
        ),
    ),
    (
        (A(x, px), B(x)),
        (C(x, px),),
        {},
        {"coords": [x], "mom": [px]},
        S.NegativeOne
        * (
            -Derivative(C(x, px), px)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x), (x, 2))
                - Derivative(A(x, px), px, x) * Derivative(B(x), x)
            )
            - Derivative(C(x, px), x) * Derivative(A(x, px), (px, 2)) * Derivative(B(x), x)
        ),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {},
        {"coords": [x], "mom": [px]},
        S.NegativeOne
        * (
            -Derivative(C(x, px), px)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x, px), (x, 2))
                + Derivative(A(x, px), x) * Derivative(B(x, px), px, x)
                + Derivative(A(x, px), (x, 2)) * Derivative(B(x, px), px)
                - Derivative(A(x, px), px, x) * Derivative(B(x, px), x)
            )
            + Derivative(C(x, px), x)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x, px), px, x)
                - Derivative(A(x, px), (px, 2)) * Derivative(B(x, px), x)
                + Derivative(A(x, px), x) * Derivative(B(x, px), (px, 2))
                + Derivative(A(x, px), px, x) * Derivative(B(x, px), px)
            )
        ),
    ),
    (
        (A(x, px), B(x)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {"coords": [x]},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x), B(x, px)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {"coords": [x]},
        -PoissonBracket(C(x), PoissonBracket(A(x), B(x, px))),
    ),
    (
        (A(x), B(x)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {"coords": [x]},
        -PoissonBracket(C(x, px), PoissonBracket(A(x), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {"coords": [x]},
        -PoissonBracket(C(x), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {"coords": [x]},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x))),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {"coords": [x]},
        -PoissonBracket(C(x, px), PoissonBracket(A(x, px), B(x, px))),
    ),
    (
        (A(x, px), B(x)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {"coords": [x], "mom": [px]},
        Derivative(C(x), x) * Derivative(A(x, px), (px, 2)) * Derivative(B(x), x),
    ),
    (
        (A(x), B(x, px)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {"coords": [x], "mom": [px]},
        -Derivative(C(x), x) * Derivative(A(x), x) * Derivative(B(x, px), (px, 2)),
    ),
    ((A(x), B(x)), (C(x, px),), {"coords": [x], "mom": [px]}, {"coords": [x], "mom": [px]}, 0),
    (
        (A(x, px), B(x, px)),
        (C(x),),
        {"coords": [x], "mom": [px]},
        {"coords": [x], "mom": [px]},
        -Derivative(C(x), x)
        * (
            -Derivative(A(x, px), px) * Derivative(B(x, px), px, x)
            - Derivative(A(x, px), (px, 2)) * Derivative(B(x, px), x)
            + Derivative(A(x, px), x) * Derivative(B(x, px), (px, 2))
            + Derivative(A(x, px), px, x) * Derivative(B(x, px), px)
        ),
    ),
    (
        (A(x, px), B(x)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {"coords": [x], "mom": [px]},
        S.NegativeOne
        * (
            -Derivative(C(x, px), px)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x), (x, 2))
                - Derivative(A(x, px), px, x) * Derivative(B(x), x)
            )
            - Derivative(C(x, px), x) * Derivative(A(x, px), (px, 2)) * Derivative(B(x), x)
        ),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {"coords": [x], "mom": [px]},
        S.NegativeOne
        * (
            -Derivative(C(x, px), px)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x, px), (x, 2))
                + Derivative(A(x, px), x) * Derivative(B(x, px), px, x)
                + Derivative(A(x, px), (x, 2)) * Derivative(B(x, px), px)
                - Derivative(A(x, px), px, x) * Derivative(B(x, px), x)
            )
            + Derivative(C(x, px), x)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x, px), px, x)
                - Derivative(A(x, px), (px, 2)) * Derivative(B(x, px), x)
                + Derivative(A(x, px), x) * Derivative(B(x, px), (px, 2))
                + Derivative(A(x, px), px, x) * Derivative(B(x, px), px)
            )
        ),
    ),
    ((A(x, px), B), (C,), {}, {}, PoissonBracket(PoissonBracket(A(x, px), B), C)),
    (
        (x ** 2, x + px),
        (C,),
        {"coords": [x], "mom": [px]},
        {},
        PoissonBracket(PoissonBracket(x ** 2, x + px, coords=[x], mom=[px]), C),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {},
        PoissonBracket(PoissonBracket(A(x, px), B(x, px)), C(x, px)),
    ),
    (
        (A(x, px), B(x, px)),
        (C,),
        {"coords": [x], "mom": [px]},
        {},
        PoissonBracket(PoissonBracket(A(x, px), B(x, px)), C),
    ),
    (
        (A(x, px), B(x, px)),
        (C(x, px),),
        {"coords": [x], "mom": [px]},
        {"coords": [x], "mom": [px]},
        S.NegativeOne
        * (
            -Derivative(C(x, px), px)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x, px), (x, 2))
                + Derivative(A(x, px), x) * Derivative(B(x, px), px, x)
                + Derivative(A(x, px), (x, 2)) * Derivative(B(x, px), px)
                - Derivative(A(x, px), px, x) * Derivative(B(x, px), x)
            )
            + Derivative(C(x, px), x)
            * (
                -Derivative(A(x, px), px) * Derivative(B(x, px), px, x)
                - Derivative(A(x, px), (px, 2)) * Derivative(B(x, px), x)
                + Derivative(A(x, px), x) * Derivative(B(x, px), (px, 2))
                + Derivative(A(x, px), px, x) * Derivative(B(x, px), px)
            )
        ),
    ),
]


@pytest.mark.parametrize("input1,input2,kwargs1,kwargs2,expected", double_pb_doit_pass)
def test_double_pb_doit_parametrized_pass(input1, input2, kwargs1, kwargs2, expected):
    pb1 = PoissonBracket(*input1, **kwargs1)
    pb2 = PoissonBracket(pb1, *input2, **kwargs2)
    print(pb2.doit() - expected)
    assert pb2.doit() == expected


# @pytest.mark.parametrize('input1,input2,kwargs1,kwargs2,expected',double_pb_doit_fail)
# def test_double_pb_doit_parametrized_fail(input1, input2, kwargs1, kwargs2, expected):
#     with pytest.raises(AttributeError):
#         pb1 = PoissonBracket(*input1, **kwargs1)
#         pb2 = PoissonBracket(pb1, *input2, **kwargs2)

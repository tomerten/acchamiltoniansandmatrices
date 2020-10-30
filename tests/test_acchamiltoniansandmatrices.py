from sympy import Derivative, Function, Rational, symbols

from acchamiltoniansandmatrices.LieMaps.Poisson import PoissonBracket

A = Function("A", commutative=False)
B = Function("B", commutative=False)
C = Function("C", commutative=False)
D = Function("D", commutative=False)

x, px, y, py = symbols("x px y py")

half = Rational(1, 2)


def test_single_pb_undef_Function_number():
    test0 = PoissonBracket(half, A)
    assert test0 == 0


def test_single_pb_undef_Function():
    test1 = PoissonBracket(A, B)
    assert test1.free_symbols == set()


def test_single_pb_explicit_expr_no_indep():
    test2 = PoissonBracket(x ** 2, x + px)
    assert test2.doit() == test2
    assert test2.free_symbols == {x, px}


def test_single_pb_Function_with_var_no_indep():
    test3 = PoissonBracket(A(x, px), B(y, py))
    assert test3.doit() == test3
    assert test3.free_symbols == {x, px, py, y}


def test_single_pb_explicit_expr_indep():
    test4 = PoissonBracket(x ** 2, x + px, coords=[x], mom=[px])
    assert test4.free_symbols == {x, px}
    assert test4.doit() == 2 * x


def test_single_pb_Function_with_var_indep():
    test5 = PoissonBracket(A(x, px), B(x, y, py), coords=[x, y], mom=[px, py])
    assert test5.free_symbols == {x, px, py, y}
    assert test5.doit() == -Derivative(A(x, px), px) * Derivative(B(x, y, py), x)


def test_double_pb_2undef_Function_no_var_undef_Function_no_var_no_indep():
    test1 = PoissonBracket(A, B)
    test6 = PoissonBracket(test1, C)
    assert test6.free_symbols == set()
    assert test6.doit() == test6


def test_double_pb_2explicit_no_var_undef_Function_no_var_no_indep():
    test2 = PoissonBracket(x ** 2, x + px)
    test7 = PoissonBracket(test2, C)
    assert test7.free_symbols == {x, px}
    assert test7.doit() == test7


def test_double_pb_Function_with_var_undef_Function_no_var_no_indep():
    test3 = PoissonBracket(A(x, px), B(y, py))
    test8 = PoissonBracket(test3, C)
    assert test8.doit() == test8
    assert test8.free_symbols == {x, px, py, y}


def test_double_pb_explicit_expr_indep_undef_Function_no_var_no_indep():
    test4 = PoissonBracket(x ** 2, x + px, coords=[x], mom=[px])
    test9 = PoissonBracket(test4, C)
    assert test9.free_symbols == {x, px}
    assert test9.doit() == test9


def test_double_pb_Function_with_var_indep_undef_Function_no_var_no_indep():
    test5 = PoissonBracket(A(x, px), B(x, y, py), coords=[x, y], mom=[px, py])
    test10 = PoissonBracket(test5, C)
    assert test10.free_symbols == {x, px, py, y}
    assert test10.doit() == test10


def test_double_pb_Function_with_var_indep_Function_with_var_no_indep():
    test5 = PoissonBracket(A(x, px), B(x, y, py), coords=[x, y], mom=[px, py])
    test11 = PoissonBracket(test5, C(x, px))
    assert test11.free_symbols == {x, px, py, y}
    assert test11.doit() == test11


def test_double_pb_Function_with_var_indep_Function_with_var_indep():
    test5 = PoissonBracket(A(x, px), B(x, y, py), coords=[x, y], mom=[px, py])
    test12 = PoissonBracket(test5, C(x, px), coords=[x, y], mom=[px, py])
    res12 = (
        -Derivative(A(x, px), px) * Derivative(B(x, y, py), (x, 2))
        - Derivative(A(x, px), px, x) * Derivative(B(x, y, py), x)
    ) * Derivative(C(x, px), px) + Derivative(A(x, px), (px, 2)) * Derivative(
        B(x, y, py), x
    ) * Derivative(
        C(x, px), x
    )
    assert test12.free_symbols == {x, px, py, y}
    assert test12.doit() == res12


def test_single_pb_sum_Function_with_var_Function_with_var_indep():
    test13 = PoissonBracket(A(x, px) + B(x, y, py), C(x, px), coords=[x, y], mom=[px, py])
    res13 = (Derivative(A(x, px), x) + Derivative(B(x, y, py), x)) * Derivative(
        C(x, px), px
    ) - Derivative(A(x, px), px) * Derivative(C(x, px), x)
    assert test13.free_symbols == {x, px, py, y}
    assert test13.doit() == res13


def test_single_pb_Function_with_var_sum_Function_with_var_indep():
    test14 = PoissonBracket(A(x, px), B(x, y, py) + C(x, px), coords=[x, y], mom=[px, py])
    res14 = -Derivative(A(x, px), px) * (
        Derivative(B(x, y, py), x) + Derivative(C(x, px), x)
    ) + Derivative(A(x, px), x) * Derivative(C(x, px), px)
    assert test14.free_symbols == {x, px, py, y}
    assert test14.doit() == res14


def test_single_pb_product_Function_with_var_Function_with_var_indep():
    test15 = PoissonBracket(A(x, px) * B(x, y, py), C(x, px), coords=[x, y], mom=[px, py])
    res15 = (
        A(x, px) * Derivative(B(x, y, py), x) + Derivative(A(x, px), x) * B(x, y, py)
    ) * Derivative(C(x, px), px) - Derivative(A(x, px), px) * B(x, y, py) * Derivative(C(x, px), x)
    assert test15.free_symbols == {x, px, py, y}
    assert test15.doit() == res15


def test_single_pb_Function_with_var_product_Function_with_var_indep():
    test16 = PoissonBracket(A(x, px), B(x, y, py) * C(x, px), coords=[x, y], mom=[px, py])
    res16 = -Derivative(A(x, px), px) * (
        B(x, y, py) * Derivative(C(x, px), x) + Derivative(B(x, y, py), x) * C(x, px)
    ) + Derivative(A(x, px), x) * B(x, y, py) * Derivative(C(x, px), px)
    assert test16.free_symbols == {x, px, py, y}
    assert test16.doit() == res16


def test_single_pb_Pow_Function_with_var_Function_with_var_indep():
    test17 = PoissonBracket(A(x, px) ** 3, B(x, y, py), coords=[x, y], mom=[px, py])
    res17 = (
        -3
        * A(x, px) ** 3
        * Derivative(A(x, px), px)
        * A(x, px) ** (-1)
        * Derivative(B(x, y, py), x)
    )
    assert test17.free_symbols == {x, px, py, y}
    assert test17.doit() == res17


def test_triple_pb_3Function_with_var_indep():
    test5 = PoissonBracket(A(x, px), B(x, y, py), coords=[x, y], mom=[px, py])
    test12 = PoissonBracket(test5, C(x, px), coords=[x, y], mom=[px, py])
    test18 = PoissonBracket(test12, D(x, y), coords=[x, y], mom=[px, py])
    res18 = -(
        (
            -Derivative(A(x, px), px) * Derivative(B(x, y, py), py, (x, 2))
            - Derivative(A(x, px), px, x) * Derivative(B(x, y, py), py, x)
        )
        * Derivative(C(x, px), px)
        + Derivative(A(x, px), (px, 2)) * Derivative(B(x, y, py), py, x) * Derivative(C(x, px), x)
    ) * Derivative(D(x, y), y) - (
        (
            -Derivative(A(x, px), px) * Derivative(B(x, y, py), (x, 2))
            - Derivative(A(x, px), px, x) * Derivative(B(x, y, py), x)
        )
        * Derivative(C(x, px), (px, 2))
        + (
            -Derivative(A(x, px), (px, 2)) * Derivative(B(x, y, py), (x, 2))
            - Derivative(A(x, px), (px, 2), x) * Derivative(B(x, y, py), x)
        )
        * Derivative(C(x, px), px)
        + Derivative(A(x, px), (px, 2)) * Derivative(B(x, y, py), x) * Derivative(C(x, px), px, x)
        + Derivative(A(x, px), (px, 3)) * Derivative(B(x, y, py), x) * Derivative(C(x, px), x)
    ) * Derivative(
        D(x, y), x
    )
    assert test18.free_symbols == {x, px, py, y}


def test_doube_pb_rsum_deep():
    test13 = PoissonBracket(A(x, px) + B(x, y, py), C(x, px), coords=[x, y], mom=[px, py])
    test19 = PoissonBracket(test13, D(x, y), coords=[x, y], mom=[px, py])
    res19 = -(
        (Derivative(A(x, px), x) + Derivative(B(x, y, py), x)) * Derivative(C(x, px), (px, 2))
        - Derivative(A(x, px), px) * Derivative(C(x, px), px, x)
        - Derivative(A(x, px), (px, 2)) * Derivative(C(x, px), x)
        + Derivative(A(x, px), px, x) * Derivative(C(x, px), px)
    ) * Derivative(D(x, y), x) - Derivative(B(x, y, py), py, x) * Derivative(
        C(x, px), px
    ) * Derivative(
        D(x, y), y
    )
    assert test19.free_symbols == {x, px, py, y}

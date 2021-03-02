from sympy import Rational, oo, simplify, symbols
from sympy.core.numbers import NegativeOne, One, Zero

from ..Factorization.Factorization import truncate
from ..Hamiltonians.LatticeElementHamiltonians import (
    HamDrift6D,
    HamQuad6D,
    HamSext6D,
    HamSQuad6D,
)
from ..LieMaps.LieOperator import LieOperator


def AssignHam(element, order=3, length=1, k=1, flag=1):
    """
    Element:
    1 = drift space
    2 = quadrupole
    25 = skew quadrupole
    3 = sextupole
    35 = skew sextupole - tbp
    """
    # define symbols
    beta0, gamma0 = symbols("beta_0 gamma_0", real=True, positive=True)
    x, px, y, py, z, delta, eps, h, k0, k1, k2, sigma, betag, f = symbols(
        "x p_x y p_y z delta epsilon h k_0 k_1 k_2 sigma beta_gamma f", real=True
    )
    order = order + 1
    betagamma_rep = list(zip([beta0 * gamma0, beta0], [oo, 1]))
    series_rep = list(zip([px, py, delta], [eps * px, eps * py, eps * delta]))
    coord_rep = list(zip([x, y], [eps * x, eps * y]))

    thin_rep = list(zip([px, py, delta], [0, 0, 0]))
    if element == 1:
        # drift hamiltonian
        H = HamDrift6D(beta0, gamma0, length, x, px, y, py, delta)
        H = H.subs(betagamma_rep)
        H = H.subs(series_rep).series(eps, n=order).removeO()
        H = simplify(H.subs(eps, One()) - H.subs(eps, Zero()))
    #         H = drift(order, length)

    elif element == 2:
        # quadrupole - thin is set by flag
        H = (
            NegativeOne()
            * Rational(1, 2)
            * HamQuad6D(beta0, gamma0, length, x, px, y, py, delta, k)
        )
        H = H.subs(betagamma_rep)
        H = H.subs(series_rep).series(eps, n=order).removeO()
        H = H.subs(coord_rep)
        H = simplify(H.subs(eps, One()) - H.subs(eps, Zero()))

        if flag:
            H = H.subs(thin_rep)

    #         H = quad(order, length, strength, flag)
    elif element == 25:
        H = (
            NegativeOne()
            * Rational(1, 2)
            * HamSQuad6D(beta0, gamma0, length, x, px, y, py, delta, k)
        )
        H = H.subs(betagamma_rep)
        H = H.subs(series_rep).series(eps, n=order).removeO()
        H = H.subs(coord_rep)
        H = simplify(H.subs(eps, One()) - H.subs(eps, Zero()))

        if flag:
            H = H.subs(thin_rep)
    #         H = skew_quad(order, length, strength, flag)

    elif element == 3:
        H = (
            NegativeOne()
            * Rational(1, 2)
            * HamSext6D(beta0, gamma0, length, x, px, y, py, delta, k)
        )
        H = H.subs(betagamma_rep)
        H = H.subs(series_rep).series(eps, n=order).removeO()
        H = H.subs(coord_rep)
        H = simplify(H.subs(eps, One()) - H.subs(eps, Zero()))

        if flag:
            H = H.subs(thin_rep)
    #         H = sext(order, length, strength, flag)
    #     elif H == 35:
    #         H = skew_sext(order, length, strength)
    else:
        raise ValueError(element, " no such reference in library.")

    return H


def RingHam(
    beamline, BCH_order, poly_cutoff, doit=False
):  # combine beamline with BCH into one map
    x, y, z, px, py, delta = symbols("x y z p_x p_y delta")
    poly_cutoff = poly_cutoff + 1
    for i, element in enumerate(beamline):
        if i == 0:
            H0 = AssignHam(element[0], element[1], element[2], element[3], element[4])
            H_int = LieOperator(H0, [x, y, z], [px, py, delta])
        else:
            H0 = AssignHam(element[0], element[1], element[2], element[3], element[4])
            H0 = LieOperator(H0, [x, y, z], [px, py, delta])

            H_int = H_int.BCH(H0, BCH_order)

    H_int = H_int.doit()

    temp_H = truncate(H_int, poly_cutoff)  # cutoff Hamiltonian at speciefied polynomial degree

    H = LieOperator(temp_H, [x, y, z], [px, py, delta])

    return H

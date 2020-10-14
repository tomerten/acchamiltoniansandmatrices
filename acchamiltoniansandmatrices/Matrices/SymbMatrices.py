from sympy import Matrix, Rational, cos, cosh, sin, sinh, sqrt


def half():
    """
    Helper function to write half as a rational
    in a sympy expression.
    """
    return Rational(1, 2)


def RsymbDrift6D(beta0, gamma0, L):
    """
    6D R matrix for drift.

    Arguments:
    ----------
    beta0   : float | sympy symbol
        Relativistic beta
    gamma0  : float | sympy symbol
        Relativistic gamma
    L       : float | sympy symbol
        Length of the drift.
    """

    return Matrix(
        [
            [1, L, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, L, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, L / (beta0 * gamma0) ** 2],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RsymbDrift4D(beta0, gamma0, L):
    RsymbDrift6D(beta0, gamma0, L)[0:4, 0:4]


def RsymbQuad6D(beta0, gamma0, L, k1):
    omega = sqrt(k1)
    omegaL = omega * L
    c = cos(omegaL)
    s = sin(omegaL)
    ch = cosh(omegaL)
    sh = sinh(omegaL)

    return Matrix(
        [
            [c, s / omega, 0, 0, 0, 0],
            [-omega * s, c, 0, 0, 0, 0],
            [0, 0, ch, sh / omega, 0, 0],
            [0, 0, omega * sh, ch, 0, 0],
            [0, 0, 0, 0, 1, L / (beta0 * gamma0) ** 2],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RsymbQuad4D(beta0, gamma0, L, k1):
    return RsymbQuad6D(beta0, gamma0, L, k1)[0:4, 0:4]


def RsymbSQuad6D(beta0, gamma0, L, k1s):
    omega = sqrt(k1s)
    omegaL = omega * L
    c = cos(omegaL)
    s = sin(omegaL)
    ch = cosh(omegaL)
    sh = sinh(omegaL)
    cp = c + ch
    sp = s + sh
    cm = c - ch
    sm = s - sh
    return Matrix(
        [
            [half() * cp, half() / omega * sp, half() * cm, half() / eomega * sm, 0, 0],
            [-half() * omega * sm, half() * cp, -half() * omega * sp, half() * cm, 0, 0],
            [half() * cm, half() / omega * sm, half() * cp, half() / omega * sp, 0, 0],
            [-half() * omega * sp, half() * cm, -omega * half() * sm, half() * cp, 0, 0],
            [0, 0, 0, 0, 1, L / (beta0, gamma0) ** 2],
            [0, 0, 0, 0, 0, 1],
        ]
    )

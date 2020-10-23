# Ref: Wolski Lectures on Linear Dynamics
from sympy import (
    Derivative,
    Rational,
    S,
    Symbol,
    besselj,
    cos,
    pi,
    series,
    sin,
    sqrt,
    symbols,
)


def half():
    return Rational(1, 2)


def HamDrift6D(beta0, gamma0, L, x, px, y, py, delta):
    """
    6D drift Hamiltonian.

    Arguments:
    ----------
    beta0  : float | sympy symbol
        Relativistic Beta
    gamma0 : float | sympy symbol
        Relativistic gamma
    L      : float | sympy symbol
        Length of the drift
    x      : float | sympy symbol
        Horizontal coordinate [meter]
    px     : float | sympy symbol
        Horizontal momentum
    y      :  float | sympy symbol
        Vertical coordinate
    py      :  float | sympy symbol
        Vertical momentum
    delta   :  float | sympy symbol
        Longitudinal momentum deviation
    """
    return L * (
        delta / beta0
        - sqrt((1 / beta0 + delta) ** 2 - px ** 2 - py ** 2 - 1 / (beta0 * gamma0) ** 2)
    )


def HamDrift4D(beta0, gamma0, L, x, px, y, py):
    """
    4D drift Hamiltonian.

    Arguments:
    ----------
    beta0  : float | sympy symbol
        Relativistic Beta
    gamma0 : float | sympy symbol
        Relativistic gamma
    L      : float | sympy symbol
        Length of the drift
    x      : float | sympy symbol
        Horizontal coordinate [meter]
    px     : float | sympy symbol
        Horizontal momentum
    y      :  float | sympy symbol
        Vertical coordinate
    py      :  float | sympy symbol
        Vertical momentum
    """
    return HamDrift6D(beta0, gamma0, L, x, px, y, py, 0)


def HamDrift6DParaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta):
    return L * (half() * px ** 2 + half() * py ** 2 + half() * (delta / (beta0 * gamma0)) ** 2)


def HamDipole6D(beta0, gamma0, L, x, px, y, py, delta, h, k0):
    """
    6D thick dipole Hamiltonian.

    Arguments:
    ----------
    beta0  : float | sympy symbol
        Relativistic Beta
    gamma0 : float | sympy symbol
        Relativistic gamma
    L      : float | sympy symbol
        Length of the drift
    x      : float | sympy symbol
        Horizontal coordinate [meter]
    px     : float | sympy symbol
        Horizontal momentum
    y      : float | sympy symbol
        Vertical coordinate
    py     : float | sympy symbol
        Vertical momentum
    delta  : float | sympy symbol
        Longitudinal momentum deviation
    h      : float | sympy symbol
        Local curvature of the curvy linear coordinate system
    k0     : float | sympy symbol
        Normalised dipole field strength (k0 = q B0 / P0 )
    """
    t0 = delta / beta0
    t1 = 1 / beta0 + delta
    t2 = 1 / (beta0 * gamma0)
    t3 = x - (h * x ** 2) / (2 * (1 + h * x))
    factor = 1 + h * x
    sroot = sqrt(t1 ** 2 - px ** 2 - py ** 2 - t2 ** 2)
    return L * (t0 - factor * sroot + factor * k0 * t3)


def HamDipole4D(beta0, gamma0, L, x, px, y, py, h, k0):
    """
    4D thick dipole Hamiltonian.

    Arguments:
    ----------
    beta0  : float | sympy symbol
        Relativistic Beta
    gamma0 : float | sympy symbol
        Relativistic gamma
    L      : float | sympy symbol
        Length of the drift
    x      : float | sympy symbol
        Horizontal coordinate [meter]
    px     : float | sympy symbol
        Horizontal momentum
    y      : float | sympy symbol
        Vertical coordinate
    py     : float | sympy symbol
        Vertical momentum
    h      : float | sympy symbol
        Local curvature of the curvy linear coordinate system
    k0     : float | sympy symbol
        Normalised dipole field strength (k0 = q B0 / P0 )
    """
    return HamDipole6D(beta0, gamma0, L, x, px, y, py, 0, h, k0)


def HamDipole6DParaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta, h, k0):
    return L * (
        half() * px ** 2
        + half() * py ** 2
        + (k0 - h) * x
        + half() * k0 * h * x ** 2
        - h / beta0 * x * delta
        + half() * (delta / beta0 * gamma0) ** 2
    )


def HamQuad6D(beta0, gamma0, L, x, px, y, py, delta, k1):
    """
    6D thick quadruple Hamiltonian.

    Arguments:
    ----------
    beta0  : float | sympy symbol
        Relativistic Beta
    gamma0 : float | sympy symbol
        Relativistic gamma
    L      : float | sympy symbol
        Length of the drift
    x      : float | sympy symbol
        Horizontal coordinate [meter]
    px     : float | sympy symbol
        Horizontal momentum
    y      : float | sympy symbol
        Vertical coordinate
    py     : float | sympy symbol
        Vertical momentum
    delta  : float | sympy symbol
        Longitudinal momentum deviation
    k1     : float | sympy symbol
        Normalised quadrupole gradient (k1 = q b2 / (P0 r0) )
    """

    return HamDrift6D(beta0, gamma0, L, x, px, y, py, delta) + L * Rational(1, 2) * k1 * (
        x ** 2 - y ** 2
    )


def HamQuad4D(beta0, gamma0, L, x, px, y, py, k1):
    """
    4D thick quadruple Hamiltonian.

    Arguments:
    ----------
    beta0  : float | sympy symbol
        Relativistic Beta
    gamma0 : float | sympy symbol
        Relativistic gamma
    L      : float | sympy symbol
        Length of the drift
    x      : float | sympy symbol
        Horizontal coordinate [meter]
    px     : float | sympy symbol
        Horizontal momentum
    y      : float | sympy symbol
        Vertical coordinate
    py     : float | sympy symbol
        Vertical momentum
    delta  : float | sympy symbol
        Longitudinal momentum deviation
    k1     : float | sympy symbol
        Normalised quadrupole gradient (k1 = q b2 / (P0 r0) )
    """
    return HamQuad6D(beta0, gamma0, L, x, px, y, py, 0, k1)


def HamQuad6DParaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta, k1):
    return L * (
        half() * px ** 2
        + half() * py ** 2
        + half() * k1 * x ** 2
        - half() * k1 * y ** 2
        + half() * (delta / (beta0 * gamma0)) ** 2
    )


def HamQuad6DParaxialSecondOrderChroma(beta0, gamma0, L, x, px, y, py, delta, k1):
    D = sqrt(1 + 2 * delta / beta0 + delta ** 2)
    return L * (
        delta / beta0
        - D
        + half() * px ** 2 / D
        + half() * py ** 2 / D
        + half() * k1 * x ** 2
        - half() * k1 * y ** 2
    )


def HamSQuad6D(beta0, gamma0, L, x, px, y, py, delta, k1s):
    """
    6D thick skew quadruple Hamiltonian.

    Arguments:
    ----------
    beta0  : float | sympy symbol
        Relativistic Beta
    gamma0 : float | sympy symbol
        Relativistic gamma
    L      : float | sympy symbol
        Length of the drift
    x      : float | sympy symbol
        Horizontal coordinate [meter]
    px     : float | sympy symbol
        Horizontal momentum
    y      : float | sympy symbol
        Vertical coordinate
    py     : float | sympy symbol
        Vertical momentum
    delta  : float | sympy symbol
        Longitudinal momentum deviation
    k1     : float | sympy symbol
        Normalised quadrupole gradient (k1s = -q a2 / (P0 r0) )
    """

    return HamDrift6D(beta0, gamma0, L, x, px, y, py, delta) + L * Rational(1, 2) * k1s * x * y


def HamSQuad6DParaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta, k1s):
    """
    6D thick skew quadruple - paraxial second order approximation of the Hamiltonian.

    Arguments:
    ----------
    beta0  : float | sympy symbol
        Relativistic Beta
    gamma0 : float | sympy symbol
        Relativistic gamma
    L      : float | sympy symbol
        Length of the drift
    x      : float | sympy symbol
        Horizontal coordinate [meter]
    px     : float | sympy symbol
        Horizontal momentum
    y      : float | sympy symbol
        Vertical coordinate
    py     : float | sympy symbol
        Vertical momentum
    delta  : float | sympy symbol
        Longitudinal momentum deviation
    k1     : float | sympy symbol
        Normalised quadrupole gradient (k1s = -q a2 / (P0 r0) )
    """
    return L * (
        half() * px ** 2
        + half() * py ** 2
        + k1s * x * y
        + half() * (delta / (beta0 * gamma0)) ** 2
    )


def HamRFTM0106D(beta0, gamma0, L, x, px, y, py, z, delta, phi0, s, k, rho, Es, q, P0, omega):
    """
    omega/k = c
    """
    return HamDrift6D(beta0, gamma0, L, x, px, y, py, delta) + L * (
        (q / P0) * (Es / omega) * besselj(0, k * rho) * cos(k / beta0 * s - k * z + phi0)
    )


def HamRFTM0106DAvg(beta0, gamma0, L, x, px, y, py, z, delta, phi0, k, rho, Es, q, P0, omega):
    T = 2 * beta0 / pi * sin(pi / (2 * beta0))
    alpha = pi * q / P0 * Es / omega * T
    return HamDrift6D(beta0, gamma0, L, x, px, y, py, delta) - L * (
        alpha / pi * besselj(0, k * rho) * cos(phi0 - k * z)
    )


def HamRFTM0106DAvgParaxialSecondOrder(
    beta0, gamma0, L, x, px, y, py, z, delta, phi0, k, rho, Es, q, P0, omega
):
    T = 2 * beta0 / pi * sin(pi / (2 * beta0))
    alpha = pi * q / P0 * Es / omega * T
    return HamDrift6DParaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta) + L * (
        alpha / (4 * pi) * cos(phi0) * k ** 2 * (x ** 2 + y ** 2)
        - alpha / pi * sin(phi0) * k * z
        + alpha / (4 * pi) * cos(phi0) * k ** 2 * z ** 2
    )


def HamSolenoid6D(beta0, gamma0, L, x, px, y, py, delta, ks):
    """
    ks = 1/2 * q/P0*B0
    """
    return L * (
        (delta / beta0)
        - sqrt(
            (1 / beta0 + delta) ** 2
            - (px + ks * y) ** 2
            - (py - ks * x) ** 2
            - (1 / (beta0 * gamma0)) ** 2
        )
    )


def HamSolenoid6DparaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta, ks):
    return HamDrift6DParaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta) + L * (
        half() * ks ** 2 * x ** 2
        + half() * ks ** 2 * y ** 2
        - half() * ks * x * py
        + half() * ks * px * y
    )


# combined function magnets
def HamCombBend6DParaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta, k0, k1, h):
    """
    k0 = q/P0*b1
    k! = q/P0 * b2/r0
    """
    return HamDrift6DParaxialSecondOrder(beta0, gamma0, L, x, px, y, py, delta) + L * (
        (k0 - h) * x ** 2 - half() * k1 * y ** 2 - h / beta0 * x * delta
    )


def HamSext6D(beta0, gamma0, L, x, px, y, py, delta, k2):
    return HamDrift6D(beta0, gamma0, L, x, px, y, py, delta) + L * (
        1 / 6 * k2 * (x ** 3 - 3 * x * y ** 2)
    )

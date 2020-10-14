# Ref: Wolski Lectures on Linear Dynamics
from sympy import Derivative, Rational, S, Symbol, series, sqrt, symbols


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


def HamSext6D(beta0, gamma0, L, x, px, y, py, delta, k1):
    pass

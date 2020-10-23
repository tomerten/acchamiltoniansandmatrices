from sympy import Matrix, Rational, cos, cosh, pi, simplify, sin, sinh, sqrt, tan


def half():
    """
    Helper function to write half as a rational
    in a sympy expression.
    """
    return Rational(1, 2)


def CosPhi(M):
    """
    Returns the cos(phi) of the one turn map.
    """
    assert M.shape == (2, 2)
    return half() * M.trace()


def SinPhiOver2(M):
    """
    Returns the sin(phi/2) of the one turn map.
    """
    assert M.shape == (2, 2)
    return simplify(sqrt(half() * (1 - CosPhi(M))))


def SinPhi(M):
    """
    Returns the sin(phi) of the one turn map.
    """
    assert M.shape == (2, 2)
    return sqrt(1 - CosPhi(M) ** 2)


def CSBeta(M):
    """
    Returns Courant-Snyder beta.
    """
    assert M.shape == (2, 2)
    return M[0, 1] / SinPhi(M)


def CSAlpha(M):
    """
    Returns Courant-Snyder alpha.
    """
    assert M.shape == (2, 2)
    return half() * (M[0, 0] - M[1, 1]) / SinPhi(M)


def symbCSBeta(M, phi):
    """
    Return symbolic CS beta.
    """
    assert M.shape == (2, 2)
    return M[0, 1] / sin(phi)


def symbCSAlpha(M, phi):
    """
    Returns symbolic CS alpha.
    """
    assert M.shape == (2, 2)
    return (M[0, 0] - M[1, 1]) / (2 * sin(phi))

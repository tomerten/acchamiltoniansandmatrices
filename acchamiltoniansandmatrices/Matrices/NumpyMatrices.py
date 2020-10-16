from sympy import (
    Matrix,
    Rational,
    cos,
    cosh,
    lambdify,
    pi,
    sin,
    sinh,
    sqrt,
    symbols,
    tan,
)

from .SymbMatrices import (
    RsymbDipole,
    RsymbDipoleComb,
    RsymbDipoleFringe,
    RsymbDrift4D,
    RsymbDrift6D,
    RsymbQuad4D,
    RsymbQuad6D,
    RsymbQuad6DChroma,
    RsymbQuad6dThin,
    RsymbSQuad6D,
)

# Relativistic Symbols
beta0, gamma0 = symbols("beta_0 gamma_0")

# Phase-space coordinates
x, px, y, py, z, delta = symbols("x p_x y p_y z delta")

# Cury-linear coordinate
s = symbols("s")

# RF related symbols

# Lattice element related symbols
L, k0, k1, k1s, k2, h = symbols("L k_0 k_1 ks_1 k_2 h")

# Fringe fields
K1 = symbols("K_1")

# ARGS list
relArgs = (beta0, gamma0)
phsArgs4d = (x, px, y, py)
phsArgs5d = phsArgs4d + (delta,)

# DRIFT MATRICES
driftargs = relArgs + (L,)
RnpDrift6D = lambdify(driftargs, RsymbDrift6D(*driftargs), modules="numpy")
RnpDrift4D = lambdify(driftargs, RsymbDrift4D(*driftargs), modules="numpy")

# DIPOLE MATRICES
dipoleargs = relArgs + (L, k0)
RnpDipole = lambdify(dipoleargs, RsymbDipole(*dipoleargs), "numpy")
RnpDipoleComb = lambdify((dipoleargs + (k1,)), RsymbDipoleComb(*dipoleargs, k1), "numpy")
RnpDipoleFringe = lambdify(K1, RsymbDipoleFringe(K1), "numpy")

# QUADRUPOLE MATRICES
quadArgs = relArgs + (L, k1)
RnpQuad6D = lambdify(quadArgs, RsymbQuad6D(*quadArgs), "numpy")
RnpQuad4D = lambdify(quadArgs, RsymbQuad4D(*quadArgs), "numpy")
RnpQuad6DChroma = lambdify(quadArgs + (delta,), RsymbQuad6DChroma(*quadArgs, delta), "numpy")
RnpSQuad6D = lambdify(relArgs + (L, k1s), RsymbSQuad6D(*relArgs, L, k1s), "numpy")
RnpQuad6DThin = lambdify((L, k1), RsymbQuad6dThin(L, k1), "numpy")

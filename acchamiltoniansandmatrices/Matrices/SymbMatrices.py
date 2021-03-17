from sympy import I, Matrix, Rational, cos, cosh, pi, sin, sinh, sqrt, tan


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
    """
    4D R matrix for drift.

    Arguments:
    ----------
    beta0   : float | sympy symbol
        Relativistic beta
    gamma0  : float | sympy symbol
        Relativistic gamma
    L       : float | sympy symbol
        Length of the drift.
    """
    return RsymbDrift6D(beta0, gamma0, L)[0:4, 0:4]


def RsymbDipole6D(beta0, gamma0, L, k0):
    """
    6D R matrix for a dipole.

    Arguments:
    ----------
    beta0   : float | sympy symbol
        Relativistic beta
    gamma0  : float | sympy symbol
        Relativistic gamma
    L       : float | sympy symbol
        Length of the dipole.
    k0: float | sympy symbol
        Magnet strength
        k0 = 1/ rho , ol = omega * L = k0 * L
    """
    ol = k0 * L
    c = cos(ol)
    s = sin(ol)
    return Matrix(
        [
            [c, s / k0, 0, 0, 0, (1 - c) / (k0 * beta0)],
            [-k0 * s, c, 0, 0, 0, s / beta0],
            [0, 0, 1, L, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [
                -s / beta0,
                -(1 - c) / (k0 * beta0),
                0,
                0,
                1,
                L / (beta0 * gamma0) ** 2 - (k0 * L - s) / (k0 * beta0 ** 2),
            ],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RsymbDipoleSmallAngle(beta0, gamma0, L, k0):
    theta = k0 * L
    return Matrix(
        [
            [1, L, 0, 0, 0, L * theta * half() / beta0],
            [0, 1, 0, 0, 0, theta / beta0],
            [0, 0, 1, L, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [
                0,
                -L * theta * half() / beta0,
                0,
                0,
                1,
                L / (beta0 * gamma0) ** 2 - L / beta0 ** 2,
            ],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RsymbDipole4D(beta0, gamma0, L, k0):
    return RsymbDipole6D(beta0, gamma0, L, k0)[0:4, 0:4]


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


def RsymbQuad6DChroma(beta0, gamma0, L, k1, delta):
    D = sqrt(1 + 2 * delta / beta0 + delta ** 2)
    o = sqrt(k1 / D + 0 * I)
    ol = o * L
    od = o * D
    c = cos(ol)
    s = sin(ol)
    ch = cosh(ol)
    sh = sinh(ol)
    return Matrix(
        [
            [c, s / od, 0, 0, 0, 0],
            [-od * s, c, 0, 0, 0, 0],
            [0, 0, ch, sh / od, 0, 0],
            [0, 0, od * sh, ch, 0, 0],
            [0, 0, 0, 0, 1, L / (beta0 * gamma0) ** 2],
            [0, 0, 0, 0, 0, 1],
        ]
    )


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
            [half() * cp, half() / omega * sp, half() * cm, half() / omega * sm, 0, 0],
            [-half() * omega * sm, half() * cp, -half() * omega * sp, half() * cm, 0, 0],
            [half() * cm, half() / omega * sm, half() * cp, half() / omega * sp, 0, 0],
            [-half() * omega * sp, half() * cm, -omega * half() * sm, half() * cp, 0, 0],
            [0, 0, 0, 0, 1, L / (beta0 * gamma0) ** 2],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RsymbRFTM0106D(beta0, gamma0, L, phi0, q, P0, Es, omega):
    T = 2 * beta0 / pi * sin(pi / (2 * beta0))
    alpha = pi * q / P0 * Es / omega * T
    psit = sqrt(half() * pi * alpha * cos(phi0))
    psip = sqrt(pi * alpha * cos(phi0)) / (gamma0 * beta0)
    c = cos(psit)
    s = sin(psit)
    cp = cos(psip)
    sp = sin(psip)
    return Matrix(
        [
            [c, L / psit * s, 0, 0, 0, 0],
            [-psit / L * s, c, 0, 0, 0, 0],
            [0, 0, c, L / psit * s, 0, 0],
            [0, 0, -psit / L * s, c, 0, 0],
            [0, 0, 0, 0, cp, (1 / (beta0 * gamma0)) ** 2 * L / psip * sp],
            [0, 0, 0, 0, -(beta0 ** 2) * gamma0 ** 2 * psip / L * sp, cp],
        ]
    )


def RMsymbRFTM0106D(beta0, gamma0, L, phi0, q, P0, Es, omega):
    T = 2 * beta0 / pi * sin(pi / (2 * beta0))
    alpha = pi * q / P0 * Es / omega * T
    psip = sqrt(pi * alpha * cos(phi0)) / (gamma0 * beta0)
    return Matrix(
        [
            [0],
            [0],
            [0],
            [0],
            [2 / pi * L * sin((psip / 2)) * tan(phi0)],
            [alpha * sin(psip) / psip * sin(phi0)],
        ]
    )


def RsymbSolenoid(beta0, gamma0, L, ks):
    ol = ks * L
    c = cos(ol) ** 2
    s = sin(2 * ol)
    s2 = sin(ol) ** 2
    return Matrix(
        [
            [c, half() * s / ks, half() * s, s2 / ks, 0, 0],
            [-half() * ks * s, c, -ks * s2, half() * s, 0, 0],
            [-half() * s, -s2 / ks, c, half() * s / ks, 0, 0],
            [ks * s2, -half() * s, -half() * ks * s, c, 0, 0],
            [0, 0, 0, 0, 1, L / (beta0 * gamma0) ** 2],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RsymbDipoleFringe(K1):
    """
    K1 = -q/P0*B0*tan(psi)
    """
    return Matrix(
        [
            [1, 0, 0, 0, 0, 0],
            [-K1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, K1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RsymbDipoleComb(beta0, gamma0, L, h, k0, k1):
    """
    k0 = q/P0*b1
    k1=q/P0*b2/r0
    """
    ox = sqrt(h * k0 + k1)
    oy = sqrt(k1)
    cx = cos(ox * L)
    sx = sin(ox * L)
    chy = cosh(oy * L)
    shy = sinh(oy * L)
    hover = h / beta0
    return Matrix(
        [
            [cx, sx / ox, 0, 0, 0, hover * (1 - cx) / ox ** 2],
            [-ox * sx, cx, 0, 0, 0, hover * sx / ox],
            [0, 0, chy, shy / oy, 0, 0],
            [0, 0, oy * shy, chy, 0, 0],
            [
                -hover * sx / ox,
                -hover * (1 - cx) / ox ** 2,
                0,
                0,
                1,
                L / (beta0 * gamma0) ** 2 - hover ** 2 * (ox * L - sx) / ox ** 3,
            ],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RMsymbDipoleComb(beta0, gamma0, L, h, k0, k1):
    ox = sqrt(k0 ** 2 + k1)
    oy = sqrt(k1)
    cx = cos(ox * L)
    sx = sin(ox * L)
    chy = cosh(oy * L)
    shy = sinh(oy * L)
    kob = k0 / beta0
    return Matrix(
        [
            [(h - k0) * (1 - cx) / ox ** 2],
            [(h - k0) * sx / ox],
            [0],
            [0],
            [0],
            [0],
        ]
    )


# thin lens
def RsymbQuad6DThin(L, k1):
    f = 1 / (k1 * L)
    return Matrix(
        [
            [1, 0, 0, 0, 0, 0],
            [-1 / f, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1 / f, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )


def RsymbFODO(beta0, gamma0, L, f):
    fq = f
    fd = f
    return Matrix(
        [
            [1 - half() * L ** 2 / fq ** 2, L / fq * (L + 2 * fq), 0, 0, 0, 0],
            [L / (4 * fq ** 3) * (L - 2 * fq), 1 - half() * L ** 2 / fq ** 2, 0, 0, 0, 0],
            [0, 0, 1 - half() * L ** 2 / fd ** 2, -L / fd * (L - 2 * fd), 0, 0],
            [0, 0, -L / (4 * fd ** 3) * (L + 2 * fd), 1 - half() * L ** 2 / fd ** 2, 0, 0],
            [0, 0, 0, 0, 1, 2 * L / (beta0 * gamma0) ** 2],
            [0, 0, 0, 0, 0, 1],
        ]
    )

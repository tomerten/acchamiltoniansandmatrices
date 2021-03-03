from sympy import Matrix, Poly, cos, sin, symbols


def geteigenvects(R, symbols):
    eig = R.eigenvects()

    if eig[0][1] > 1:
        eig1 = eig[0][2][0]
        eig2 = eig[0][2][1]
    else:
        eig1 = eig[0][2][0]
        eig2 = eig[1][2][0]

    return (
        (eig1.T * Matrix([symbols[0], symbols[1]]))[0],
        (eig2.T * Matrix([symbols[0], symbols[1]]))[0],
    )


def prime_factors(n):
    """
    Adapted from:
    REF : https://stackoverflow.com/questions/15347174/python-finding-prime-factors
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return set(factors)


def RotationMatrix2D(angle):
    R = Matrix([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    return R


def explicitCheck(new_pols, rot_rep):
    for p in new_pols:
        if p == p.subs(rot_rep).expand():
            print(p)
        else:
            print("{} nok".format(p))


def getpoly(u, v, symorder, porder, symbols):
    """
    Returns polynomials of order below porder that
    are invariant under a rotation over an angle 2 pi / symorder.

    Arguments:
    ----------
    u           :   sympy expression
        complex eigen vector of rotation matrix
    v           :   sympy expression
        complex eigen vector of rotation matrix
    symorder    : order of rotational invariance
    porder      : max order of the polynomials to calculate

    Returns:
    --------
    List of invariant polynomials.
    """
    x = symbols[0]
    y = symbols[1]
    so = symorder
    o = porder

    so = symorder
    o = porder

    pols = []
    for a in range(porder + 1):
        for b in range(porder + 1):
            if (a - b) % so == 0 and (a + b <= o):
                # print(a, b, a - b, a + b)
                #                 print(a,b,a-b)
                ex = (u ** a * v ** b).expand().as_real_imag()
                # print(ex)
                if ex[0] != 0:
                    pols.append(ex[0])
                if ex[1] != 0:
                    pols.append(ex[1])

    # xy acts like a length for even symorders
    if symorder >= 4 and symorder % 2 == 0:
        for i in range(1, porder // 4 + 1):
            pols.append(((x ** 2 * y ** 2) ** i).expand())

    # remove double entries with opposite sign
    new_pols = []
    for p in pols:
        if (p in new_pols) or (-p in new_pols):
            continue
        new_pols.append(p)

    return new_pols


def codict(expr, *x):
    collected = Poly(expr, *x).as_expr()
    return dict(i.as_independent(*x)[::-1] for i in Add.make_args(collected))

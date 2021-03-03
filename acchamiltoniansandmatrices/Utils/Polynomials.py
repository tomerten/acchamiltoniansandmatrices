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


def getpoly(u, v, symorder, porder):
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
    so = symorder
    o = porder

    pols = []
    for a in range(5):
        for b in range(5):
            if (a - b) % so == 0 and (a + b <= o):
                #                 print(a,b,a-b)
                ex = (u ** a * v ** b).expand().as_real_imag()
                if ex[0] != 0:
                    pols.append(ex[0])
                if ex[1] != 0:
                    pols.append(ex[1])

    # xy acts like a length for even symorders
    if symorder >= 4 and symorder % 2 == 0:
        from sympy import symbols

        x, y = symbols("x y")
        for i in range(1, porder // 4 + 1):
            pols.append(((x ** 2 * y ** 2) ** i).expand())

    # remove double entries with opposite sign
    new_pols = []
    for p in pols:
        if (p in new_pols) or (-p in new_pols):
            continue
        new_pols.append(p)

    return new_pols

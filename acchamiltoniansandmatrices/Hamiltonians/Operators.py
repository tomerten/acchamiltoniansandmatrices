from sympy import S


def Poisson(a, b, q_list, p_list):
    hp = S(0)
    for qi, pi in zip(q_list, p_list):
        hp += a.diff(qi) * b.diff(pi)
        hp -= a.diff(pi) * b.diff(qi)
    return hp

from sympy import Function, symbols

# coord
s = symbols("s")

# magnetic
B = symbols("B")

# functions
x = Function("x")(s)
y = Function("y")(s)
rho = Function("rho")(s)
By = Function("By")(x)


def Hill_Bend_Quad():
    """
    Returns the Hill's equations.
    """
    eq1 = x.diff(s, 2) + (1 / rho ** 2 - 1 / (B * rho))

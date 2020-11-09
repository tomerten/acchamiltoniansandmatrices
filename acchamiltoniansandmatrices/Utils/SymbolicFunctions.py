from sympy import Derivative, Eq, NumberSymbol, latex, solve, symbols


def pla(e):
    """
    Help function to generate latex for using in the markdown sections during the
    creation of this notebook.
    """
    print(latex(e))


class SymbolTrick(NumberSymbol):
    """
    https://stackoverflow.com/questions/39665207/define-a-variable-in-sympy-to-be-a-constant
    """

    def __new__(self, name):
        obj = NumberSymbol.__new__(self)
        obj._name = name
        return obj

    __str__ = lambda self: str(self._name)
    _as_mpf_val = 1.0

    def _latex(self, printer, *args):
        return r"{}".format(self._name)


def fixedpoints2D(ham, x, px):
    partialderiv_x = Derivative(ham, x)
    partialderiv_p = Derivative(ham, px)
    H_x = partialderiv_x.doit()
    H_p = partialderiv_p.doit()

    # Using the full drift Hamiltonian requires more computing power therefore takes much more time
    # Should implement parallel computing i.e. run on all cores

    x_FP = Eq(H_x, 0)  # dH/dx = 0
    p_FP = Eq(H_p, 0)  # dH/dp = 0

    # compute fixed points
    fixed_points = solve((x_FP, p_FP), x, px)
    return fixed_points

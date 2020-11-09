from sympy import NumberSymbol, latex


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

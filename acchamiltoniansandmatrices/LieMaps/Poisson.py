"""The Poisson bracket = {A,B} = Sum_i (Der(A,q_i)*Der(B,p_i) - Der(A,p_i)*Der(B,q_i) """

from __future__ import division, print_function

from sympy import Add, Expr, Function, Mul, Pow, S, symbols
from sympy.core.decorators import _sympifyit, call_highest_priority
from sympy.core.function import UndefinedFunction
from sympy.physics.quantum import Operator
from sympy.physics.quantum.operator import Operator
from sympy.printing.latex import print_latex
from sympy.printing.pretty.stringpict import prettyForm

__all__ = ["PoissonBracket"]

# -----------------------------------------------------------------------------
# Poisson Bracket
# -----------------------------------------------------------------------------


class PoissonBracket(Expr):
    """The Poisson Bracket, in an unevaluated state.

    Evaluating a Poisson Bracket is defined [1]_ as:
    ``{A,B} = Sum_i (Der(A,q_i)*Der(B,p_i) - Der(A,p_i)*Der(B,q_i)``.
    This class returns the Poisson Bracket in an unevaluated form. To evaluate the Poisson
    Bracket, use the ``.doit()`` method.

    Arguments:
    ==========

    A: Expr
        The first argument of the bracket {A,B}
    B: Expr
        The second argument of the bracket {A,B}


    IMPORTANT:
    ==========
    As the Poisson Bracket depends on taking partial derivatives with respect to
    phase-space coordinates, these need to be provided in order for the ``.doit()`` method
    to work. They can be provided through kwargs: coords=list, mom=list) at the creation
    of the instance or after creation by setting indep_coords and indep_mom.

    When working with generic functions, for symbolic evaluation, one needs to define the
    inputs as non-commutative for the ordering of Mul and Add to work correctly when combining


    Examples
    ========

    >>> from acchamiltonianandmatrices.LieMaps.Poisson import PoissonBracket
    >>> from sympy import Function, symbols
    >>> A = Function('A', commutative=False)
    >>> B = Function('B', commutative=False)
    >>> C = Function('C', commutative=False)
    >>> D = Function('D', commutative=False)

    Create a Poisson Bracket and use ``.doit()`` to evaluate it:

    >>> pb = PoissonBracket(A,B)
    >>> pb
    {A,B}
    """

    #     _op_priority = 11.0
    _is_commutative = False

    def __new__(cls, A, B, **kwargs):
        r = cls.eval(A, B)
        if r is not None:
            return r
        obj = Expr.__new__(cls, A, B)
        obj._A = A
        obj._B = B
        obj._indep_coords = kwargs.get("coords", None)
        obj._indep_mom = kwargs.get("mom", None)
        return obj

    @classmethod
    def eval(cls, a, b):
        if not (a and b):
            return S.Zero
        if a == b:
            return S.Zero

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, value):
        self._A = value

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, value):
        self._B = value

    @property
    def indep_coords(self):
        return self._indep_coords

    @indep_coords.setter
    def indep_coords(self, value):
        self._indep_coords = value

    @property
    def indep_mom(self):
        return self._indep_mom

    @indep_mom.setter
    def indep_mom(self, value):
        self._indep_mom = value

    @property
    def free_symbols(self):
        return self.A.free_symbols.union(self.B.free_symbols)

    @property
    def expr(self):
        if self.indep_coords and self.indep_mom:
            hp = S(0)
            for qi, pi in zip(self.indep_coords, self.indep_mom):
                hp += self.A.diff(qi) * self.B.diff(pi)
                hp -= self.A.diff(pi) * self.B.diff(qi)
            return hp
        else:
            return self

    def _eval_derivative(self, symbol):
        """
        Explicitly perform the derivative. This is
        necessary to get the full expanded expression
        when one of the arguments is itself a PoissonBracket.
        In other words to allow evalutation of nested Poisson Brackets.
        """
        new_expr = self.expr.diff(symbol)

        return new_expr

    def doit(self, **hints):
        """ Evaluate commutator """
        A = self.A
        B = self.B

        if isinstance(A, Operator) and isinstance(B, Operator):
            try:
                comm = A._eval_commutator(B, **hints)
            except NotImplementedError:
                try:
                    comm = -1 * B._eval_commutator(A, **hints)
                except NotImplementedError:
                    comm = None
            if comm is not None:
                return comm.doit(**hints)
        if isinstance(A, UndefinedFunction) and isinstance(B, UndefinedFunction):
            return self

        return self.expr

    def _expand_pow(self, A, B, sign):
        exp = A.exp
        if not exp.is_integer or not exp.is_constant() or abs(exp) <= 1:
            # nothing to do
            return self
        base = A.base
        if exp.is_negative:
            base = A.base ** -1
            exp = -exp
        comm = PoissonBracket(base, B).expand(commutator=True)

        result = base ** (exp - 1) * comm
        for i in range(1, exp):
            result += base ** (exp - 1 - i) * comm * base ** i
        return sign * result.expand()

    def _eval_expand_commutator(self, **hints):
        A = self.A  # self.args[0]
        B = self.B  # self.args[1]

        if isinstance(A, Add):
            # [A + B, C]  ->  [A, C] + [B, C]
            sargs = []
            for term in A.args:
                comm = PoissonBracket(term, B)
                if isinstance(comm, PoissonBracket):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            return Add(*sargs)

        elif isinstance(B, Add):
            # [A, B + C]  ->  [A, B] + [A, C]
            sargs = []
            for term in B.args:
                comm = PoissonBracket(A, term)
                if isinstance(comm, PoissonBracket):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            return Add(*sargs)

        elif isinstance(A, Mul):
            # [A*B, C] -> A*[B, C] + [A, C]*B
            a = A.args[0]
            b = Mul(*A.args[1:])
            c = B
            comm1 = PoissonBracket(b, c)
            comm2 = PoissonBracket(a, c)
            if isinstance(comm1, PoissonBracket):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, PoissonBracket):
                comm2 = comm2._eval_expand_commutator()
            first = Mul(a, comm1)
            second = Mul(comm2, b)
            return Add(first, second)

        elif isinstance(B, Mul):
            # [A, B*C] -> [A, B]*C + B*[A, C]
            a = A
            b = B.args[0]
            c = Mul(*B.args[1:])
            comm1 = PoissonBracket(a, b)
            comm2 = PoissonBracket(a, c)
            if isinstance(comm1, PoissonBracket):
                comm1 = comm1._eval_expand_commutator()
            if isinstance(comm2, PoissonBracket):
                comm2 = comm2._eval_expand_commutator()
            first = Mul(comm1, c)
            second = Mul(b, comm2)
            return Add(first, second)

        elif isinstance(A, Pow):
            # [A**n, C] -> A**(n - 1)*[A, C] + A**(n - 2)*[A, C]*A + ... + [A, C]*A**(n-1)
            return self._expand_pow(A, B, 1)

        elif isinstance(B, Pow):
            # [A, C**n] -> C**(n - 1)*[C, A] + C**(n - 2)*[C, A]*C + ... + [C, A]*C**(n-1)
            return self._expand_pow(B, A, -1)

        return self

    def _latex(self, printer, *args):
        news = []
        for arg in self.args[:2]:
            if isinstance(arg, Add):
                news.append(
                    " + ".join(
                        [
                            printer._print(a.func, *args)
                            if a.is_Function
                            else printer._print(a, *args)
                            for a in arg.args
                        ]
                    )
                )

            elif isinstance(arg, PoissonBracket):
                news.append(
                    "\\lbrace %s,%s\\rbrace"
                    % tuple(
                        [
                            printer._print(a.func, *args)
                            if a.is_Function and not isinstance(a, UndefinedFunction)
                            else printer._print(a, *args)
                            for a in arg.args[:2]
                        ]
                    )
                )

            elif isinstance(arg, Mul):
                news.append(" * ".join([printer._print(arg, *args) for arg in self.args[:2]]))

            elif isinstance(arg, Pow):
                if isinstance(arg.args[0], UndefinedFunction):
                    news.append("{}^{}".format([printer._print(a) for a in arg.args]))
                elif isinstance(arg.args[0], Function):
                    news.append(
                        "{}^{}".format(
                            printer._print(arg.args[0].func, *args),
                            printer._print(arg.args[1], *args),
                        )
                    )
                else:
                    news.append(printer._print(arg, *args))

            elif isinstance(arg, Function) and not (isinstance(arg, UndefinedFunction)):
                news.append(printer._print(arg.func, *args))

            else:  # isinstance(arg, Function) and isinstance(arg,UndefinedFunction):
                news.append(printer._print(arg, *args))

        print(news)
        return "\\lbrace %s,%s\\rbrace" % tuple(news)

    # legacy - kept for demo
    # return "\\lbrace %s,%s\\rbrace" % tuple(
    #     [
    #         printer._print(arg.func, *args)
    #         if arg.is_Function and not (isinstance(arg, UndefinedFunction))
    #         else printer._print(arg, *args)
    #         for arg in self.args[:2]
    #     ]
    # )

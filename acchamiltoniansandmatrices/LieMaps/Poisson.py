from __future__ import division, print_function

from sympy import (
    Add,
    Derivative,
    Expr,
    Function,
    Mul,
    Pow,
    Rational,
    S,
    simplify,
    symbols,
)
from sympy.core.decorators import _sympifyit, call_highest_priority
from sympy.core.function import UndefinedFunction
from sympy.physics.quantum import Operator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator
from sympy.printing.latex import print_latex
from sympy.printing.pretty.stringpict import prettyForm


class PoissonBracket(Expr):
    """
    Operator class for the Poisson Bracket - allows to have it as a symbolic operator
    """

    #     _op_priority = 11.0
    _is_commutative = False

    def __new__(cls, A, B, debug=False, **kwargs):
        # before return the object, do basic
        # testing and formatting of the bracket.
        # - orders arguments and updated sign
        # - returns S.Zero if:
        #    * one of the argumnents is missing
        #    * the arguments are equal
        #    * one of the arguments is a sympy number
        r = cls.eval(A, B, debug=debug, **kwargs)

        # if a valid object is returned from eval
        # than return this object
        if r is not None:
            if debug:
                print("Eval returned a value")
            return r

        # if eval has not returned anything
        # create a new object and return that one.
        obj = Expr.__new__(cls, A, B)
        obj.A = A
        obj.B = B

        # if independent coordinates and momenta are
        # provided store them in the object
        obj.indep_coords = kwargs.get("coords", None)
        obj.indep_mom = kwargs.get("mom", None)

        # store if the ordering has been reversed
        # this is used in the doit method, otherwise
        # the sign is wrong.
        obj.order = 1

        # set debug to true to get intermediate outputs
        # for code debugging
        obj.debug = debug
        return obj

    @classmethod
    def eval(cls, a, b, debug, **kwargs):
        # check if both arguments of the
        # bracket are given otherwise return Zero
        if not (a and b):
            if debug:
                print("Missing term")
            return S.Zero

        # check if both arguments of the bracket
        # are equal, if yes return Zero
        if a == b:
            if debug:
                print("Bracket of equal aruments is Zero.")
            return S.Zero

        # check if one of the bracket arguments is
        # a sympy number, if yes return Zero.
        if a.is_number or b.is_number:
            if debug:
                print("At least one argument is a number, bracket is Zero")
            return S.Zero

        # order the elements
        #
        # IMPORTANT NOTE:
        # ---------------
        # Something tricky I do not fully understand:
        # In order for this to work below "-nob" needs
        # to be returned, when returning "S.NegativeOne * nob",
        # self.indep_coords and self.indep_mom are overwritten
        # with None such that the expr method fails.
        # Probably has something to do with how sympy handles
        # "Mul" objects, but I was unable to understand the cause
        #

        if isinstance(a, UndefinedFunction):
            tmp = symbols("tmp")
            if isinstance(b, UndefinedFunction):
                if a(tmp).compare(b(tmp)) == 1:
                    if debug:
                        print(
                            "Interchanging order of arguments and put minus sign in front of bracket."
                        )
                    nob = cls(b, a, **kwargs)
                    nob.order = -1
                    return -nob
            else:
                if a(tmp).compare(b) == 1:
                    if debug:
                        print(
                            "Interchanging order of arguments and put minus sign in front of bracket."
                        )
                    nob = cls(b, a, **kwargs)
                    nob.order = -1
                    return -nob
        elif isinstance(b, UndefinedFunction):
            tmp = symbols("tmp")
            if a.compare(b(tmp)) == 1:
                if debug:
                    print(
                        "Interchanging order of arguments and put minus sign in front of bracket."
                    )
                nob = cls(b, a, **kwargs)
                nob.order = -1
                return -nob
        else:
            if a.compare(b) == 1:
                if debug:
                    print(
                        "Interchanging order of arguments and put minus sign in front of bracket."
                    )
                nob = cls(b, a, **kwargs)
                nob.order = -1
                return -nob

    @property
    def free_symbols(self):
        """
        Return the free symbols.
        """
        return self.A.free_symbols.union(self.B.free_symbols)

    @property
    def expr(self):
        """
        Return the full expression.
        """
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
        Necessary to get the full expanded expression
        when one of the arguments is itself a PoissonBracket.
        In other words to allow evalutation of nested Poisson Brackets.
        """
        new_expr = self.expr.diff(symbol)
        return new_expr

    def doit(self, debug=False, **hints):
        """ Evaluate commutator """
        A = self.args[0]
        B = self.args[1]
        order = self.order
        if debug:
            print(self)
            print(self.__class__)
            print(self.args)
            try:
                print(self.expr)
            except:
                print("expr failed")
                pass

        if order == 1:
            if debug:
                print("order 1")
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

            if isinstance(A, UndefinedFunction) or isinstance(B, UndefinedFunction):
                if debug:
                    print("One is undefined function")
                return self

            elif isinstance(A, Function) or isinstance(B, Function):
                if debug:
                    print("in Function")
                try:
                    return self.expr
                except RecursionError:
                    print("Check coords and mom if you expected this to evaluate.")
                    return self

            return self.expr

        else:
            if debug:
                print("order -1")

            if isinstance(A, UndefinedFunction) or isinstance(B, UndefinedFunction):
                if debug:
                    print("One is undefined function")
                    print(self)
                h = self
                return h

            elif isinstance(A, Function) or isinstance(B, Function):
                if debug:
                    print("in Function")

                try:
                    h = self.expr
                    return h
                except RecursionError:
                    print("Check coords and mom if you expected this to evaluate.")
                    return self

            return (
                S.NegativeOne
                * PoissonBracket(A, B, coords=self.indep_coords, mom=self.indep_mom).expr
            )

    def _eval_expand_commutator(self, **hints):
        A = self.A  # self.args[0]
        B = self.B  # self.args[1]

        if isinstance(A, Add):
            # [A + B, C]  ->  [A, C] + [B, C]
            sargs = []
            for term in A.args:
                comm = PoissonBracket(term, B)
                print(comm.__class__)
                if isinstance(comm, PoissonBracket):
                    comm = comm._eval_expand_commutator()
                sargs.append(comm)
            #             print(sargs)
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
            #             print(first,second)
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

    def _latex(self, printer, *args):
        news = []
        for arg in self.args[:2]:
            if isinstance(arg, Add):
                news.append(
                    " + ".join(
                        [
                            printer.doprint(a.func, *args)
                            if a.is_Function
                            else printer.doprint(a, *args)
                            for a in arg.args
                        ]
                    )
                )

            elif isinstance(arg, PoissonBracket):
                news.append(printer.doprint(arg, *args))

            elif isinstance(arg, Mul):
                news.append(
                    " * ".join(
                        [
                            printer.doprint(a.func, *args)
                            if a.is_Function
                            else printer.doprint(a, *args)
                            for a in arg.args
                        ]
                    )
                )
            elif isinstance(arg, Pow):
                if isinstance(arg.args[0], UndefinedFunction):
                    news.append("{}^{}".format([printer.doprint(a) for a in arg.args]))
                elif isinstance(arg.args[0], Function):
                    news.append(
                        "{}^{}".format(
                            printer.doprint(arg.args[0].func, *args),
                            printer.doprint(arg.args[1], *args),
                        )
                    )
                else:
                    news.append(printer.doprint(arg, *args))

            elif isinstance(arg, Function) and not (isinstance(arg, UndefinedFunction)):
                news.append(printer.doprint(arg.func, *args))

            else:  # isinstance(arg, Function) and isinstance(arg,UndefinedFunction):
                news.append(printer.doprint(arg, *args))

        return "\\lbrace %s,%s\\rbrace" % tuple(news)

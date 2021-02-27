from sympy import Add


def PrintAllTerms(expr):
    for i, t in enumerate(Add.make_args(expr)):
        print("Term number {} \n {t}".format(i, t))

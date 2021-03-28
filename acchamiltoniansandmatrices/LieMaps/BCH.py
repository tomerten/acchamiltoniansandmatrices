import numpy as np
import pandas as pd
from sympy import Function, Rational, symbols

from .Poisson import PoissonBracket

# ref : https://www.youtube.com/watch?v=E7V36JvHbsA


################################################################################
#
# BCH coefficients are calculated from bicolored rooted trees.
# The algorithm is not implemented here completely yet but at least
# the relevant bicolored trees can be generated that are used in
# the coefficient calculation.
#
# ref: http://www.gicas.uji.es/Fernando/MyPapers/JMP09.pdf
#
################################################################################


class bicoloredTree:
    """ Bicolored Rooted Tree object"""

    def __init__(self, root):
        assert isinstance(root, Node)
        self.root = root
        self.children = []
        self.Nodes = []

    def addNode(self, node):
        assert isinstance(node, Node)
        self.children.append(node)

    def getAllNodes(self):
        self.Nodes.append(self.root)
        for child in self.children:
            self.Nodes.append(child.color)
        for child in self.children:
            if child.getChildNodes(self.Nodes) != None:
                child.getChildNodes(self.Nodes)

    def __mul__(self, other):
        """
        Overwrite multiplication.
        Right hand tree is grafted on the root
        of the left hand tree.
        """
        newtree = bicoloredTree(self.root)
        for child in self.children:
            newtree.children.append(child)
        newtree.children.append(other)

        return newtree

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.root.color) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


class Node:
    """ Node class for use in bicolored rooted tree"""

    def __init__(self, color):
        assert color in ["black", "white"]
        self.color = color
        self.children = []

    def addNode(self, node):
        self.children.append(node)

    def getChildNodes(self, Tree):
        for child in self.children:
            if child.children:
                child.getChildNodes(Tree)
                Tree.append(child.color)

            else:
                Tree.append(child.color)

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.color) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


def generate_bch_expression(order=4, coords=[], mom=[]):
    """
    Generate BCH expression up to order 'order'.

    ref: http://www.gicas.uji.es/Fernando/MyPapers/JMP09.pdf

    coefficients file: http://www.gicas.uji.es/research

    Arguments:
    ----------
    order   : BCH order
    coords  : independent coordinates to use in the Poisson Brackets
    mom     : independent momenta to use in the Poisson Brackets

    """
    X = Function("X")(*coords, *mom)
    Y = Function("Y")(*coords, *mom)

    # read in table with coefficients
    dat = pd.read_csv(
        "bchHall20.dat",
        index_col=None,
        header=None,
        delim_whitespace=True,
        names=["i'", 'i"', "p", "q"],
    )

    # init of the dictionaries

    iprime = {1: 1, 2: 2}
    idprime = {1: 0, 2: 0}
    ia = {1: 1, 2: 1}
    ei = {1: X, 2: Y}
    si = {1: 1, 2: 1}
    ki = {1: 0, 2: 0}

    roottreeblack = bicoloredTree(Node("black"))
    roottreewhite = bicoloredTree(Node("white"))
    ui = {1: roottreeblack, 2: roottreewhite}

    # init loop dummy variable
    i = 3

    # generate the table
    for n in range(2, order + 1):
        for j in range(1, i):
            for k in range(j + 1, i):
                if ia[j] + ia[k] == n and j >= idprime[k]:
                    idprime[i] = j
                    iprime[i] = k
                    ia[i] = n

                    # ki
                    if idprime[iprime[i]] != idprime[i]:
                        ki[i] = 1
                    else:
                        ki[i] = ki[iprime[i]] + 1
                    si[i] = ki[i] * si[iprime[i]] * si[idprime[i]]
                    ui[i] = ui[iprime[i]] * ui[idprime[i]]  # rooted tree diagrams
                    ei[i] = PoissonBracket(ei[iprime[i]], ei[idprime[i]], coords=coords, mom=mom)
                    # print(
                    #    "i:{:2} , |i|:{:2}, i':{:2}, i\":{:2}, s_i:{:2} ".format(
                    #        i, ia[i], iprime[i], idprime[i], si[i], display(ei[i])
                    #    )
                    # )
                    i = i + 1

    # generate the expression
    res = X + Y

    for i in range(2, order + 1):
        # get the relevant brackets
        ilist = [k for k, v in ia.items() if v == i]
        bracketlist = [ei[j] for j in ilist]

        # get the coeffcients
        coefflist = [Rational(dat.loc[j, "p"], dat.loc[j, "q"]) for j in ilist]

        for c, b in zip(coefflist, bracketlist):
            res += c * b

    return res

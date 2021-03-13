import numpy as np
import pandas as pd
from sympy import (
    Derivative,
    Eq,
    Function,
    Matrix,
    Poly,
    Rational,
    S,
    Symbol,
    collect,
    cos,
    cosh,
    init_printing,
    lambdify,
    latex,
    limit,
    oo,
    series,
    simplify,
    sin,
    sinh,
    solve,
    sqrt,
    symbols,
)

init_printing()

#%matplotlib notebook
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from pymad_hzb.PlotTools import new_plot_elems_madx
from sympy.printing.latex import print_latex

from acchamiltoniansandmatrices.Hamiltonians.LatticeElementHamiltonians import (
    HamDrift6D,
    HamDrift6DParaxialSecondOrder,
    HamQuad6D,
    HamQuad6DParaxialSecondOrder,
    HamQuad6DParaxialSecondOrderChroma,
)
from acchamiltoniansandmatrices.Hamiltonians.Operators import Poisson
from acchamiltoniansandmatrices.LieMaps.LieOperator import LieOperator
from acchamiltoniansandmatrices.LieMaps.Poisson import PoissonBracket
from acchamiltoniansandmatrices.Matrices.NumpyMatrices import (
    RnpDrift6D,
    RnpFODO,
    RnpQuad6D,
    RnpQuad6DChroma,
    RnpQuad6DThin,
)
from acchamiltoniansandmatrices.Matrices.SymbMatrices import (
    RsymbDrift6D,
    RsymbFODO,
    RsymbQuad6D,
    RsymbQuad6DChroma,
    RsymbQuad6DThin,
)
from acchamiltoniansandmatrices.Tracking.LinearMatrixTracking import (
    GenerateNDimCoordinateGrid,
    LinMap,
    nestList,
)
from acchamiltoniansandmatrices.Utils.JupyterHelpFunctions import hide_toggle
from acchamiltoniansandmatrices.Utils.SymbolicFunctions import (
    SymbolTrick,
    fixedpoints2D,
)

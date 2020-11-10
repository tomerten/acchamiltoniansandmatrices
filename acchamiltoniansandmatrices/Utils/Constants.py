from numpy import pi as npi
from scipy import constants as const
from scipy.constants import physical_constants as pc
from sympy import symbols
from sympy.pyhsics.quantum.constants import hbar

# symbolic expressions
m, c, re, e, alpha, pi = symbols("m c r_e e alpha pi")

SymExpRestEnergy = m * c ** 2
SymExpElectronClassicalRadius = e ** 2 / SymExpRestEnergy
SymExprComptonWavelengthOverTwoPi = hbar / (m * c)
SymExprFineStructureConst = e ** 2 / (hbar * c)
SymExprImpedanceFreeSpace = 4 * pi / c
SymExprAlferovCurrent = e * c / SymExpElectronClassicalRadius

# electron rest energy
ELEMENTARY_CHARGE = const.e
ELECTRON_REST_ENERGY_MEV = pc["electron mass energy equivqlent in MeV"]
ELECTRON_CLASSICAL_RADIUS = pc["classical electron radius"]
ELECTRON_COMPTON_WAVELENGTH_OVER_TWO_PI = pc["reduced Compton wavelength"]
FINE_STRUCTURE_CONSTANT = const.alpha
IMPEDANCE_OF_FREE_SPACE = 4 * npi / const.c
ALFEROV_CURRENT = ELEMENTARY_CHARGE * const.c / ELECTRON_CLASSICAL_RADIUS

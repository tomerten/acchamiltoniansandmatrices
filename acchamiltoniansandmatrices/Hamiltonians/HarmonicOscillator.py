from sympy import Rational, lambdify, symbols, sympify

x, p, m, omega = symbols("x p m omega")


def PotentialSymb1DHarmonicOscillator(x, m, omega):
    """ Potential Harmonic Oscillator """
    return Rational(1, 2) * m * omega ** 2 * x ** 2


potargs = (x, m, omega)
Potentialnp1DHarmonicOscillator = lambdify(
    potargs, PotentialSymb1DHarmonicOscillator(*potargs), "numpy"
)


def HamSymb1DHarmonicOscillator(x, p, m, omega):
    """ Hamiltonian Harmonic Oscillator """
    return p ** 2 / (2 * m) + Rational(1, 2) * m * omega ** 2 * x ** 2


hamargs = (x, p, m, omega)
Hamnp1DHarmonicOscillator = lambdify(hamargs, HamSymb1DHarmonicOscillator(*hamargs), "numpy")

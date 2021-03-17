import collections
import copy
import functools
import operator
from functools import reduce  # Required in Python 3
from operator import mul

import mpmath as mp
import numpy as np
from mpmath import fac
from sympy import O, init_printing, poly, symbols
from sympy.core.decorators import _sympifyit, call_highest_priority
from termcolor import colored
from tqdm import tqdm

from .. import log
from ..LieMaps.LieOperator import LieOperator

# define symbols to use in the functions
sym_x, sym_y, sym_z, sym_px, sym_py, sym_pz = symbols("x y z p_x p_y delta")


def get_indep_coords_taylor(taylor):
    """
    Method the get the free symbols or
    independent variables from a Taylor
    vector.

    Arguments:
    ----------
    taylor  :   taylor vector
        input list of polynomials

    """
    coords = set()
    for p in taylor:
        coords = coords.union(poly(p).free_symbols)

    return list(coords)


def prod(iterable):
    """
    akin to built - in sum() but for product
    """
    return reduce(operator.mul, iterable, 1)


def deg2_lie(taylor, coords):
    """
    Method to extract the linear transfer matrix from the linear part of the Taylor
    polynomial.

    Arguments:
    ----------
    taylor: polynomial
        Taylor polynomial

    coords: list
        independent coordinates used in the Taylor vector
    """
    log.setup()
    log.info("Extracting linear transfer matrix from linear part of {}".format(taylor))

    I = np.identity(6)
    R = []
    i = len(taylor)

    for polynomial in taylor:
        p = poly(polynomial, *coords)  # sym_x, sym_px, sym_y, sym_py, sym_z, sym_pz)
        monomials = p.monoms()
        coeffs = p.coeffs()

        log.debug("Monomials      {}".format(monomials))
        log.debug("Monomial coeff {}".format(coeffs))

        Matrow = []
        for j, row in enumerate(I):
            try:
                index = monomials.index(tuple(row))
                Matrow.append(coeffs[index])
            except:
                if j < i:  # make sure the matrix stays quadratic
                    Matrow.append(0)

        R.append(Matrow)

    # convert to numpy so linalg can invert it
    R = np.asarray(R, dtype="float")

    return R


def taylor_to_lie(taylor, degree, coords):
    """
    Taylor map to Lie map.

    Arguments:
    ----------
    taylor:
        Taylor map vector.
    degree: int
        degree of the desired Lie map.
    coords: list
        independent coordinates
    """
    log.setup()
    log.info("Transforming Taylor to Lie.")
    log.info("coords are {}".format(coords))
    # decide if linear or nonlinear
    if degree == 2:
        lie = deg2_lie(taylor, coords)  # linear case
    else:
        lie = degN_lie(taylor, degree, coords)  # higher order maps

    return lie


def degN_lie(taylor, degree, coords):  # higher order Lie maps
    """
    Method to transform higher order Taylor maps to higher order
    Lie maps (beyond the linear parts).

    Arguments:
    ----------
    taylor:
        Taylor map vector.
    degree: int
        degree of the desired Lie map.
    coords: list
        independent coordinates

    """
    # define temporary symbol to deal with the order
    # of the differnt polynomials
    _epstemp = symbols("e")

    # immutable variable tuple
    variables = tuple(coords)

    # immutable tuple to track the sorting order of the various derivatives
    derivatives = (1, 0, 3, 2, 5, 4)  # order: d/dpx, d/dx, d/dpy, d/dy, d/dz, d/dz

    f = poly(
        0,
        *variables
        # sym_x, sym_px, sym_y, sym_py, sym_z, sym_pz
    )  # Lie map hom poly generated from taylor

    for var, polynomial in enumerate(taylor):
        # make sympy polynomial from the polynomial
        p = poly(polynomial, *variables)  # sym_x, sym_px, sym_y, sym_py, sym_z, sym_pz)

        # generate an array of the monomial degrees
        order = [sum(mon) for mon in p.monoms()]

        # iterate over all monomials in taylor
        for index, monomial in enumerate(p.monoms()):
            # check hom level -> derivative is order - 1
            if order[index] == (degree - 1):
                # print(variables,monomial)
                # reconstruct monomial
                mon = prod(a ** b for a, b in zip(variables, monomial))

                # avoid double sum of same coeff
                if (f.coeff_monomial(mon * variables[derivatives[var]])) == 0:
                    # normalize derivative power
                    power = monomial[derivatives[var]]
                    f = f + (p.coeffs()[index] / (power + 1.0)) * mon * variables[
                        derivatives[var]
                    ] * (-1) ** (var)

                    print(f)

    return f.subs(_epstemp, 0)


def dragt_finn_factorization(taylor, coords):
    """
    Dragt-Finn factorization of a Taylor map vector.

    Arguments:
    ----------
    taylor:

    debug: bool
        Debug Flag, if set, intermediate steps are written to log.
    coords: list
        independent coordinates


    """
    log.setup()
    log.info("Starting dragt-finn factorization.")

    LieProduct = []
    degree = 0

    for polynomial in taylor:
        # highest degree of hom poly in Lie map
        comp_degree = poly(polynomial).total_degree()

        if degree <= comp_degree:
            degree = comp_degree

    for i in range(2, degree + 2):
        # coeff match to get hom poly in Lie product maps
        T = taylor_to_lie(taylor, i, coords)

        # Lie maps product as array
        LieProduct.append(T)
        taylor = transform_taylor(
            T, taylor, i, coords, degree
        )  # adjust higher order taylor coeff for next coeff extraction

        if i > 5:
            print("Implemented only to 5th order so far.")
            break

    return LieProduct


def transform_taylor(ham, taylor, hom_order, coords, degree=3):  # adjust higher order coeffs
    # getaround for .subs() being iterative
    """
    Method to clean up the Taylor vector map.

    Adjust higher order coeffs, getaround for .subs() being iterative.

    Arguments:
    ----------
    ham
    taylor
    hom_order
    coords
    degree

    """
    log.setup()
    log.info("Cleaning up Taylor vector map.")
    # log.info("coords are {}".format(coords))

    sym_x1, sym_y1, sym_z1, sym_px1, sym_py1, sym_pz1 = symbols(
        "x_1 y_1 z_1 p_{x1} p_{y1} delta_1"
    )

    variables = tuple(coords)
    new_variables = (sym_x1, sym_px1, sym_y1, sym_py1, sym_z1, sym_pz1)

    if hom_order == 2:  # linear case needs more checking
        print(ham)
        R_inv = np.linalg.inv(ham)
        vec = [new_variables[i] for i in range(len(R_inv))]

        new_coords = np.dot(R_inv, vec)  # exp(-:G_2:) z_1 = z_0 + higher orders
        int_taylor = [
            polynomial.subs([(i, j) for i, j in zip(variables, new_coords)])
            for polynomial in taylor
        ]
        taylor = [
            polynomial.subs([(i, j) for i, j in zip(vec, variables)]) for polynomial in int_taylor
        ]
    else:  # higher order - note the order of the variables !!!!
        log.warning(
            "coords used - {}, phases used - {}".format(
                [variables[0], variables[2], variables[4]],
                [variables[1], variables[3], variables[5]],
            )
        )
        LiePoly = LieOperator(
            -ham,
            [variables[0], variables[2], variables[4]],
            [variables[1], variables[3], variables[5]],
        )  # use hom poly ham of degree = hom_order
        mod_taylor = taylorize(
            LiePoly, degree + 1
        )  # create symplectic jet to adjust the coeffs of taylor polynomial
        taylor = [
            old_poly - new_poly for old_poly, new_poly in zip(taylor, mod_taylor)
        ]  # exp(-:G_n:)z -> subtract from taylor poly the symplectic jet

    return taylor


def taylorize(LieHam, degree):
    """
    Apply Lie map to get taylor map vector on 6d vector
    """
    taylor_maps = []

    for i in tqdm(LieHam.indep_coords):
        fct = LieHam.LieMap(i, degree).doit()
        fct = truncate(fct, degree)

        taylor_maps.append(fct.expand())

    for i in tqdm(LieHam.indep_mom):
        fct = LieHam.LieMap(i, degree).doit()
        fct = truncate(fct, degree)

        taylor_maps.append(fct.expand())

    # reorder taylor maps to x, px, y, py, z, pz
    if len(LieHam.indep_coords) == 2:
        taylor_maps[1], taylor_maps[2] = taylor_maps[2], taylor_maps[1]  # swtich y and px

    elif len(LieHam.indep_coords) == 3:
        taylor_maps[1], taylor_maps[3] = taylor_maps[3], taylor_maps[1]  # swtich y and px
        taylor_maps[2], taylor_maps[3] = taylor_maps[3], taylor_maps[2]  # switch z and y
        taylor_maps[3], taylor_maps[4] = taylor_maps[4], taylor_maps[3]  # switch z and py

    return taylor_maps


def truncate(LieHam, degree):
    """
    cutoff Hamiltonian at specified degree
    """
    _epstemp = symbols("e")
    from sympy import N

    fct = N(LieHam.ham)

    for i in LieHam.indep_coords:
        fct = fct.subs(i, i * _epstemp)

    for i in LieHam.indep_mom:
        fct = fct.subs(i, i * _epstemp)

    fct = fct.expand() + O(_epstemp ** degree)
    fct = fct.removeO().subs(_epstemp, 1)

    return fct


def getKronPowers(state, order, dim_reduction=True):
    """
    Calculates Kroneker powers of state vector
    with dimension reduction

    e.g. for (x y) and order=2 returns:
    1, (x y), (x x*y y)

    Returns:
    list of numpy arrays, index corresponds to power
    """
    powers = [state]
    index = [np.ones(len(state), dtype=bool)]
    for _ in range(order - 1):
        state_i = np.kron(powers[-1], state)
        reduced, red_ind = cust_reduce(state_i)
        if dim_reduction:
            powers.append(reduced)
        else:
            powers.append(state_i)

        index.append(red_ind)

    powers.insert(0, np.array([1]))
    index.insert(0, [True])
    return powers, index


def cust_reduce(state):
    """
    Custom reduce funtion.
    """
    state_str = state.astype(str)
    reduced_state = []
    unique = []

    index = []
    for variable, variable_str in zip(state, state_str):
        if variable_str not in unique:
            unique.append(variable_str)
            index.append(True)
            reduced_state.append(variable)
        else:
            index.append(False)

    return np.array(reduced_state), index


def taylor_to_weight_mat(_taylor, coords):
    """
    Method to transform a Taylor map into weight matrices.

    ADD COORDS
    """
    log.setup()
    log.info("Extracting weight matrices from Taylor vector.")

    degree = 0

    for polynomial in _taylor:
        comp_degree = poly(polynomial).total_degree()  # highest degree of hom poly in Lie map
        if degree <= comp_degree:
            degree = comp_degree

    # if len(_taylor) == 2:
    #     coords = [sym_x, sym_px]
    # elif len(_taylor) == 4:
    #     coords = [sym_x, sym_px, sym_y, sym_py]
    # elif len(_taylor) == 6:
    #     coords = [sym_x, sym_px, sym_y, sym_py, sym_z, sym_pz]
    # else:
    #     raise TypeError("The dimension of the Taylor map vector does not match the phase space.")

    state_vectors, index = getKronPowers(coords, degree, dim_reduction=False)
    # print(state_vectors)

    W = []

    for i in range(degree + 1):
        if i == 0:
            #             print("Displacement not yet programmed.")
            continue

        w_sub = np.zeros((len(_taylor), len(state_vectors[i])))

        for j, taylor_row in enumerate(_taylor):
            taylor_sub = poly(taylor_row, state_vectors[1])

            for k, (state_vector, flag) in enumerate(zip(state_vectors[i], index[i])):
                coeff = taylor_sub.coeff_monomial(state_vector)

                if coeff != 0 and flag == True:
                    w_sub[j, k] = coeff
        W.append(w_sub.T)

    return W

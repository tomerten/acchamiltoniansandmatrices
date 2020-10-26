import numpy as np


class MissingArguemnt(ValueError):
    pass


def nestList(f, x, c, **kwargs):
    """
    Simple implementation of Mathematica's
    NestList command using generators.

    Arguments:
    ----------
    f: function
        function you want to apply iteritavely
    x: function argument
        argument of the function you want to apply the function to
    c: int
        number of iterations
    """
    for i in range(c):
        yield x
        x = f(x, **kwargs)
    yield np.array(x)


def LinMap(X, **kwargs):
    """
    Function to apply a matrix multiplication
    to an input vector, the matrix needs to be
    given in kwargs with key R.

    Arguments:
    ----------
    X : np.array
        input vector
    R (kwargs): np.array
        matrix for the multiplication
    """
    if not "R" in kwargs.keys():
        raise MissingArguemnt("kwargs is missing key R ")

    Rm = kwargs.get("R")
    return np.dot(Rm, X)


def GenerateNDimCoordinateGrid(N, NPOINTS, pmin=1e-6, pmax=1e-4, man_ranges=None):
    """
    Method to generate an N dimensional coordinate grid for tracking,
    with fixed number of point in each dimension.
    The final shape is printed at creation.

    IMPORTANT:
        Number of grid points scales with N * NPOINTS**N, i.e.
        very large arrays are generated already with
        quite some small numbers for NPOINTS and N.

        Example: NPOINTS = 2, N = 6 -> 6*2*6 = 384 elements

    Arguments:
    ----------
    N: int
        dimension of the coordinate grid
    NPOINTS: int
        number of points in each dimension
    pmin: float
        min coordinate value in each dim
    pmax: float
        max coordinate value in each dim

    """
    rangelist = [np.linspace(pmin, pmax, NPOINTS)] * N
    if man_ranges is not None:
        for k, v in man_ranges.items():
            rangelist[int(k)] = v
    grid = np.meshgrid(*rangelist)
    coordinate_grid = np.array([*grid])
    print(
        "Shape: {} - Number of paritcles: {} ".format(coordinate_grid.shape, coordinate_grid.size)
    )
    return coordinate_grid


def LinTrack(coord, nturns, R):
    """
    Simple linear tracking.

    Arguments:
    ----------
    coord: np.array
        coordinate grid array

    nturns: int

    """

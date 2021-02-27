import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def multi_countour_plot(
    hamlist,
    titlelist=[],
    xlabels=[],
    ylabels=[],
    nlevels=50,
    npoints=100,
    xrange=1,
    prange=0.99,
    size=(12, 12),
    d3=False,
):
    """
    Plotting method to plot a list of contours
    in two columns - main application is
    Hamiltonian contours.

    Arguments:
    ----------
    hamlist:    list
        list of functions to plot - depending on 2 variables
    titlelist:  list
        list of titles for the individual plots
    xlabels:    list
        list of xlabels to use
    ylabels:    list
        list of ylabels to use
    nlevels:    int
        number of contours to plot
    npoints:    int
        number of points to build meshgrid in each dim
    xrange:     int
        range of first variable to use
    prange:     int
        range of second variable to use
    size:       tuple(int,int)
        figsize tuple
    d3:         boolean
        switch between contour and 3D surface
    """
    # create meshgrid
    X = np.linspace(-xrange, xrange, npoints)
    P = np.linspace(-prange, prange, npoints)
    Xg, Pg = np.meshgrid(X, P)

    # evaluate the function on the grid
    Hlist = [ham(Xg, Pg) for ham in hamlist]

    # determine the number of rows needed for the plots
    nrows = len(hamlist) // 2

    if nrows == 0:
        nrows = 1

    # create the figure
    fig = plt.figure(constrained_layout=True, figsize=size)
    gs = fig.add_gridspec(nrows, 2)

    if d3:
        axes = [
            fig.add_subplot(gs[i, j], projection="3d")
            for j in range(2)
            for i in range(nrows - 1)
        ]
    else:
        axes = [fig.add_subplot(gs[i, j]) for j in range(2) for i in range(nrows - 1)]

    if len(hamlist) % 2 == 0:
        if d3:
            axes.append(fig.add_subplot(gs[nrows - 1, 0], projection="3d"))
            axes.append(fig.add_subplot(gs[nrows - 1, 1], projection="3d"))
        else:
            axes.append(fig.add_subplot(gs[nrows - 1, 0]))
            axes.append(fig.add_subplot(gs[nrows - 1, 1]))
    else:
        if d3:
            axes.append(fig.add_subplot(gs[nrows - 1, :], projection="3d"))
        else:
            axes.append(fig.add_subplot(gs[nrows - 1, :]))

    for i, (h, a) in enumerate(zip(Hlist, axes)):
        if d3:
            a.plot_surface(Xg, Pg, h, alpha=0.4, rstride=nlevels)
        a.contour(X, P, h, levels=nlevels, cmap=cm.jet_r)

        if len(titlelist) == len(hamlist):
            a.set_title(titlelist[i], fontsize=12)

        if len(xlabels) == len(hamlist):
            a.set_xlabel(xlabels[i], fontsize=12)

        if len(ylabels) == len(hamlist):
            a.set_ylabel(ylabels[i], fontsize=12)

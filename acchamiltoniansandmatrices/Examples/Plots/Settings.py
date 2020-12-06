_k1 = 0.5
_k2 = -0.5
_ld = 2.5
_lqf = 0.25
_lqd = 0.5

_r0 = [
    [10e-7, 0, 10e-7, 0],
    [10e-4, 0, 10e-4, 0],
    [0.1, 0, 0.1, 0],
    [0.25, 0, 0.25, 0],
    [0.5, 0, 0.5, 0],
    [1, 0, 1, 0],
    [2, 0, 2, 0],
]

# The exact tracking in the plots is referred
# to using the full drift space and keeping the quads linear,
# for truncated Lie maps and Runge-Kutta I left the quads linear
# (and symplectified them as before when tracking through the linear FODO)
# and the drifts terminate, so these weight matrices are symplectic by definition.

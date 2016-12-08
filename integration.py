import numpy as np

# from Numerical Recipes in C Second Edition pg. 717
# available here: http://www2.units.it/ipl/students_area/imm2/files/Numerical_Recipes.pdf
a1, a2, a3, a4, a5, a6 = 0., 1 / 5, 3 / 10, 3 / 5, 1., 7 / 8
b21 = 1 / 5
b31, b32 = 3 / 40, 9 / 40,
b41, b42, b43 = 3 / 10, -9 / 10, 6 / 5
b51, b52, b53, b54 = -11 / 54, 5 / 2, -70 / 27, 35 / 27
b61, b62, b63, b64, b65 = 1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096
c1, c2, c3, c4, c5, c6 = 37 / 378, 0., 250 / 621, 125 / 594, 0., 512 / 1771
c1s, c2s, c3s, c4s, c5s, c6s = 2825 / 27648, 0., 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4


def adaptive_step_control(f, x, y, h0, args=(), n=1e-13):
    """ adaptive step control for Cash-Karp Embedded Runga-Kutta method

    Arguments:
    f    :    derivative function, from .derivs
    x    :    mass extrema
    y    :    r, l, P, T values for arguments of .derivs
    h0   :    initial mass step size
    args :    args pass to .derivs
    n    :    tolerance parameter

    Returns:

    ystep:    + dr, dl, dP, dT vector
    h1   :    next mass step size
    """

    x = np.asarray(x)
    y = np.asarray(y)
    k = np.zeros((6, y.shape[0]), dtype=float)  # equation 16.2.4

    k[0] = h0 * f(x + a1 * h0, y, *args)
    k[1] = h0 * f(x + a2 * h0, y + b21 * k[0], *args)
    k[2] = h0 * f(x + a3 * h0, y + b31 * k[0] + b32 * k[1], *args)
    k[3] = h0 * f(x + a4 * h0, y + b41 * k[0] + b42 * k[1] + b43 * k[2], *args)
    k[4] = h0 * f(x + a5 * h0, y + b51 * k[0] + b52 * k[1] + b53 * k[2] + b54 * k[3], *args)
    k[5] = h0 * f(x + a6 * h0, y + b61 * k[0] + b62 * k[1] + b63 * k[2] + b64 * k[3] + b65 * k[4], *args)

    ystep = y + (c1 * k[0] + c2 * k[1] + c3 * k[2] + c4 * k[3] + c5 * k[4] + c6 * k[5])
    delta = (c1 - c1s) * k[0] + (c2 - c2s) * k[1] + (c3 - c3s) * k[2] + (c4 - c4s) * k[3] + (c5 - c5s) * k[4] + (c6 - c6s) * k[5]  # equation 16.2.6
    S = .9
    if np.max(np.abs(n * k[0] / delta)) >= 1.:
        ratio = np.max(np.abs(n * k[0] / delta)**(.2))  # equation 16.2.7 with equation 16.2.9
    else:
        ratio = np.max(np.abs(n * k[0] / delta)**(.25))
    if not np.isfinite(ratio):
        ratio = 1 / S
        print('bad step')
    h1 = S * h0 * ratio  # equation 16.2.7
    return ystep, h1


def integrate(f, x, y0, h0, args=(), n=1e-13, lim=1e6):
    """ integrate derivative along mass coordinates

    Arguments:

    f    :    derivative function, from .derivs
    x    :    mass extrema
    y0   :    initial conditions, load1 or load2
    h0   :    inital mass step size, negative for surface to fp
    args :    args that pass to .derivs
    n    :    tolerance parameter
    lim  :    max step

    Returns:

    y    :    vector evaluated at each xs
    xs   :    mass coordinates in g for adaptive step sizes
    """
    hc = h0; xc = x[0]; xn = xc; yc = y0
    xs, ys, hs = [], [], []
    i = 0
    while (xc - x[-1]) / (x[-1] - x[0]) <= 0 and i < lim:
        i += 1
        xn += hc
        yc, hc = adaptive_step_control(f, xc, yc, hc, args=args, n=n)

        xs.append(xc)
        ys.append(yc)
        hs.append(hc)
        xc = xn
    # one last step past fp
    yc, hc = adaptive_step_control(f, xc, yc, hc, args=args, n=n)

    xs.append(xc)
    ys.append(yc)
    hs.append(hc)

    return ys, xs

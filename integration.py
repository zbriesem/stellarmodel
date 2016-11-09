import numpy as np


a21, = 1. / 4.,
a31, a32, = 3. / 32., 9. / 32.,
a41, a42, a43, =  1932. / 2197., -7200. / 2197., 7296. / 2197.,
a51, a52, a53, a54, =    439. / 216., -8., 3680. / 513., -845. / 4104.,
a61, a62, a63, a64, a65, = -8. / 27., 2., - \
    3544. / 2565., 1859. / 4104., -11. / 40,
b1, b2, b3, b4, b5, b6 = 16. / 135., 0., 6656. / \
    12825., 28561. / 56430., -9. / 50, 2. / 55.
b1s, b2s, b3s, b4s, b5s, b6s = 25. / 216., 0., 1408. / 2565., 2197. / 4104., -1. / \
    5, 0.
c1, c2, c3, c4, c5, c6 = 0., 1. / 4., 3. / 8., 12. / 13., 1., 1. / 2.


def adaptive_step_control(f, x, y, h, args=(), tol=1e-8):

    x = np.asarray(x)
    y = np.asarray(y)
    assert(y.ndim == 1, "y must be one dimensional")
    g = lambda x, y: f(y, x, *args)
    k = np.empty((6, y.shape[0]))

    k[0] = h * g(x + c1 * h, y)
    k[1] = h * g(x + c2 * h, y + a21 * k[0])
    k[2] = h * g(x + c3 * h, y + a31 * k[0] + a32 * k[1])
    k[3] = h * g(x + c4 * h, y + a41 * k[0] + a42 * k[1] + a43 * k[2])
    k[4] = h * g(x + c5 * h, y + a51 * k[0] + a52 *
                 k[1] + a53 * k[2] + a54 * k[3])
    k[5] = h * g(x + c6 * h, y + a61 * k[0] + a62 * k[1] +
                 a63 * k[2] + a64 * k[3] + a65 * k[4])

    ystep = y + (b1s * k[0] + b2s * k[1] + b3s * k[2] +
                 b4s * k[3] + b5s * k[4] + b6s * k[5])
    err = (b1 * k[0] + b2 * k[1] + b3 * k[2] + b4 * k[3] + b5 * k[4] + b6 * k[5]) - \
        (b1s * k[0] + b2s * k[1] + b3s * k[2] +
         b4s * k[3] + b5s * k[4] + b6s * k[5])
    adj = tol / np.abs(np.min(err))
    if adj > 1e3:
        adj = 1e3
    if not np.isfinite(adj):
        adj = 1.0
    opts = np.abs(np.hstack((y / k[0], 1.1)))
    hn = 0.1 * h * adj
    return ystep, hn


def integrate(g, x, y0, h0, args=(), tol=1e-8, N=10000):
    """integrate

    Arguments:
    
    g    :    derivative function, from .derivs
    x    :    steps
    y0   :    initial conditions
    h0   :    step size
    args :    args that pass to .derivs
    tol  :    tolerance parameter
    N    :    max step

    Returns:

    y    :    interpolation function for integral evaluated at each step
    data :    adaptive step sizes
    """
    data = {}
    x = np.asarray(x)
    y0 = np.asarray(y0)
    nv = y0.shape[0]
    no = x.shape[0]
    y = np.empty((no, nv))
    hc = h0
    xc = x[0]
    xn = xc
    yc = y0
    yo = np.empty((N, nv))
    xo = np.empty((N))
    ho = np.empty((N))
    i = 0
    while (xc - x[-1]) / (x[-1] - x[0]) <= 0 and i < N:
        i += 1
        xn += hc
        yc, hc = adaptive_step_control(g, xc, yc, hc, args=args, tol=tol)
        xo[i] = xc
        yo[i] = yc
        ho[i] = hc
        xc = xn
        print(hc)

    yo = yo[:i]
    xo = xo[:i]
    ho = ho[:i]
    data['hu'] = ho

    y = interp1d(xo, yo, axis=0, fill_value=np.nan)(x)

    return y, data

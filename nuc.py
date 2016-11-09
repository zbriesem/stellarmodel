import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.ndimage import gaussian_filter1d


def eppeff(T, rho, X, Y):
    """ returns the specific energy generation due to pp-chain, with linear interpolation to correct for the transition between pp1 and pp3

    Arguments:

    T    :    Temperature in K
    rho  :    density in g/cm^2
    X    :    Hydrogen mass fraction
    Y    :    Helium mass fraciton

    Returns:

    epp  :    effective power per gram of material with pp-chain under temperature and density conditions
    """
    T9 = T / 1e9
    psi0 = psi(T, Y)
    f11 = 1  # weak screening
    g11 = 1 + 3.82 * T9 + 1.51 * T9**2 + .144 * T9**3 - .0114 * T9**4
    return 2.57e4 * psi0 * f11 * g11 * rho * X**2 / T9**(2 / 3) * np.exp(-3.381 / T9**(1 / 3))


def eCNOeff(T, rho, X, XCNO):
    """ returns the specific energy generation due to CNO-chain

    Arguments:

    T    :    Temperature in K
    rho  :    density in g/cm^2
    X    :    Hydrogen mass fraction
    XCNO :    Carbon, nitrogen, oxygen mass fraciton

    Returns:

    eCNO :    effective power per gram of material with CNO cycle under temperature and density conditions
    """
    T9 = T / 1e9
    g141 = (1 - 2.00 * T9 + 3.41 * T9**2 - 2.43 * T9**3)
    return 8.24e25 * g141 * rho * X * XCNO / T9**(2 / 3) * np.exp(-15.231 / T9**(1 / 3) - (T9 / .8)**2)


def psi(T, Y):
    """ linear interpolation to correct for the transition between pp1 and pp3, with minor cheat to make uniformly negative second derivative

    Arguments:

    T    :    Temperature in K
    Y    :    Helium mass fraciton

    Returns:

    psi  :    fudge factor accounting for the transition between pp1 and pp3 domination due to temperature
    """
    T = np.asarray(T)
    T7 = T / 1e7
    Yvals = [.1, .5, .9]
    T7vals = np.arange(.75, 3.5, .25)
    Psivals = np.array([[1., 1., 1.03, 1.05, 1.15, 1.25, 1.36, 1.4, 1.42, 1.43, 1.43], [1., 1.03, 1.1, 1.36, 1.67,1.78, 1.65, 1.58, 1.50, 1.44, 1.43], [1., 1.17, 1.63, 1.84, 1.94, 1.93, 1.8, 1.69, 1.58, 1.5, 1.43]])
    Ygrid, T7grid = np.meshgrid(Yvals, T7vals)

    f = LinearNDInterpolator(
        (Ygrid.flatten(), T7grid.flatten()), Psivals.T.flatten())
    g = gaussian_filter1d(f(Y, T7vals), 1)
    h = interp1d(T7vals, g)

    Ts = T7.clip(.5, 3.5)

    return np.round(h(Ts), 2)

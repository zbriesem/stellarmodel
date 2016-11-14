import numpy as np
from . import nuc, density

G = 6.673e-8  # Newton's gravitational constant
nabla_ad = .4  # Adiabatic temperature gradient
c = 2.99792458e10  # speed of light
sbc = 5.6704e-5  # stefan-boltzmann constant
a = (4.0 * sbc) / c  # radiation constant
mH = 1.67262e-24  # H+ mass (proton)
kb = 1.3807e-16  # boltzmann constant


def drdm(r, rho):
    """ mass derivative of radius

    Arguments:

    r    :    radius in cm
    rho  :    density in g/cm^2

    Returns:

    drdm :    mass derivative of radius at radius r
    """
    return 1. / (4. * np.pi * r**2 * rho)


def dPdm(r, m):
    """ mass derivative of pressure

    Arguments:

    r    :    radius in cm
    m    :    mass in g

    Returns:

    dPdm :    mass derivative of pressure
    """
    return -1 * G * m / (4 * np.pi * r**4)


def dldm(rho, T, X, Y, XCNO):
    """ mass derivative of luminosity

    Arguments:

    rho  :    density in g/cm^2
    T    :    temperature in K
    X    :    Hydrogen mass fraction
    Y    :    Helium mass fraction
    XCNO :    Carbon, nitrogen, oxygen mass fraction

    Returns:

    dldm :    mass derivative of luminosity
    """
    return nuc.eppeff(T, rho, X, Y) + nuc.eCNOeff(T, rho, X, XCNO)


def dTdm(r, m, T, lum, P, k):
    """ mass derivative of temperature

    Arguments:

    r    :    radius in cm
    m    :    mass in g
    T    :    temperature in K
    lum  :    luminosity in erg/s
    P    :    pressure in ddyne/cm^2
    k    :    opacity in cm^2/g

    Returns:

    dTdm :    mass derivative of temperature
    """
    grad = 3 * k * lum * P / (16 * np.pi * a * c * G * m * T**4)

    if grad >= nabla_ad:
        grad = nabla_ad

    return -G * m * T / (4. * np.pi * r**4 * P) * grad


def total_der(m, vals, *args):
    """ mass derivative of r, l, P, T. Packed for star.py

    Arguments:

    m    :    mass in g
    vals :    input [r, l, P, T]
    args :    first must be opacity

    Returns:

    list :    mass derivative of r, l, P, T
    """
    assert(len(args) >= 1, "didn't pass opacities!")
    ks = args[0]

    r, l, P, T = vals
    rho = density.density(P, T, ks.X, ks.Y, ks.Z)
    k = ks.Rosseland_mean_opacity(rho, T)

    return np.array([drdm(r, rho), dldm(rho, T, ks.X, ks.Y, ks.XCNO), dPdm(r, m), dTdm(r, m, T, l, P, k)])

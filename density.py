import numpy as np

G = 6.673e-8  # Newton's gravitational constant
nabla_ad = .4  # Adiabatic temperature gradient
c = 2.99792458e10  # speed of light
sbc = 5.6704e-5  # stefan-boltzmann constant
a = (4.0 * sbc) / c  # radiation constant
mH = 1.67262e-24  # H+ mass (proton)
kb = 1.3807e-16  # boltzmann constant


def mu(X, Y, Z, e=False):
    """mean molecular weight, or mean molecular weight per electron

    Arguments:
    X    :    Hydrogen mass fraction
    Y    :    Helium mass fraciton
    Z    :    Metal mass fraction
    e    :    True for mean molecular weight per electron

    Returns:

    mu   :    mean molecular weight (per electron if e True)
    """
    if e:
        return 2. / (1. + X)
    else:
        return 2. / (1. + 3. * X + Y / 2.)


def density(P, T, X, Y, Z):
    """density given ideal gas equation of state

    Arguments:

    P    :    pressure in dyne/cm^2
    T    :    temperature in K
    Z    :    Metal mass fraction
    Y    :    Helium mass fraciton
    Z    :    Metal mass fraction

    Returns:

    rho  :    density in g/cm^3
    """
    mu0 = mu(X, Y, Z)
    rho = (mu0 * mH) / (kb * T) * (P - a / 3 * T**4)
    return rho


def beta(P, T, X, Y, Z):
    """gas pressure fraction
    potentially unnecessary"""
    Prad = a / 3 * T**4
    beta = (P - Prad) / P
    return beta

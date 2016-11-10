import numpy as np
from . import nuc, density, opacity
G = 6.673e-8  # Newton's gravitational constant
nabla_ad = .4  # Adiabatic temperature gradient
c = 2.99792458e10  # speed of light
sbc = 5.6704e-5  # stefan-boltzmann constant
a = (4.0 * sbc) / c  # radiation constant
mH = 1.67262e-24  # H+ mass (proton)
kb = 1.3807e-16  # boltzmann constant


def center_vec(Pc, Tc, L, m, ks, coreconv=False):
    """ central values of radius, luminosity, pressure and temperature

    Arguments:

    Pc   :    central pressure in dyne/cm^2
    Tc   :    central temperature in K
    L    :    luminosity of star in erg/s
    m    :    arbitrarily small mass coordinate in g
    X    :    Hydrogen mass fraction
    Y    :    Helium mass fraciton
    Z    :    Metal mass fraction

    Returns:

    r    :    radius corresponding to mass coordinate in cm
    L    :    luminosity at mass corrdinate m of star in erg/s
    P    :    pressure at mass corrdinate m in dyne/cm^2
    T    :    temperature at mass corrdinate m of star in K
    """
    rhoc = density.density(Pc, Tc, ks.X, ks.Y, ks.Z)

    epsilon = nuc.eppeff(Tc, rhoc, ks.X, ks.Y) + nuc.eCNOeff(Tc, rhoc, ks.X, ks.XCNO)
    lum = lum_near_center(m, epsilon)

    kc = ks.Rosseland_mean_opacity(rhoc, Tc)
    P = pressure_near_center(m, Pc, rhoc)

    T = temp_near_center(m, Tc, Pc, rhoc, epsilon, kc=kc, coreconv=coreconv)
    r = radius_near_center(m, rhoc)

    return r, lum, P, T


def radius_near_center(m, rho):
    """ radius of certain mass coordinate near center

    Arguments:

    m    :    mass coordinate in g
    rho  :    density at mass coordinate m in g/cm^3

    Returns:

    r    :    radius coordinate in cm
    """
    return (3. * m / (4. * np.pi * rho))**(1 / 3)


def pressure_near_center(m, Pc, rho):
    """ pressure at certain mass coordinate near center

    Arguments:

    m    :    mass coordinate in g
    Pc   :    central pressure in dyne/cm^2
    rho  :    density at mass coordinate m in g/cm^3

    Returns:

    P    :    pressure at mass coordinate m in dyne/cm^2
    """
    P = -1. * (3. * G) / (8 * np.pi) * (4 * np.pi * rho / 3.)**(4 / 3) * m**(2 / 3)
    return Pc + P


def temp_near_center(m, Tc, Pc, rho, epsilon, kc=None, coreconv=False):
    """ temperature at certain mass coordinate near center

    Arguments:

    m    :    mass coordinate in g
    Tc   :    central temperature in K
    Pc   :    central pressure in dyne/cm^2
    rho  :    density at mass coordinate m in g/cm^3
    epsilon:  nuclear energy generation at central temperature in erg/g/cm^2
    kc   :    central opacity in cm^2/g
    conv :    True for core convection

    Returns:

    T    :    temperature at mass corrdinate m in K
    """
    if kc is None and not coreconv:
        print("Requires either convection or opacity at core")
        raise ValueError
    if coreconv:
        T1 = -1. * (np.pi / 6)**(1 / 3) * G * nabla_ad * rho**(4 / 3) / Pc * m**(2 / 3)
        T = np.exp(np.log(Tc) + T1)

    else:
        T1 = -1 / (2 * a * c) * (3 / (4 * np.pi))**(2 / 3) * kc * epsilon * rho**(4 / 3) * m**(2 / 3)
        T = (Tc**4 + T1)**(1 / 4)
    return T


def lum_near_center(m, epsilon):
    """ luminosity due to nuclear energy generation at certain mass coordinate near center

    Arguments:

    m    :    mass coordinate in g
    epsilon:  nuclear energy generation at central temperature in erg/g/cm^2

    Returns:

    L    :    luminosity at mass corrdinate m of star in erg/s
    """
    return m * epsilon




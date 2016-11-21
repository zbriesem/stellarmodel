import numpy as np
from . import density
from scipy.optimize import minimize_scalar
G = 6.673e-8  # Newton's gravitational constant
nabla_ad = .4  # Adiabatic temperature gradient
c = 2.99792458e10  # speed of light
sbc = 5.6704e-5  # stefan-boltzmann constant
a = (4.0 * sbc) / c  # radiation constant
mH = 1.67262e-24  # H+ mass (proton)
kb = 1.3807e-16  # boltzmann constant


def surface_vec(m, R, L, ks):
    """ surface values of radius, luminosity, pressure and temperature

    Arguments:

    m    :    total mass of star in g
    R    :    radius of star in cm
    L    :    luminosity of star in erg/s
    ks   :    opacity table in cm^2/g

    Returns:

    R    :    radius at surface of star in cm
    L    :    luminosity at surface of star in erg/s
    P    :    pressure at optical depth 2/3 in dyne/cm^2
    T    :    effective temperature of star in K
    """
    T = temp_eff(L, R)
    X, Y, Z = ks.X, ks.Y, ks.Z
    Plower = guess(1.1, T, X, Y, Z, step=2)
    Pupper = guess(1.1e8, T, X, Y, Z, step=.5)
    
    # taken from Numerical Recipes in C Second Edition. Chapter 10
    P = minimize_scalar(residual, method='bounded', bounds=[Plower, Pupper], args=(m, T, R, ks))  # faster
    return R, L, P.x, T


def pressure_surface(m, R, kb):
    """ surface values pressure
    equation 11.13 in Kippenhahn

    Arguments:

    m    :    total mass of star in g
    R    :    radius of star in cm
    kb   :    opacity at surface in cm^2 / g


    Returns:

    P    :    pressure at optical depth 2/3 in dyne/cm^2
    """
    return 2 * G * m / (3 * R**2 * kb)


def temp_eff(L, R):
    """ surface values temperature
    equation 11.14 in Kippenhahn

    Arguments:

    R    :    radius of star in cm
    L    :    luminosity of star in erg/s

    Returns:

    T    :    effective temperature of star in K
    """
    return (L / (4 * np.pi * R**2 * sbc))**(1 / 4)


def guess(Pguess, T, X, Y, Z, step=3):
    """ determines extrema of pressure represented on the opacity table

    Arguments:

    Pguess:   underestimation of lowest pressure or overestimation of highest pressure represented on opacity table in dyne/cm^2
    T    :    effective temperature of star in K
    X    :    Hydrogen mass fraction
    Y    :    Helium mass fraction
    Z    :    Metal mass fraction
    step :    multiplicative step for next guess of Pguess

    Returns:

    P    :    an extremum of the pressure on the opacity table at a given temperature
    """
    ontable = False
    while not ontable:
        rho = density.density(Pguess, T, X, Y, Z)
        T6 = T / 1e6
        log_R = np.log10(rho / T6**3)
        log_T = np.log10(T)
        if 3.75 <= log_T <= 8.7 and -8. <= log_R <= 1.:
            ontable = True
        else:
            Pguess *= step
    return Pguess


def residual(P, m, T, R, ks):
    """ residual of the calculated surface pressure and the bounds found by guess()

    Arguments:

    P    :    surface pressure from pressure_surface() in dyne/cm^2
    m    :    total mass of star in g
    T    :    effective temperature of star in K
    R    :    radius of star in cm
    ks   :    opacity table

    Returns:

    P    :    an extremum of the pressure on the opacity table at a given temperature
    """
    rho = density.density(P, T, ks.X, ks.Y, ks.Z)
    kappa_bar = ks.Rosseland_mean_opacity(rho, T)
    Ps = pressure_surface(m, R, kappa_bar)
    resid = np.abs(Ps - P)
    return resid

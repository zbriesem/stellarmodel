import numpy as np
from . import nuc, opacity

G = 6.673e-8  # Newton's gravitational constant
nabla_ad = .4  # Adiabatic temperature gradient
c = 2.99792458e10  # speed of light
sbc = 5.6704e-5  # stefan-boltzmann constant
a = (4.0 * sbc) / c  # radiation constant
mH = 1.67262e-24  # H+ mass (proton)
kb = 1.3807e-16  # boltzmann constant


def drdm(r, rho):
    """mass derivative of radius"""
    return 1. / (4. * np.pi * r**2 * rho)


def dPdm(r, m):
    """mass derivative of pressure"""
    return -1 * G * m / (4 * np.pi * r**4)


def dldm(rho, T, X, Y, Z):
    """mass derivative of luminosity"""
    return nuc.eppeff(T, rho, X, Y) + nuc.eCNOeff(T, rho, X, Z)


def dTdm(r, m, T, lum, P, rho, X, Y, Z):
    """mass derivative of temperature"""
    opacities = opacity.OpacityTable(X, Y, Z)
    k = opacities.Rosseland_mean_opacity(rho, T)
    grad = 3 * k * lum * P / (16 * np.pi * a * c * G * m * T**4)
    if grad > nabla_ad:
        grad = nabla_ad
    return -G * m * T / (4. * np.pi * r**4 * P) * grad

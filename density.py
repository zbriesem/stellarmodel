G = 6.673e-8  # Newton's gravitational constant
nabla_ad = .4  # Adiabatic temperature gradient
c = 2.99792458e10  # speed of light
sbc = 5.6704e-5  # stefan-boltzmann constant
a = (4.0 * sbc) / c  # radiation constant
mH = 1.67262e-24  # H+ mass (proton)
kb = 1.3807e-16  # boltzmann constant


def density(P, T, X, Y, Z):
    """density given ideal gas equation of state

    Arguments:

    P    :    pressure in dyne/cm^2
    T    :    temperature in K
    Z    :    Metal mass fraction
    Y    :    Helium mass fraction
    Z    :    Metal mass fraction

    Returns:

    rho  :    density in g/cm^3
    """
    mu = 2. / (1. + 3. * X + Y / 2.)
    rho = (mu * mH) / (kb * T) * (P - a / 3 * T**4)
    return rho


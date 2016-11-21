from . import opacity, derivs, load1, load2, integration
from scipy.interpolate import interp1d
import numpy as np

Rs = 6.96e10  # radius of sun in cm
Ls = 3.828e33  # luminosity of sun in erg/s
Ms = 1.9891e33  # mass of sun in g


class Star:

    def __init__(self, comp):
        """ Star object, initally only defined by composition

        Arguments:
        comp :   X, Y, Z mass fractions, tuple

        Instance variables:
        ks   :   opacity table defined by composition
        civec:   near central conditions r, l, Pc, Tc
        sivec:   surface conditions
        coutvecs: integrated values for r, l, P, T at mass coordinates cxs, from center to fitting point
        cxs  :   mass coordinates, adaptive step sizes, from center to fitting point
        cfp  :   r, l, p, T at fitting point integrated from the center
        soutvecs: integrated values for r, l, P, T at mass coordinates sxs, from surface to fitting point
        sxs  :   mass coordinates, adaptive step sizes, from surface to fitting point
        sfp  :   r, l, p, T at fitting point integrated form the surface

        Functions:
        set_mass: set mass in Ms and fitting point as fraction of M
        set_initial: set initial values of R in cm, L in erg/s, Pc in dynes/cm^2, Tc in K, M in Ms
        center: integrate outward from center to fitting point
        surface: integrate inward from surface to fitting point
        return_vec: returns initial conditions necessary to match at fitting point, use after Newton Raphson
        """
        self.X, self.Y, self.Z = comp
        self.ks = opacity.OpacityTable(self.X, self.Y, self.Z)
        self.XCNO = self.ks.XCNO

    def set_mass(self, M, fp=.5):
        """ set mass in Ms and fitting point as fraction of M, 0 < fp < M"""
        self.M = M * Ms
        self.fp = fp * self.M
        self.dm = self.M * 1e-10
        assert(self.fp < self.M, "Your fitting point is outside the star!")

    def set_initial(self, *args):
        """ set initial values of R in cm, L in erg/s, Pc in dynes/cm^2, Tc in K, M in Ms
        *args must be (R, L, Pc, Tc)
        """
        self.R = args[0]
        self.L = args[1]
        self.Pc = args[2]
        self.Tc = args[3]

    def center(self):
        """ center-out integration
        """
        self.civec = load1.center_vec(self.Pc, self.Tc, self.dm, self.ks)
        m = [self.dm, self.fp]

        self.coutvecs, self.cxs = integration.integrate(derivs.total_der, m, self.civec, self.dm, args=(self.ks,))
        self.cfp = interp1d(self.cxs, self.coutvecs, axis=0)(self.fp)

    def surface(self):
        """ surface-in integration
        """
        self.sivec = load2.surface_vec(self.M, self.R, self.L, self.ks)
        m = [self.M, self.fp]

        self.soutvecs, self.sxs = integration.integrate(derivs.total_der, m, self.sivec, -1. * self.dm, args=(self.ks,))

        self.sfp = interp1d(self.sxs, self.soutvecs, axis=0)(self.fp)

    def return_vec(self):
        """ returns initial conditions necessary to match at fitting point, use after Newton Raphson
        """
        return np.array([self.R, self.L, self.Pc, self.Tc])

    def profiles(self):
        self.xs = np.hstack((self.cxs, self.sxs[::-1]))
        self.outvecs = np.vstack((np.asarray(self.coutvecs), np.asarray(self.soutvecs)[::-1]))






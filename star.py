from . import opacity, derivs, load1, load2, integration, density
from scipy.interpolate import interp1d
import numpy as np

Rs = 6.96e10  # radius of sun in cm
Ls = 3.828e33  # luminosity of sun in erg/s
Ms = 1.9891e33  # mass of sun in g


class Star:

    def __init__(self, comp):
        """ Star object, initally only defined by composition

        Arguments:
        comp :   input X, Y, Z mass fractions, tuple
                 if not explicitly in opacity table, all compositions will be changed to nearest represented composition

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
        self.ks = opacity.OpacityTable(*comp)
        self.X, self.Y, self.Z = self.ks.X, self.ks.Y, self.ks.Z
        self.XCNO = self.ks.XCNO
        self.n = 1e-14

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

    def set_tol(self, n):
        """ set integration tolerance parameter
        """
        self.n = n

    def center(self):
        """ center-out integration
        """
        self.civec = load1.center_vec(self.Pc, self.Tc, self.dm, self.ks)
        m = [self.dm, self.fp]

        self.coutvecs, self.cxs = integration.integrate(derivs.total_der, m, self.civec, self.dm, args=(self.ks,), n=self.n)
        self.cfp = interp1d(self.cxs, self.coutvecs, axis=0)(self.fp)

    def surface(self):
        """ surface-in integration
        """
        self.sivec = load2.surface_vec(self.M, self.R, self.L, self.ks)
        m = [self.M, self.fp]

        self.soutvecs, self.sxs = integration.integrate(derivs.total_der, m, self.sivec, -1. * self.dm, args=(self.ks,), n=self.n)

        self.sfp = interp1d(self.sxs, self.soutvecs, axis=0)(self.fp)

    def return_vec(self):
        """ returns initial conditions necessary to match at fitting point, use after Newton Raphson
        """
        return np.array([self.R, self.L, self.Pc, self.Tc])

    def profiles(self):
        """ stages profiles in self.outvecs, as well as density, opacity and temperature gradient
        """
        self.xs = np.hstack((self.cxs[:-1], self.fp, self.sxs[::-1][1:]))
        self.outvecs = np.vstack((np.asarray(self.coutvecs)[:-1], (self.cfp + self.sfp) / 2, np.asarray(self.soutvecs)[::-1][1:]))
        self.den = density.density(self.outvecs[:, 2], self.outvecs[:, 3], self.X, self.Y, self.Z)

        ops = []
        for i in range(len(self.outvecs)):
            ops.append(self.ks.Rosseland_mean_opacity(self.den[i], self.outvecs[i, 3]))
        self.ops = np.asarray(ops)

        nablas = []
        for i in range(0, len(self.xs) - 1):
            one = np.log(self.outvecs[i + 1, 3]) - np.log(self.outvecs[i, 3])
            two = np.log(self.outvecs[i + 1, 2]) - np.log(self.outvecs[i, 2])
            nablas.append(one / two)
        nablas.append(np.nan)
        self.nabla = np.asarray(nablas)






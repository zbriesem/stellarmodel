from . import opacity, derivs, load1, load2, integration
from scipy.interpolate import interp1d

Rs = 6.96e10  # radius of sun in cm
Ls = 3.828e33  # luminosity of sun in erg/s
Ms = 1.9891e33  # mass of sun in g


class Star:

    def __init__(self, comp):
        self.X, self.Y, self.Z = comp
        self.ks = opacity.OpacityTable(self.X, self.Y, self.Z)
        self.XCNO = self.ks.XCNO

    def set_fp(self, m):
        """set fitting point to mass coordinate m in Ms, 0 < m < M"""
        self.fp = m * Ms

    def set_init(self, R, L, Pc, Tc, M):
        """set initial values of R in Rs, L in Ls, Pc in dynes/cm^2, Tc in K, M in Ms
        """
        self.R = R * Rs
        self.L = L * Ls
        self.Pc = Pc
        self.Tc = Tc
        self.M = M * Ms
        self.dm = self.M * 1e-10
        assert(self.fp < self.M, "Your fitting point is outside the star!")

    def center(self):
        """center-out integration
        """
        ivec = load1.center_vec(self.Pc, self.Tc, self.L, self.dm, self.ks)
        m = [self.dm, self.fp]

        self.coutvecs, self.cxs = integration.integrate(derivs.total_der, m, ivec, self.dm, args=(self.ks,))
        self.cfp = interp1d(self.cxs, self.coutvecs, axis=0)(self.fp)

    def surface(self):
        """surface-in integration
        """
        ivec = load2.surface_vec(self.M, self.R, self.L, self.ks)
        m = [self.M, self.fp]

        self.soutvecs, self.sxs = integration.integrate(derivs.total_der, m, ivec, -1. * self.dm, args=(self.ks,))

        self.sfp = interp1d(self.sxs, self.soutvecs, axis=0)(self.fp)

    def return_vec(self):
        """returns final vec, use after integration and matching fitting point
        """
        return [self.R, self.L, self.Pc, self.Tc]



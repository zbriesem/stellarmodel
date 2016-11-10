import numpy as np
from . import opacity, derivs, load1, load2, integration, density

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
        self.dm = M * 1e-10
        assert(self.fp < self.M, "Your fitting point is outside the star!")

    def center(self):
        ivec = load1.center_vec(self.Pc, self.Tc, self.L, self.m, self.X, self.Y, self.Z)
        m = [self.dm, self.fp]

        self.coutvecs, self.chs = integration.integrate(derivs.total_der, m, ivec, self.dm, self.X, self.Y, self.Z)



    def surface(self):
        ivec = load2.surface_vec(self.M, self.R, self.L, self.X, self.Y, self.Z)
        m = [self.M, self.fp]

        self.soutvecs, self.hs = integration.integrate(derivs.total_der, m, ivec, self.dm, self.X, self.Y, self.Z)



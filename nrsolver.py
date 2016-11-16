import numpy as np


class NewtonRaphson:

    def __init__(self, Star):
        self.Star = Star  # must already initialize mass

    def set_init(self, *args):
        R, L, Pc, Tc = args
        self.Star.set_initial(R, L, Pc, Tc)

    def discrepancy_vec(self):
        self.Star.center()
        self.Star.surface()

        return np.asarray(self.Star.sfp - self.Star.cfp)

    def jacobian(self, y0, F0, step=1e-10):
        dy0 = step * y0

        J = np.zeros((4, 4))
        F = np.zeros((4, 4))
        for ii in range(4):
            y = y0
            y[ii] += dy0[ii]
            self.set_init(*y)
            F[ii] = self.discrepancy_vec()
            F1 = F[ii]

            J[ii] = (F1 - F0) / dy0[ii]  # this is Jji

        if any(np.abs(F[-1]) > np.abs(F0)):
            print('attempting other direction')
            return self.jacobian(y0, F0, step=-.1 * step)
        else:
            return J.T

    def solve(self, *args):
        self.set_init(*args)
        y0 = np.asarray(args)
        F0 = self.discrepancy_vec()

        while any(F0 > 1e-10 * y0):
            J = self.jacobian(y0, F0)
            jinv = np.linalg.inv(J)
            delV = -np.dot(jinv, F0)
            print(delV)
            y0 = y0 + delV
            self.set_init(*y0)
            F0 = self.discrepancy_vec()

        self.set_init(*y0)

import numpy as np


class NewtonRaphson:

    def __init__(self, Star):
        self.Star = Star  # must already initialize mass

    def set_init(self, *args):
        self.Star.set_initial(*args)

    def discrepancy_vec(self):
        self.Star.center()
        self.Star.surface()
        return np.asarray(self.Star.sfp - self.Star.cfp)

    def jacobian(self, y0, F0, step=1e-6):
        dy0 = step * y0

        J = np.zeros((4, 4))
        for ii in range(4):
            y = y0.copy()
            y[ii] += dy0[ii]
            self.set_init(*y)
            F1 = self.discrepancy_vec()

            J[:, ii] = (F1 - F0) / dy0[ii]

        return J

    def solve(self, *args):
        self.set_init(*args)
        y0 = np.asarray(args)
        F0 = self.discrepancy_vec()

        while any(F0 > 1e-2 * y0):
            J = self.jacobian(y0, F0)
            jinv = np.linalg.inv(J)
            delV = -np.dot(jinv, F0)
            y0 = y0 + delV
            self.set_init(*y0)
            F0 = self.discrepancy_vec()

        self.set_init(*y0)

import numpy as np


class NewtonRaphson:

    def __init__(self, Star):
        """ finds Newton Raphson solution for shooting to a fitting point method

        Arguments:
        Star    :    Star object, initialized with composition and Star.set_mass(M, fp)

        Functions:
        set_init:   passes initial r, l, Pc, Tc vector to Star object
        discrepancy_vec:    runs Runge-Kutta from center to fitting point and surface to fitting point
                            Returns difference of r, l, Pc, Tc at fitting point
        jacobian:   estimates the Jacobian matrix of the discrepancy vector due to small perturbation
                    Returns Jacobian matrix
        solve:      attempts to find Newton Raphson solution. Output in Star object
        """
        self.Star = Star
        if self.Star.M is None:
            print('You must run NewtonRaphson.Star.set_mass(M, fp) to assign a mass and fitting point!')

    def set_init(self, *args):
        """passes initial R, L, Pc, Tc vector to Star object
        """
        self.Star.set_initial(*args)

    def discrepancy_vec(self):
        """ runs Runge-Kutta from center to fitting point and surface to fitting point
        Returns difference of r, l, P, T at fitting point
        """
        self.Star.center()
        self.Star.surface()
        return np.asarray(self.Star.sfp - self.Star.cfp)

    def jacobian(self, y0, F0, step=1e-2):
        """ estimates the Jacobian matrix of the discrepancy vector due to small perturbation

        Argument:
        y0   :   vector of initial conditions R, L, Pc, Tc, array
        F0   :   discrepancy vector r, l, P, T at fitting point, array
        step :   scale of perturbation of inital conditions y0

        Returns:
        J    :   Jacobian matrix
        """
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
        """ attempts to find Newton Raphson solution. Output in Star object

        Arguments:
        args    :   R in cm, L in erg/s, Pc in dyne/cm^2, Tc in K

        Returns:
        Star    :   Star object, updated to convergent initial values
                    Run NewtonRaphson.Star.return_vec() for these values
                    """
        print('Initializing Star object')
        self.set_init(*args)
        y0 = np.asarray(args)
        F0 = self.discrepancy_vec()
        loop = 1
        while not all(np.abs(F0) < 1e-4 * y0):  # make convergence condition robust
            print('Beginning Newton Raphson Iteration', loop)
            J = self.jacobian(y0, F0)
            jinv = np.linalg.inv(J)
            delV = -np.dot(jinv, F0)
            y0 = y0 + delV
            self.set_init(*y0)
            F0 = self.discrepancy_vec()
            loop += 1
        print('After', loop - 1, 'iterations of Newton Raphson, the difference vector at the fitting point is', F0)


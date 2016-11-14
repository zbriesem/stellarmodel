import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from .setup import home_dir


class OpacityTable:

    def __init__(self, X, Y, Z, fname=home_dir() + '/data/OP17.zbriesem@ucsc.edu04112016054844.tab'):
        """finds the closest composition match to input in OP data file and does a linear interpolation over the opacity table

        Arguments:

        X    :    Hydrogen mass fraction
        Y    :    Helium mass fraciton
        Z    :    Metal mass fraction
        fname:    location of an OP data file

        Instance variables:

        CNOfrac :    Carbon, Nitrogen, Oxygen mass fraciton
        tables:   all compositions contained in data file
        ID   :    the table ID in data file which the nearest routine returned to best fit input composition
        log_R:    log10(rho / T6**3)
        log_T:    log10(T)
        log_k:    log10(k), where k are Rosseland mean opacities in cm^2/g

        Functions:

        Rosseland_mean_opacity(rho, T)    :    returns interpolated value for any rho, T on opacity table
        """
        with open(fname, 'r') as file:
            self.lines = file.readlines()
        self.X = X
        self.Y = Y
        self.Z = Z
        assert(self.X + self.Y + self.Z == 1, "Mass fractions don't add to unity!")
        self.get_data()
        self.XCNO = self.Z * sum([float(self.lines[ii].strip().split()[3]) for ii in [36, 37, 38]])

    def read_summary(self, begin=62, end=188):
        """read all compositions available in data file"""
        summary = [self.lines[begin:end][i].strip().split()
                   for i in range(end - begin)]
        Xs = [np.float(summary[i][4].split('=')[-1])
              for i in range(end - begin)]
        Ys = [np.float(summary[i][5].split('=')[-1])
              for i in range(end - begin)]
        Zs = [np.float(summary[i][6].split('=')[-1])
              for i in range(end - begin)]
        self.tables = list(zip(Xs, Ys, Zs))

    def match_composition(self):
        """finds nearest composition to input"""
        self.read_summary()
        comps = np.asarray(self.tables)
        interpolator = NearestNDInterpolator(comps, range(len(comps)))
        self.ID = interpolator((self.X, self.Y, self.Z))

    def get_data(self):
        """extracts the table associated with composition"""
        self.match_composition()
        self.log_R = np.asarray(
            self.lines[self.ID * 77 + 244].strip().split()[1:], dtype=float)
        self.log_k = np.asarray([self.lines[246 + self.ID * 77:316 + self.ID * 77][
                                i].strip().split()[1:] for i in range(70)], dtype=float)
        self.log_T = np.asarray([self.lines[246 + self.ID * 77:316 + self.ID * 77][
                                i].strip().split()[0] for i in range(70)], dtype=float)
        self.log_k[self.log_k == 9.999] = np.nan

    def Rosseland_mean_opacity(self, rho, T):
        """linear interpolation over the values of the Rosseland mean opacities at the initialized composition

        Arguments:

        rho  :    density in g/cm^3
        T    :    temperature in K

        Returns:

        k    :    Rosseland mean opacity in cm^2/g"""
        T6 = T / 1e6
        R = rho / T6**3
        k = 10**self.log_k

        grid_R, grid_T = np.meshgrid(10**self.log_R, 10**self.log_T)
        interpolator = LinearNDInterpolator((grid_R.flatten(), grid_T.flatten()), k.flatten())

        return interpolator(R, T)

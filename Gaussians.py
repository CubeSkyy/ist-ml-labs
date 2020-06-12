import numpy as np
import itertools

class Gaussian1D:
    def mean(self, x):
        return np.mean(x)

    def std(self, x, verbose=True):
        std = np.std(x, ddof=1)
        mean = self.mean(x)
        if verbose:
            print("Std: (1/(%d - 1)) * sqrt(" % (x.shape[0]), end="")
        sum = 0
        for i in range(0, x.shape[0]):
            sum += (x[i] - mean)**2
            if verbose:
                print("(%.2f %s %.2f)^2"
                      % (x[i], '-' if mean >= 0 else '+', abs(mean))
                      , end=" + " if i != x.shape[0] - 1 else ") = " + str(np.sqrt(sum/ (x.shape[-1] - 1))) + "\n")

        return std

    def Normal1D(self, x):
        mean = self.mean(x)
        std = np.std(x, ddof=1)
        print("N(x | u, std) = 1/(%f * sqrt(2pi) * exp(-1/2 * (x - %f / %f)^2)"
              % (std, mean, std))

class Gaussian2D:
    def mean(self, x):
        return np.mean(x, axis=1)

    def covariance(self, x, verbose=True):
        cov = np.cov(x)
        mean = self.mean(x)
        lst = list(itertools.product([0, 1], repeat=cov.shape[0]))
        for e in lst:
            if verbose:
                print("Cov%s: (1/(%d - 1)) * " % (e, x.shape[-1]), end="")
            sum = 0
            for i in range(0, x.shape[-1]):
                sum += (x[e[0]][i] - mean[e[0]]) * (x[e[1]][i] - mean[e[1]])
                if verbose:
                    print("(%.2f %s %.2f) (%.2f %s %.2f)"
                          % (x[e[0]][i], '-' if mean[e[0]] >= 0 else '+', abs(mean[e[0]]),
                             x[e[1]][i], '-' if mean[e[1]] >= 0 else '+', abs(mean[e[1]]))
                          , end=" + " if i != x.shape[-1] - 1 else " = " + str(sum / (x.shape[-1] - 1)) + "\n")
        if verbose:
            print("\nCov. Matrix:\n", cov)
        return cov

    def determinant(self, x, verbose=True):
        det = np.linalg.det(x)
        if verbose:
            print("Determinant:", det)
        return det

    def inverse(self, x, verbose=True):
        inv = np.linalg.inv(x)
        if verbose:
            print("Inverse:\n", inv)
        return inv

    def Normal2D(self, x, verbose = False):
        mean = np.array2string(self.mean(x))
        cov = self.covariance(x, verbose=verbose)
        covStr = np.array2string(cov)
        detCov = self.determinant(cov, verbose=verbose)
        invCov = np.around(self.inverse(cov, verbose=verbose), 4).tolist()
        print("N(x | u, cov) = 1/(2pi * sqrt(%.4f) * exp(-1/2 * ([x0, x1] - %s)^T * %s * ([x0, x1] - %s))"
              % (detCov, mean, invCov, mean))
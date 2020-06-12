import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, x, w, t, rate):
        self.x = x
        self.w = w
        self.t = t
        self.rate = rate
        self.solved = False
        self.activation = self.sign

    def step(self, x):
        return 1 if x >= 0 else 0

    def sign(self, x):
        return 1 if x >= 0 else -1

    def output(self, e, i="", verbose=True):
        out = self.activation(np.dot(e, self.w))
        if verbose:
            print("o%s = sign(w . x) = sign(%s . %s) = %s" % (
                i, np.array2string(self.w), np.array2string(e), out))
        return out

    def solve(self, verbose=True):
        x = np.append(np.ones((self.x.shape[0], 1)), self.x, axis=1)
        converged = False
        z = 0
        while not converged:
            converged = True
            if (verbose):
                print("-" * 100)
                print("Iteration %d:" % (z))
            for i in range(x.shape[0]):
                if (verbose):
                    print("-" * 30)
                    print("Point %d: %s" % (i, x[i]))
                out = self.output(x[i], i=str(i), verbose=verbose)
                if (verbose):
                    print("Output:", out, "Target:", self.t[i])
                if (out != self.t[i]):
                    converged = False
                    oldw = self.w
                    self.w = self.w + self.rate * (self.t[i] - out) * x[i]
                    if verbose:
                        print(
                            "Mistake was made.\nwi = wi + rate * (t - o) * xi  <=>\nwi = %s + %d * (%d - %d) * %s = %s"
                            % (np.array2string(oldw), self.rate, self.t[i], out, self.x[i], np.array2string(self.w)))
            z += 1
        if verbose:
            print("-" * 100)
            print("\nAlgorithm converged after %d steps.\n" % (z))
        self.solved = True

    def getSeparationLine(self):
        print("w0 +", end=" ")
        for i in range(1, self.w.shape[0]):
            print("w%d x%d %s" % (i, i, "+" if i != self.w.shape[0] - 1 else ""), end=" ")
        print("= 0 <=>")

        print("%s +" % (self.w[0]), end=" ")
        for i in range(1, self.w.shape[0]):
            print("%s x%d %s" % (self.w[i], i, "+" if i != self.w.shape[0] - 1 else ""), end=" ")
        print("= 0")

    def drawPlot(self):
        print("Separation line:", end=" ")
        self.getSeparationLine()
        x_ = np.arange(0.0, 1.1, 0.1)
        y = (x_ * -self.w[1] - self.w[0]) / (self.w[2])
        plt.plot(x_, y)

        for i in range(0, self.x.shape[0]):
            plt.plot(self.x[i][0], self.x[i][1], marker='o', color='r' if self.t[i] == 1 else 'g', ls='None')

        plt.grid(alpha=.4, linestyle='--')
        plt.show()

    def queryPoint(self, p):
        if not self.solved:
            self.solve(verbose=False)
        print("Query point: %s" % (p))
        out = self.output(np.insert(p, 0, 1, axis=0))
        print("Output: %s" % (out))

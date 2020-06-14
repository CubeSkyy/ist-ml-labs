import numpy as np
from Gaussians import Gaussian1D
from Gaussians import Gaussian2D
from KMeans import _KMeans
from NaiveBayes import NaiveBayes

# ------------------------------------------------------------------------
# Lab 1 - Perceptron
# ------------------------------------------------------------------------
# 1.
# 1. a) Solve Perceptron

# x = np.asarray([[0, 0], [0, 2], [1, 1], [1, -1]])
# w = np.asarray([1, 1, 1])
# t = np.asarray([-1, 1, 1, -1])
# rate = 1
# perceptron = Perceptron(x, w, t, rate)
# perceptron.solve()

# 1. b) Calculate and Draw Separation Line

# x = np.asarray([[0, 0], [0, 2], [1, 1], [1, -1]])
# w = np.asarray([1, 1, 1])
# t = np.asarray([-1, 1, 1, -1])
# rate = 1
# perceptron = Perceptron(x, w, t, rate)
# perceptron.solve(verbose=False)
# perceptron.drawPlot()


# 2. c) Query a point

# x = np.asarray([[0, 0,0], [0, 2,1], [1, 1,1], [1, -1,0]])
# w = np.asarray([1, 1, 1,1])
# t = np.asarray([-1, 1, 1, -1])
# rate = 1
# perceptron = Perceptron(x, w, t, rate)
# perceptron.queryPoint(np.asarray([0, 0, 1]))

# 2. c) Step Activation Function

# x = np.asarray([[0, 0,0], [0, 2,1], [1, 1,1], [1, -1,0]])
# w = np.asarray([1, 1, 1,1])
# t = np.asarray([0, 1, 1, 0])
# rate = 1
# perceptron = Perceptron(x, w, t, rate)
# perceptron.activation = perceptron.step
# perceptron.solve()

# ------------------------------------------------------------------------
# Lab 1 - Decision Trees - ID3
# ------------------------------------------------------------------------

# 1. a)

# f1 = 'a,c,c,b,a,b'.split(',')
# f2 = 'a,b,a,a,b,b'.split(',')
# f3 = 'a,c,c,a,c,c'.split(',')
# output = '+,+,+,-,-,-'.split(',')
#
# dataset = {'f1': f1, 'f2': f2, 'f3': f3, 'output': output}
# df = pd.DataFrame(dataset, columns=['f1', 'f2', 'f3', 'output'])
# id3 = ID3()
#
# tree = id3.buildTree(df)
# id3.printTree(tree)

# 2. a)

# f1 = 'd,c,c,d,c,c'.split(',')
# f2 = 'a,a,a,a,b,b'.split(',')
# f3 = 'b,b,a,a,a,b'.split(',')
# output = 'm,n,y,y,f,f'.split(',')
#
# dataset = {'f1': f1, 'f2': f2, 'f3': f3, 'output': output}
# df = pd.DataFrame(dataset, columns=['f1', 'f2', 'f3', 'output'])
# id3 = ID3()
#
# tree = id3.buildTree(df)
# id3.printTree(tree)

# ------------------------------------------------------------------------
# Lab 2- Gaussians
# ------------------------------------------------------------------------
gaussian1D = Gaussian1D()
gaussian2D = Gaussian2D()

# 2.

# data = np.array([180, 160, 200, 171, 159, 150])
# gaussian1D.Normal1D(data)

# 3.

# data = np.array([[-2, -1, 0, -2], [2, 3, 1, 1]])
# gaussian2D.Normal2D(data)

# 4.

# data = np.array([[2, 1, 0, 2], [-2, 3, -1, 1]])
# gaussian2D.Normal2D(data)


# ------------------------------------------------------------------------
# Lab 2- Naive Bayes
# ------------------------------------------------------------------------

# 2. a) b)
NB = NaiveBayes()

#
# x1 = '1,1,1,1,0,1'.split(',')
# x2 = '1,0,0,1,0,0'.split(',')
# x3 = '0,0,0,1,1,0'.split(',')
# x4 = '1,1,1,0,1,0'.split(',')
# x5 = '0,1,1,1,1,0'.split(',')
# output = 'a,a,a,b,b,c'.split(',')
#
# dataset = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'output': output}
# df = pd.DataFrame(dataset, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'output'])
#
# query = pd.Series({'x1': 1, 'x2': 0, 'x3': 1, 'x4': 0, 'x5': 1, })
# res =NB.query(df, query)


# 2. c)

# query = pd.Series({'x1': 1,  'x3': 1, 'x5': 1, })
# res = NB.query(df, query)


# ------------------------------------------------------------------------
# Lab 5 - KMeans
# ------------------------------------------------------------------------

# 1
# x = np.asarray([[0, 0], [1, 0], [0, 2], [2, 2]])
# c = np.asarray([[2, 0], [2, 1]])
# kmeans = _KMeans()
# kmeans.runKMeans(x, c)

# 2.a
# x = np.asarray([[1, 0, 0], [8, 8, 4], [3, 3, 0], [0, 0, 1], [0, 1, 0], [3, 2, 1]])
# c = np.asarray([x[0], x[1]])
# kmeans = _KMeans()
# kmeans.runKMeans(x, c)

# 2.b
# x = np.asarray([[1, 0, 0], [8, 8, 4], [3, 3, 0], [0, 0, 1], [0, 1, 0], [3, 2, 1]])
# c = np.asarray([x[0], x[1], x[2]])
# kmeans = _KMeans()
# kmeans.runKMeans(x, c, k=3)

# ------------------------------------------------------------------------
# Lab 8 - PCA
# ------------------------------------------------------------------------
#
# g2D = Gaussian2D()
#
# data = np.array([[0, 0], [4, 0], [2, 1], [6, 3]]).T
# g2D.mean(data)
# cov = g2D.covariance(data)
# w, v = np.linalg.eig(cov)
#
# print("Eigenvalues: %s \n" % w)
# print("Eigenvectors: %s \n" % v.T)
# print("KT-Transform: %s \n" % v)
#
# # Transform dataset with KT Transform
# for e in data.T:
#     print("KT * X = %s * %s = %s" % (v.T, e, np.dot(v.T,e)))




# ------------------------------------------------------------------------
# Other Stuff
# ------------------------------------------------------------------------

def mpInverse(x, verbose=True):
    res = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
    if verbose:
        print("\n(X.T * X)^-1  * X.T = ")
        print("\n(%s * %s)^-1 * %s = " % (x.T, x, x.T))
        print("\n(%s)^-1 * %s = " % (np.dot(x.T, x), x.T))
        print("\n%s * %s = " % (np.linalg.inv(np.dot(x.T, x)), x.T))
        print("\n%s" % (res))

    return res


def sigmoid(x):
    return 1 / (1 + np.exp(-float(x)))


def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def weightDelta(x, y, w, rate=1, output=sigmoid):
    return rate * x * (y - output(np.dot(w, x)))


# ------------------------------------------------------------------------
# RBF Network
# ------------------------------------------------------------------------

# initc = np.array([[0, 0], [-1, -0]])
# max_epochs_kmeans = 2
# rate = 1
# b = 1
# w = np.array([1, 1])
# x = np.array([[0, 0], [0, -1], [-1, 0], [-1, -1]], dtype=float)
# sigma = 1
# target = np.array([1, 0, 0, 1])
# max_epochs = 3
#
# rbfNetwork = RBFNetwork(initc, rate, b, w, x, sigma, target, max_epochs, max_epochs_kmeans)
# rbfNetwork.train()
# rbfNetwork.query(np.array([0, 0]))
# rbfNetwork.query(np.array([-1, 0]))
# rbfNetwork.query(np.array([0, -1]))
# rbfNetwork.query(np.array([-1, -1]))

# ------------------------------------------------------------------------
# Linear regression SSE closed form
# ------------------------------------------------------------------------

# x = np.asarray([[1, 1], [2, 1], [1, 3], [3, 3]])
# x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
# y = np.asarray([[1.4, 0.5, 2.0, 2.5]])
#
# print(np.dot(mpInverse(x, verbose=False), y.T))

# ------------------------------------------------------------------------
# Logistic Regression gradient descent
# ------------------------------------------------------------------------

# x = np.asarray([[1, 1], [2, 1], [1, 3], [3, 3]])
# x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
# y = np.asarray([1, 1, 0, 0])
# w = np.asarray([1, 1, 1]).astype('float')
# rate = 1

# ------------------------
# Normal GD
# ------------------------
# sum = w.copy()
# for i, e in enumerate(x):
#     sum += weightDelta(x[i], y.T[i], w, rate=1, output=sigmoid)
#
# print(sum)

# ------------------------
# Stochastic GD
# ------------------------
# print(w + weightDelta(x[0], y.T[0], w, rate=1, output=sigmoid))

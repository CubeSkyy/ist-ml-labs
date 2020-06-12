import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import log2 as log
import pprint
from Perceptron import Perceptron
from ID3 import ID3

def sigmoid(x):
    return 1 / (1 + np.exp(-float(x)))

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

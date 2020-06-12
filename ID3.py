import numpy as np
import pprint
from numpy import log2 as log

class ID3:
    def __init__(self):
        self.eps = np.finfo(float).eps

    def find_entropy(self, df):
        Class = df.keys()[-1]  # To make the code generic, changing target variable class name
        entropy = 0
        values = df[Class].unique()
        for value in values:
            fraction = df[Class].value_counts()[value] / len(df[Class])
            entropy += -fraction * np.log2(fraction)
        return entropy

    def find_entropy_attribute(self, df, attribute):
        Class = df.keys()[-1]  # To make the code generic, changing target variable class name
        target_variables = df[Class].unique()  # This gives all 'Yes' and 'No'
        variables = df[
            attribute].unique()  # This gives different features in that attribute (like 'Hot','Cold' in Temperature)
        entropy2 = 0
        print("-"*50)
        print("Feature:", attribute)
        for variable in variables:
            print("\tVariable:", variable)
            entropy = 0
            i = 0
            res = "E("
            for target_variable in target_variables:
                i += 1
                num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
                den = len(df[attribute][df[attribute] == variable])
                fraction = num / (den + self.eps)
                entropy += -fraction * log(fraction + self.eps)
                res += str(fraction) + (', ' if i != len(target_variables) else ')')
            fraction2 = den / len(df)
            entropy2 += -fraction2 * entropy
            print("\t\t%.3f * %s = %.3f * %.3f = %.3f" % (fraction2, res, fraction2, round(entropy, 3), round(entropy * fraction2, 3)))
        print("Sum for %s: %f" % (attribute, abs(entropy2)))
        initial = self.find_entropy(df)
        print("Information gain: Initial Entropy - Feature Entropy = %f - %f = %f" % (initial, abs(entropy2), initial - abs(entropy2) ))
        return abs(entropy2)

    def find_winner(self, df):
        Entropy_att = []
        IG = []
        for key in df.keys()[:-1]:
            #         Entropy_att.append(find_entropy_attribute(df,key))
            IG.append(self.find_entropy(df) - self.find_entropy_attribute(df, key))
        print("-" * 100)
        print("Feature Chosen:", df.keys()[:-1][np.argmax(IG)])
        print("-" * 100)
        return df.keys()[:-1][np.argmax(IG)]

    def get_subtable(self, df, node, value):
        return df[df[node] == value].reset_index(drop=True)

    def buildTree(self, df, tree=None):
        Class = df.keys()[-1]  # To make the code generic, changing target variable class name

        # Here we build our decision tree

        # Get attribute with maximum information gain
        node = self.find_winner(df)

        # Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
        attValue = np.unique(df[node])

        # Create an empty dictionary to create tree
        if tree is None:
            tree = {}
            tree[node] = {}

        # We make loop to construct a tree by calling this function recursively.
        # In this we check if the subset is pure and stops if it is pure.

        for value in attValue:

            subtable = self.get_subtable(df, node, value)
            clValue, counts = np.unique(subtable['output'], return_counts=True)

            if len(counts) == 1:  # Checking purity of subset
                tree[node][value] = clValue[0]
            else:
                tree[node][value] = self.buildTree(subtable)  # Calling the function recursively
        return tree

    def printTree(self, tree):
        print("Final Tree:")
        pprint.pprint(tree, width=1)

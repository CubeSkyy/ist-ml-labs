import pandas as pd
from sklearn import preprocessing


class NaiveBayes:

    def __init__(self):
        self.priorsDone = False
        self.posteriorsDone = False

    def calculatePriors(self, df):
        if (self.priorsDone):
            return self.priors
        self.priorsDone = True
        self.priors = df['output'].value_counts() / len(df['output'])
        return self.priors

    def calculatePosteriors(self, df):
        if (self.posteriorsDone):
            return self.likelihood
        self.posteriorsDone = True
        self.likelihood = pd.DataFrame(columns=['feature', 'featureValue', 'class', 'prob'])
        for i in range(0, 2):
            for e in df['output'].unique():
                for (columnName, columnData) in df.iteritems():
                    if columnName != 'output':
                        prob = columnData.iloc[df.index[df['output'] == str(e)].tolist()].eq(str(i)).astype(
                            int).sum() / len(
                            columnData.iloc[df.index[df['output'] == str(e)].tolist()])
                        self.likelihood = self.likelihood.append(
                            {'feature': columnName, 'featureValue': i, 'class': e, 'prob': prob},
                            ignore_index=True)
        return self.likelihood

    def query(self, df, query, verbose=True):
        if not (self.posteriorsDone):
            self.calculatePosteriors(df)
        if not (self.priorsDone):
            self.calculatePriors(df)

        res = pd.DataFrame(columns=['class', 'prob'])
        for e in df['output'].unique():
            mul = self.priors[e]
            prt = '-' * 50 + '\n'
            prt += "Class: " + e + " \nP(C = " + e + ") = " + str(mul) + "\n"
            for (columnName, columnData) in query.iteritems():
                prob = self.likelihood[
                    (self.likelihood['feature'] == columnName) & (self.likelihood['featureValue'] == columnData) & (
                            self.likelihood['class'] == e)]['prob'].values[0]
                prt += "P(" + columnName + " = " + str(columnData) + " | C = " + e + " ) = " + str(prob) + "\n"
                mul *= prob
            res = res.append({'class': e, 'prob': mul}, ignore_index=True)
            prt += "Final likelihook: " + str(mul) + "\n"
            if verbose:
                print(prt)
        min_max_scaler = preprocessing.MinMaxScaler()
        res = pd.concat([res, pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(res['prob'].values).values),
                                           columns=["normalizedProb"])], axis=1)
        if verbose:
            print(res.to_string())
        return res

from sklearn.cluster import KMeans
import numpy as np


class _KMeans:

    def runKMeans(self, x, c, k=2, verbose=True):
        verboseRes = "\n" + ("-" * 100) + "\nPoints:\n" + str(x) + "\nInitial Centers:\n" + str(c)
        i = 1
        kmeans = KMeans(n_clusters=k, random_state=0, init=c, max_iter=1, n_init=1).fit(x)
        oldCenters = kmeans.cluster_centers_
        verboseRes += "\n" + ("-" * 100) + "\nIteration: " + str(i) + "\nLabels: " + str(kmeans.labels_) + \
                     "\nCenters: " + str(kmeans.cluster_centers_)
        while (True):
            i += 1
            kmeans = KMeans(n_clusters=k, random_state=0, init=kmeans.cluster_centers_, max_iter=1, n_init=1).fit(x)
            verboseRes += "\n" + ("-" * 100) + "\nIteration: " + str(i) + "\nLabels: " + str(kmeans.labels_) + \
                          "\nCenters: " + str(kmeans.cluster_centers_)

            if (np.all(oldCenters == kmeans.cluster_centers_)):
                verboseRes += "\nAlgoritm converged in " + str(i) + " steps."
                if verbose:
                    print(verboseRes)
                break
            oldCenters = kmeans.cluster_centers_
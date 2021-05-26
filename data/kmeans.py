import os
import sys
import numpy as np
from sklearn.cluster import KMeans
import random
import math
import faiss

X = []
nm = []

def norm(x):
    n = 0
    for i in range(len(x)):
        n += x[i]*x[i]
    return math.sqrt(n)

'''
for subdir, dirs, files in os.walk("./PointDA_data/shapenetcorev2"):
    for f in files:
        if f.endswith("_pd.npy"):
            #print(os.path.join(subdir, f))
            nm.append(f[:-7].replace('_', '/'))
            x = np.load(os.path.join(subdir, f))
            X.append(x)
'''
with open("./entropy_sn55.txt", 'r') as f:
    for line in f:
        idx, h0, h1, h2 = line.split(' ')
        idx = idx.split('/')[1].replace('_', '/')
        nm.append(idx)
        X.append([float(h0), float(h1), float(h2[:-1])])

#X = np.array(X)
#X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

#for idx, i in enumerate(X):
#    X[idx] = i/norm(i) 

X = np.array(X).astype(np.float32)
kmeans = faiss.Kmeans(X.shape[1], 32, niter=20, verbose=True)
kmeans.train(X)
dists, labels = kmeans.index.search(X, 1)
'''
kmeans = KMeans(n_clusters=32, random_state=0).fit(np.array(X))
labels = np.array(kmeans.labels_)
'''

with open("./clusters/k32_entropy.txt", 'w') as o:
    for idx, i in enumerate(nm):
        o.write(i + " " + str(labels[idx])[1:-1] + '\n')

d = []
for i in range(np.max(labels)+1):
    print(str(i) + ": " + str(np.sum(labels==i)))
    d.append(np.sum(labels==i))

d = np.array(d)
print(np.std(d))

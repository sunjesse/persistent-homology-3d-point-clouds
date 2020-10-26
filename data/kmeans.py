import os
import sys
import numpy as np
from sklearn.cluster import KMeans

X = []
nm = []
for subdir, dirs, files in os.walk("./PointDA_data/shapenetcorev2"):
    for f in files:
        if f.endswith("_pd.npy"):
            print(os.path.join(subdir, f))
            nm.append(f[:-7].replace('_', '/'))
            x = np.load(os.path.join(subdir, f))
            X.append(x)

kmeans = KMeans(n_clusters=64, random_state=0).fit(np.array(X))
with open("./k64snv2.txt", 'w') as o:
    for idx, i in enumerate(nm):
        o.write(i + " " + str(kmeans.labels_[idx]) + '\n')
    
print(kmeans.labels_)

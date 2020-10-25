import os
import sys
import numpy as np
from sklearn.cluster import KMeans

X = []
nm = []
for subdir, dirs, files in os.walk("/home/rexma/Desktop/JesseSun/pcsll/data/PointDA_data/shapenetcorev2"):
    for f in files:
        if f.endswith("_pd.npy"):
            print(os.path.join(subdir, f))
            nm.append(f[:-7].replace('_', '/'))
            x = np.load(os.path.join(subdir, f))
            X.append(x)

kmeans = KMeans(n_clusters=32, random_state=0).fit(np.array(X))
with open("/home/rexma/Desktop/JesseSun/pcsll/data/k32snv2.txt", 'w') as o:
    for idx, i in enumerate(nm):
        o.write(i + " " + str(kmeans.labels_[idx]) + '\n')
    
print(kmeans.labels_)

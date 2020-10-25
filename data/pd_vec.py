from gudhi.representations.vector_methods import TopologicalVector
import gudhi
from perslocsig import compute_geodesic_persistence_diagrams as gpd
import numpy as np
from ripser import Rips
import pervect
import sys
import os

np.set_printoptions(threshold=sys.maxsize)

def set_and_reshape(v, dim, setBool=True):
    if setBool:
        v = list(set(v))
        v.sort(reverse=True)

    if len(v) < dim:
        c = len(v)
        for i in range(dim-c):
            v.append(0.)

    return np.array(v[0:dim])


def get_pd_vector(npy, rips, TV, dim=256, setBool=True):
    d = np.load(npy)
    D = rips.fit_transform(d)
    H0, H1 = D[0], D[1]
    ref = gudhi.representations.preprocessing.ProminentPoints(use=True, num_pts=25)
    v1 = ref(H1)
    #print(v1)
    v1 = np.log(1+v1)
    #print(v1)
    #rips_complex = gudhi.RipsComplex(points=d, max_edge_length=0.5)
    #simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    #diag = simplex_tree.persistence(min_persistence=0.1)
    #print(diag)
    vects = TV(v1)
    return vects
    return set_and_reshape(vects, dim, setBool) #np.mean(vects, axis=0)


#v = get_pd_vector("/home/rexma/Desktop/JesseSun/pcsll/data/PointDA_data/shapenet/table/train/010563.npy", rips)
#print(v)
#print(np.amax(v))
#print(v.shape)

#rips = Rips()
#TV = TopologicalVector(threshold=-1)
stop = False

if __name__ == '__main__':
    for subdir, dirs, files in os.walk("/home/rexma/Desktop/JesseSun/pcsll/data/PointDA_data/modelnet40"):
        #if subdir != "/home/rexma/Desktop/JesseSun/pcsll/data/PointDA_data/modelnet40/32" and stop == False:
        #    continue
        #elif subdir == "/home/rexma/Desktop/JesseSun/pcsll/data/PointDA_data/modelnet40/32":
        #    stop = True

        for f in files:
            if(f.endswith(".npy")) and f.endswith("_pd.npy") == False:
                rips = Rips()
                TV = TopologicalVector(threshold=-1)
                print(os.path.join(subdir, f))
                #save_as_off(os.path.join(subdir, f))
                #vec = np.log(1+get_pd_vector(os.path.join(subdir, f), rips, TV, setBool=False))
                vec = get_pd_vector(os.path.join(subdir, f), rips, TV, setBool=False)
                #vec = get_pd_vector(os.path.join(subdir, f), rips, TV, setBool=False)
                print(vec)
                #print(vec.shape)
                if not os.path.exists(os.path.join(subdir, "pd4")):
                    os.mkdir(os.path.join(subdir, "pd4"))
                np.save(os.path.join(subdir, "pd4", f[:-4]+"_pd.npy"), vec)

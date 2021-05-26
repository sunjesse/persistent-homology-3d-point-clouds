#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data
from gudhi.representations.vector_methods import TopologicalVector
import gudhi
from perslocsig import compute_geodesic_persistence_diagrams as gpd
from ripser import Rips
import pervect
import sys
#import ripser-plusplus.python.ripser_plusplus_python as rpp
#import importlib  
#rpp = importlib.import_module("ripser-plusplus.python.ripser_plusplus_python")

shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def persistent_entropy(dgm,dim, inf = False, valInf = np.float(0.0)):
    # From a diagram generated using ripser, this function computes its persistent
    # entropy. If inf = True, this mean that we want to keep infinity bars. Therefore,
    # it is important to give a value to the infinity valInf. To keep stability,
    # the value must be the same for all persistence diagram we are comparing.
    if inf == False:
        dgm = dgm[dim][dgm[dim][:,1]<np.inf]
        l = dgm[:,1]-dgm[:,0]
        L = np.sum(l)
        p = l/L
        E = -np.sum(p*np.log(p))
    else:
        dgm_valInf = dgm
        dgm_valInf[dim][dgm_valInf[dim][:,1]==np.inf]=np.array([0,valInf])
        l = dgm_valInf[dim][:,1]-dgm_valInf[dim][:,0]
        L = np.sum(l)
        p = l/L
        E = -np.sum(p*np.log(p))
    return E

def get_pd_vector(npy, rips, TV, dim=256, setBool=True):
    D = rips.fit_transform(npy)
    e0 = persistent_entropy(D, 0)
    e1 = persistent_entropy(D, 1)
    e2 = persistent_entropy(D, 2)
    '''
    v1 = ref(H1)
    v = np.concatenate((v0, v1), axis=0)
    v = np.log(1+v)
    vects = TV(v)
    '''
    vects = np.array([e0, e1, e2])
    return vects

def save_pc_as_npy(data, label, idx, split):
    folder = "./PointDA_data/shapenetcorev2_entropy_dim2"
    root = os.path.join(folder, str(label))
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(root+"/"+split):
        os.mkdir(root+"/"+split)
    idx = idx[:-4]
    np.save(root+"/"+split+"/"+idx+"_pd.npy", data)
    print("Saved " + root+"/"+split+"/"+idx+"_pd.npy")

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', class_choice=None,
            num_points=2048, split='train', load_name=True, load_file=True,
            segmentation=False, random_rotate=False, random_jitter=False, 
            random_translate=False):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 
            'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048        

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name not in ['shapenetcorev2', 'shapenetpart'] and segmentation == True:
            raise AssertionError

        self.root = os.path.join(root, dataset_name + '_' + '*hdf5_2048')
        self.dataset_name = dataset_name
        self.class_choice = class_choice
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        
        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        self.centroid = {}

        if self.split in ['train','trainval','all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val','trainval','all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label, seg = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.path_name_all.sort()
            self.name = self.load_json(self.path_name_all)    # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = self.load_json(self.path_file_all)    # load file name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 
        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0) 

        if self.class_choice != None:
            indices = (self.name == class_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            if self.segmentation:
                self.seg = self.seg[indices]
                self.seg_num_all = shapenetpart_seg_num[id_choice]
                self.seg_start_index = shapenetpart_seg_start_index[id_choice]
            if self.load_file:
                self.file = self.file[indices]
        elif self.segmentation:
            self.seg_num_all = 50
            self.seg_start_index = 0

        self.get_centroid("/home/rexma/Desktop/JesseSun/pcsll/data/clusters/k32_entropy.txt")

    def get_centroid(self, txt):
        with open(txt, 'r') as r:
            for line in r:
                i, n = line.split(" ")
                self.centroid[i] = int(n[:-1])

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.root, '%s*_id2file.json'%type)
            self.path_file_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        #label = self.label[item]

        if self.load_name:
            name = self.name[item]  # get label name
        if self.load_file:
            file = self.file[item]  # get file name

        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)

        label = np.copy(self.centroid[file[:-4]]) 
        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        #d = os.path.join("/home/rexma/Desktop/JesseSun/pcsll/data/PointDA_data/shapenetcorev2", name, "train", file.replace("/","_")[:-4]+"_pd.npy")
        #label = np.load(d)
        #label = torch.from_numpy(label)

        if self.segmentation:
            seg = self.seg[item]
            seg = torch.from_numpy(seg)
            return point_set, label, seg, name, file
        else:
            return point_set, label, name, file.replace("/", "_")

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy
    import faiss

    root = os.getcwd() + "/PointDA_data"
    rips = Rips(maxdim=2)
    TV = TopologicalVector(threshold=-1)

    # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
    dataset_name = 'shapenetcorev2'

    # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
    # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
    split = 'train'

    d = Dataset(root=root, dataset_name=dataset_name, num_points=512, split="train")
    print("datasize:", d.__len__())
    #ps, lb, n, f = d[0]
    y = []
    nm = []
    fts = []
    c = 0
    #VR = VietorisRipsPersistence(homology_dimensions=(0, 1))
    #PE = PersistenceEntropy()
    step = 10
    #while c < len(d) or c-step < len(d):
    #X = []
    ofile = open("./entropy_sn55.txt", 'a')
    started = False
    for item in range(0, len(d)):
        ps, lb, n, f = d[item]
        if f[:-4] != "02691156_371a609f050b4ed3f6497dc58a9a6f8a" and not started:
            print("Skipped! " + f[:-4])
            continue
        elif f[:-4] == "02691156_371a609f050b4ed3f6497dc58a9a6f8a" and not started:
            started = True
            print("Skipped! " + f[:-4])
            print("DONEEEEEEEEEEEEEE")
            continue

        path = os.path.join("./PointDA_data/shapenetcorev2_entropy_dim2", str(n), split, f[:-4]+"_pd.npy")
        if os.path.exists(path):
            print("Skipped " + f[:-4])
            try:
                ens = np.load(path, allow_pickle=True)
                ofile.write(str(n)+"/"+f[:-4] + " " +  str(ens[0]) + " "  + str(ens[1]) + " " + str(ens[2]) + '\n')
            except:
                v = get_pd_vector(ps, rips, TV)
                ofile.write(str(n)+"/"+f[:-4] + " " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + '\n')
                print("Wrote", f[:-4])
            continue

        #diagrams = VR.fit_transform([ps])
        #features = PE.fit_transform(diagrams)
        #save_pc_as_npy(features, n, f, split)
        v = get_pd_vector(ps, rips, TV)
        ofile.write(str(n)+"/"+f[:-4] + " " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + '\n')
        print("Wrote", f[:-4])
        #save_pc_as_npy(v, n, f, split)
    '''
        #X.append(ps)
        #y.append(lb)
        #nm.append(f.replace("_", "/"))
    #c += step

        #if idx > 3:
        #    break
        #print(ps.size(), lb, n, f)
        #v = get_pd_vector(ps, rips, TV)
        #save_pc_as_npy(v, n, f, split)

        diagrams = VR.fit_transform(X)
        features = PE.fit_transform(diagrams)
        fts.append(features)
        print("Done writing " + "c = " + str(c))
    
    features = np.concatenate(fts, axis=0)
    print(features.shape)
    features = np.array(features, order='C').astype(np.float32)
    kmeans = faiss.Kmeans(features.shape[1], 32, niter=20, verbose=True)
    kmeans.train(features)
    dists, labels = kmeans.index.search(features, 1)

    with open("./clusters/pe_h2_32.txt", 'a') as o:
        for i in range(len(y)):
            #print(str(y[i]), str(labels[i]))
            o.write(nm[i]+ " " +  str(labels[i][0]) + "\n")
    '''

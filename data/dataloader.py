import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from utils.pc_utils import (farthest_point_sample_np, scale_to_unit_cube, jitter_pointcloud,
                            rotate_shape, random_rotate_one_axis)
#from .pd_vec import get_pd_vector
#from gudhi.representations.vector_methods import TopologicalVector
from ripser import Rips
import torch

eps = 10e-4
NUM_POINTS = 1024
idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}


def load_data_h5py_scannet10(partition, dataroot):
    """
    Input:
        partition - train/test
    Return:
        data,label arrays
    """
    DATA_DIR = dataroot + '/PointDA_data/scannet'
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, '%s_*.h5' % partition))):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:]
        label = f['label'][:]
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return np.array(all_data).astype('float32'), np.array(all_label).astype('int64')

def add_pd_path(dr, pre="pd2"):
    tmp = dr.split('/')
    direc = ""
    for i in range(len(tmp)-1):
        direc += tmp[i] + "/"
    direc += pre + "/" + tmp[len(tmp)-1][:-4] + "_pd.npy"
    return direc

class ScanNet(Dataset):
    """
    scannet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train'):
        self.partition = partition

        # read data
        self.data, self.label = load_data_h5py_scannet10(self.partition, dataroot)
        self.num_examples = self.data.shape[0]

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in scannet" + ": " + str(self.data.shape[0]))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in scannet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.data[item])[:, :3]
        #label = np.copy(self.label[item])
        #label = get_pd_vector(self.pc_list[item], self.rips)
        label = np.load(self.pc_lost[item][:-4]+"_pd.npy")
        pointcloud = scale_to_unit_cube(pointcloud)
        # Rotate ScanNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(random_rotate_one_axis(pointcloud, "z"))

        return (pointcloud, label)

    def __len__(self):
        return self.data.shape[0]

    # scannet is rotated such that the up direction is the y axis
    def rotate_pc(self, pointcloud):
        pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud


class ModelNet(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train'):
        self.partition = partition
        self.pc_list = []
        self.lbl_list = []
        self.centroid = {}
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "modelnet")

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        self.get_centroid("./k16mn10.txt")

        io.cprint("number of " + partition + " examples in modelnet : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def get_centroid(self, txt):
        with open(txt, 'r') as r:
            for line in r:
                i, n = line.split(" ")
                self.centroid[i] = int(n[:-1])

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        label = np.copy(self.centroid[self.pc_list[item].split('/')[-1][:-4]])
        #label = np.copy(self.label[item])
        #label = np.load(add_pd_path(self.pc_list[item], pre="pd4")) # load from numpy file
        pointcloud = scale_to_unit_cube(pointcloud)

        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(random_rotate_one_axis(pointcloud, "z"))
        return (pointcloud, label)

    def __len__(self):
        return len(self.pc_list)


class ShapeNet(Dataset):
    """
    Sahpenet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train'):
        self.partition = partition
        self.pc_list = []
        self.lbl_list = []
        self.centroid = {}
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "shapenet")

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(label_to_idx[_dir.split('/')[-3]])

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        self.get_centroid("/home/rexma/Desktop/JesseSun/pcsll/data/k16sn10.txt")

        io.cprint("number of " + partition + " examples in shapenet: " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in shapenet " + partition + " set: " + str(dict(zip(unique, counts))))

    def get_centroid(self, txt):
        with open(txt, 'r') as r:
            for line in r:
                i, n = line.split(" ")
                self.centroid[i] = int(n[:-1])

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        label = np.copy(self.centroid[self.pc_list[item].split('/')[-1][:-4]])
        #label = np.copy(self.label[item])

        #label = np.load(add_pd_path(self.pc_list[item], pre="pd4")) #load from npy file
        #label = get_pd_vector(self.pc_list[item], self.rips)
        pointcloud = scale_to_unit_cube(pointcloud)
        # Rotate ShapeNet by -90 degrees
        pointcloud = self.rotate_pc(pointcloud, label)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(random_rotate_one_axis(pointcloud, "z"))
        return (pointcloud, label)

    def __len__(self):
        return len(self.pc_list)

    # shpenet is rotated such that the up direction is the y axis in all shapes except plant
    def rotate_pc(self, pointcloud, label):
        if label.item(0) != label_to_idx["plant"]:
            pointcloud = rotate_shape(pointcloud, 'x', -np.pi / 2)
        return pointcloud


class ModelNet40(Dataset):
    """
    modelnet dataset for pytorch dataloader
    """
    def __init__(self, io, dataroot, partition='train'):
        self.partition = partition
        self.pc_list = []
        self.lbl_list = []
        DATA_DIR = os.path.join(dataroot, "PointDA_data", "modelnet40")

        npy_list = sorted(glob.glob(os.path.join(DATA_DIR, '*', partition, '*.npy')))

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(int(_dir.split('/')[-3]))

        self.label = np.asarray(self.lbl_list)
        self.num_examples = len(self.pc_list)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in modelnet40 : " + str(len(self.pc_list)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in modelnet40 " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.load(self.pc_list[item])[:, :3].astype(np.float32)
        label = np.copy(self.label[item])
        #label = get_pd_vector(self.pc_list[item], self.rips)

        #label = np.load(add_pd_path(self.pc_list[item], pre="pd4")) # load from numpy file
        #label = np.exp(label)

        pointcloud = scale_to_unit_cube(pointcloud)
        # sample according to farthest point sampling
        if pointcloud.shape[0] > NUM_POINTS:
            pointcloud = np.swapaxes(np.expand_dims(pointcloud, 0), 1, 2)
            _, pointcloud = farthest_point_sample_np(pointcloud, NUM_POINTS)
            pointcloud = np.swapaxes(pointcloud.squeeze(), 1, 0).astype('float32')

        # apply data rotation and augmentation on train samples
        if self.partition == 'train' and item not in self.val_ind:
            pointcloud = jitter_pointcloud(random_rotate_one_axis(pointcloud, "z"))
        return (pointcloud, label)

    def __len__(self):
        return len(self.pc_list)

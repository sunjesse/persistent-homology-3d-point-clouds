import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from data.dataloader import ScanNet, ModelNet, ShapeNet, ModelNet40, label_to_idx, NUM_POINTS
from data.dataset import Dataset
from models import PointNet, DGCNN, LinearEval 
from utils import pc_utils
import DefRec
import PCM
from data.modelnet40 import ModelNet40
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

NWORKERS=4
MAX_LOSS = 9 * (10**9)

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='sslShapeNet10',  help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet40', 'modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet40', 'modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=300, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')
parser.add_argument('--apply_PCM', type=str2bool, default=False, help='Using mixup in source')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--cls_weight', type=float, default=0.5, help='weight of the classification loss')
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--weights', type=str, default="", help='model weights')
parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbours to use')
parser.add_argument('--emb_dims', type=int, default=2048, metavar='N', help="Dimension of embeddings")
args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')


# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler

src_dataset = args.src_dataset
#trgt_dataset = args.trgt_dataset
data_func = {'modelnet40': ModelNet40, 'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

src_trainset = ModelNet40(args.emb_dims, 'train')
src_testset = ModelNet40(args.emb_dims, 'test') 

# Creating data indices for training and validation splits:
#src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
#train_ind = np.array([i for i in range(len(src_trainset)) if i % 4 == 0]).astype(np.int)
#train_sampler = SubsetRandomSampler(train_ind)

#src_trainset = Dataset(root="/home/rexma/Desktop/JesseSun/pcsll/data/PointDA_data", dataset_name='modelnet10', num_points=2048, split="train")
#src_testset = Dataset(root="/home/rexma/Desktop/JesseSun/pcsll/data/PointDA_data", dataset_name='modelnet10', num_points=2048, split="test")

# dataloaders for source and target
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size, shuffle=True, drop_last=True)
src_test_loader = DataLoader(src_testset, num_workers=NWORKERS, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
print("train len: " + str(len(src_train_loader)*args.batch_size))
'''
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                               sampler=src_train_sampler, drop_last=True)
src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                             sampler=src_valid_sampler)
src_test_loader = DataLoader(src_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)
'''
# ==================
# Init Model
# ==================
if args.model == 'pointnet':
    model = PointNet(args)
elif args.model == 'dgcnn':
    model = DGCNN(args, output_channels=32, inf=True)
else:
    raise Exception("Not implemented")

if len(args.weights) > 0:
    model.load_state_dict(
                    torch.load(args.weights, map_location=lambda storage, loc: storage), strict=False)
    print("Loaded pretrained weights!")

#model.linear3 = nn.Linear(256, 40)# only for classification
model = model.to(device)

for n, p in model.named_parameters():
    if n.startswith("linearEval") == False:
        p.requires_grad = False

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)

# ==================
# Optimizer
# ==================
opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
#criterion = nn.MSELoss()
# lookup table of regions means
lookup = torch.Tensor(pc_utils.region_mean(args.num_regions)).to(device)


# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):

    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits = model(data)
            loss = criterion(logits, labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits.max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat


# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
best_model = io.save_model(model)

train_fts = []
train_lbs = []
test_fts = []
test_lbs = []

model.eval()

# init data structures for saving epoch stats
cls_type = 'mixup' if args.apply_PCM else 'cls'
src_print_losses = {"total": 0.0, cls_type: 0.0}
if args.DefRec_on_src:
    src_print_losses['DefRec'] = 0.0
trgt_print_losses = {'DefRec': 0.0}
src_count = trgt_count = 0.0

batch_idx = 1

for data1 in src_train_loader:#, data2 in zip(src_train_loader, trgt_train_loader):

    #### source data ####
    if data1 is not None:
        src_data, src_label = data1[0].to(device), data1[1].to(device).squeeze()
        # change to [batch_size, num_coordinates, num_points]
        src_data = src_data.permute(0, 2, 1)
        batch_size = src_data.size()[0]
        src_data_orig = src_data.clone()
        device = torch.device("cuda:" + str(src_data.get_device()) if args.cuda else "cpu")

        if args.DefRec_on_src:
            src_data, src_mask = DefRec.deform_input(src_data, lookup, args.DefRec_dist, device)
            src_logits = model(src_data, activate_DefRec=True)
            loss = DefRec.calc_loss(args, src_logits, src_data_orig, src_mask)
            src_print_losses['DefRec'] += loss.item() * batch_size
            src_print_losses['total'] += loss.item() * batch_size
            loss.backward()

        if args.apply_PCM:
            src_data = src_data_orig.clone()
            src_data, mixup_vals = PCM.mix_shapes(args, src_data, src_label)
            src_cls_logits = model(src_data, activate_DefRec=False)
            loss = PCM.calc_loss(args, src_cls_logits, mixup_vals, criterion)
            src_print_losses['mixup'] += loss.item() * batch_size
            src_print_losses['total'] += loss.item() * batch_size
            loss.backward()

        else:
            src_data = src_data_orig.clone()
            # predict with undistorted shape
            features = model(src_data)
            train_fts.append(features.detach().cpu().numpy())
            train_lbs.append(src_label.detach().cpu().numpy())

            
            #loss = args.cls_weight * criterion(src_cls_logits, src_label)#.float())
            #src_print_losses['cls'] += loss.item() * batch_size
            #src_print_losses['total'] += loss.item() * batch_size
            #loss.backward()

for data in src_test_loader:
    if data is not None:
        src_data, src_label = data[0].to(device), data[1].to(device).squeeze()
        src_data = src_data.permute(0, 2, 1)
        batch_size = src_data.size()[0]
        src_data_orig = src_data.clone()
        device = torch.device("cuda:" + str(src_data.get_device()) if args.cuda else "cpu")

        src_data = src_data_orig.clone()
        features = model(src_data)
        test_fts.append(features.detach().cpu().numpy())
        test_lbs.append(src_label.detach().cpu().numpy())

train_fts = np.concatenate(train_fts, axis=0)
train_lbs = np.concatenate(train_lbs, axis=0)
test_fts = np.concatenate(test_fts, axis=0)
test_lbs = np.concatenate(test_lbs, axis=0)

print(train_fts.shape)
print(train_lbs.shape)
print(test_fts.shape)
print(test_lbs.shape)

print("Training Linear SVM!")
scaling = MinMaxScaler(feature_range=(-1, 1)).fit(train_fts)
train_fts = scaling.transform(train_fts)
test_fts = scaling.transform(test_fts)

clf = LinearSVC(random_state=0)
clf.fit(train_fts, train_lbs)
result = clf.predict(test_fts)
accuracy = np.sum(result==test_lbs).astype(float) / np.size(test_lbs)
print("Transfer linear SVM accuracy: {:.4f}%".format(accuracy*100))

# print progress
#trgt_print_losses = {k: v * 1.0 / trgt_count for (k, v) in trgt_print_losses.items()}
    #trgt_acc = io.print_progress("Target", "Trn", epoch, trgt_print_losses)
    #===================
    # Validation
    #===================
    # src_val_acc, src_val_loss, src_conf_mat = test(src_val_loader, model, "Source", "Val", epoch)
    #trgt_val_acc, trgt_val_loss, trgt_conf_mat = test(trgt_val_loader, model, "Target", "Val", epoch)
    # save model according to best source model (since we don't have target labels)
    
#io.cprint("Best model was found at epoch %d, source validation accuracy: %.4f, source validation loss: %.4f,"
#          "target validation accuracy: %.4f, target validation loss: %.4f"
#          % (best_val_epoch, src_best_val_acc, src_best_val_loss))
#io.cprint("Best validtion model confusion matrix:")
#io.cprint('\n' + str(best_epoch_conf_mat))
#===================
# Test
#===================
#model = best_model
#trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
#io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_best_val_loss))
#io.cprint("Test confusion matrix:")
#io.cprint('\n' + str(trgt_conf_mat))

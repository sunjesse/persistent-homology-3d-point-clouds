#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100:4
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=18:30:00
#SBATCH --account=def-wanglab
#SBATCH --output=201025_SN55_64.out
source /home/sunjesse/projects/def-wanglab/sunjesse/ENV/bin/activate
module load python/3.6
#python ../inf.py --epochs 1 --out_path ../experiments --dataroot ../data --src_dataset modelnet40 --apply_PCM False --cls_weight 1.0 --DefRec_weight 0.2 --lr 1e-3 --wd 5e-5 --optimizer ADAM --emb_dims 1024 --weights /home/rexma/Desktop/JesseSun/pcsll/experiments/SN55_PRE_1024_32/model.pt
python ../train.py --epochs 180 --emb_dims 1024 --out_path ../experiments --dataroot ../data --src_dataset modelnet40 --apply_PCM False --cls_weight 1.0 --DefRec_weight 0.2 --lr 1e-3 --wd 5e-5 --exp_name SN55_PRE_1024_64 --optimizer ADAM 

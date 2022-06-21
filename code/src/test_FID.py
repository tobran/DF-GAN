import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint

import torch
from torchvision.utils import save_image,make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp
from lib.utils import load_netG,load_npz
from lib.perpare import prepare_dataloaders,prepare_models
from lib.modules import eval

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DF-GAN')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./cfg/model/coco.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: 4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--train', type=bool, default=False,
                        help='if train model')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def main(args): 
    multi_gpus = args.multi_gpus
    epoch = int(args.checkpoint.split('.')[-2].split('_')[-1])
    time_stamp = get_time_stamp()
    args.val_save_dir = osp.join(args.val_save_dir, time_stamp)
    if args.save_image==True:
        if (multi_gpus==True) and (get_rank() != 0):
            None
        else:
            mkdir_p(args.val_save_dir)
    # prepare data
    train_dl, valid_dl ,train_ds, valid_ds, _ = prepare_dataloaders(args)
    args.vocab_size = train_ds.n_words
    # prepare models
    _, text_encoder, netG, _, _ = prepare_models(args)
    model_path = osp.join(ROOT_PATH, args.checkpoint)
    netG = load_netG(netG, model_path, multi_gpus, train=False)
    netG.eval()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        pprint.pprint(args)
        print('Load %s for NetG'%(args.checkpoint))
        print("************ Start testing FID ************")
    start_t = time.time()
    m1, s1 = load_npz(args.npz_path)
    with torch.no_grad():
        fid = eval(valid_dl, text_encoder, netG, args.device, m1, s1, args.save_image, args.val_save_dir, \
                        args.sample_times, args.z_dim, args.batch_size, args.truncation, args.trunc_rate)
    end_t = time.time()
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('Sampling done, %.2fs cost, The FID is : %.2f'%(end_t-start_t, fid))
 

if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)

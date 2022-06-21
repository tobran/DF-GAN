import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid
import torchvision.utils as vutils
import multiprocessing as mp
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p, get_rank, merge_args_yaml, get_time_stamp, load_netG
from lib.utils import tokenize, truncated_noise, prepare_sample_data
from lib.perpare import prepare_models


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DF-GAN')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/model/coco.yml',
                        help='optional config file')
    parser.add_argument('--imgs_per_sent', type=int, default=16,
                        help='the number of images per sentence')
    parser.add_argument('--imsize', type=int, default=256,
                        help='image szie')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='if use GPU')
    parser.add_argument('--train', type=bool, default=False,
                        help='if training')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=2,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def build_word_dict(pickle_path):
    with open(pickle_path, 'rb') as f:
        x = pickle.load(f)
        wordtoix = x[3]
        del x
        n_words = len(wordtoix)
        print('Load from: ', pickle_path)
    return n_words, wordtoix


def sample_example(wordtoix, netG, text_encoder, args):
    batch_size, device = args.imgs_per_sent, args.device
    text_filepath, img_save_path = args.example_captions, args.samples_save_dir
    truncation, trunc_rate = args.truncation, args.trunc_rate
    z_dim = args.z_dim
    captions, cap_lens, _ = tokenize(wordtoix, text_filepath)
    sent_embs, _  = prepare_sample_data(captions, cap_lens, text_encoder, device)
    caption_num = sent_embs.size(0)
    # get noise
    if truncation==True:
        noise = truncated_noise(batch_size, z_dim, trunc_rate)
        noise = torch.tensor(noise, dtype=torch.float).to(device)
    else:
        noise = torch.randn(batch_size, z_dim).to(device)
    # sampling
    with torch.no_grad():
        fakes = []
        for i in tqdm(range(caption_num)):
            sent_emb = sent_embs[i].unsqueeze(0).repeat(batch_size, 1)
            fakes = netG(noise, sent_emb)
            img_name = osp.join(img_save_path,'Sent%03d.png'%(i+1))
            vutils.save_image(fakes.data, img_name, nrow=4, range=(-1, 1), normalize=True)
            torch.cuda.empty_cache()


def main(args):
    time_stamp = get_time_stamp()
    args.samples_save_dir = osp.join(args.samples_save_dir, time_stamp)
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(args.samples_save_dir) 
    # prepare data
    pickle_path = os.path.join(args.data_dir, 'captions_DAMSM.pickle')
    args.vocab_size, wordtoix = build_word_dict(pickle_path)
    # prepare models
    _, text_encoder, netG, _, _ = prepare_models(args)
    model_path = osp.join(ROOT_PATH, args.checkpoint)
    netG = load_netG(netG, model_path, args.multi_gpus, train=False)
    netG.eval()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('Load %s for NetG'%(args.checkpoint))
        print("************ Start sampling ************")
    start_t = time.time()
    sample_example(wordtoix, netG, text_encoder, args)
    end_t = time.time()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('*'*40)
        print('Sampling done, %.2fs cost, saved to %s'%(end_t-start_t, args.samples_save_dir))
        print('*'*40)


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
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

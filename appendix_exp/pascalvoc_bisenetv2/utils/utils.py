import os
import copy
import torch
import argparse
import numpy as np
from myseg.bisenet_utils import set_model_bisenetv2

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_parser():
    parser = argparse.ArgumentParser()
    # compression
    parser.add_argument('--com_type', type=str, default='fedsrn', help='com type')
    parser.add_argument('--noise_type', type=str, default='uniform_0.01', help=' ')
    parser.add_argument('--mask_type', type=str, default='binary', help='binary/signed')
    # model arguments
    parser.add_argument('--model', type=str, default='bisenetv2', help='model name')    
    parser.add_argument('--proj_dim', type=int, default=256, help='')
    parser.add_argument('--rand_init', type=str2bool, default=False, help='')
    parser.add_argument('--losstype', type=str, default='back',choices = ['ce','ohem','back','lovasz','dice','focal','bce'], help='background loss')
    # datasets
    parser.add_argument('--dataset', type=str, default='voc', choices=['cityscapes','camvid','ade20k','voc'],help="name of dataset")
    parser.add_argument('--root_dir', type=str, default='./voc', help="root of dataset")
    parser.add_argument('--data', type=str, default='train', choices=['train', 'val'], help='cityscapes train or val')
    parser.add_argument('--use_erase_data', type=str2bool, default=True, help='if use_erase_data') #
    parser.add_argument('--num_classes', type=int, default=20, help="number of classes max is 81, pretrained is 21")
    parser.add_argument('--num_users', type=int, default=60, help="number of users: K")
    parser.add_argument('--frac_num', type=int, default=5, help="the fraction num of clients used for training")
    # training
    parser.add_argument('--test_frequency', type=int, default=5, help='number of epochs to eval global model on test data')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.0, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (default: 0.0005)')
    parser.add_argument('--epochs', type=int, default=100, help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E, default is 10")
    parser.add_argument('--local_bs', type=int, default=8, help="local batch size: B, default=8, local gpu can only set 1")
    parser.add_argument("--log_dir", type=str, default="./log/debug/", help="dir")
    parser.add_argument("--log_name", type=str, default="debug", help="log")
    parser.add_argument('--num_workers', type=int, default=4, help='test colab gpu num_workers=1 is faster')
    parser.add_argument('--gpu', type=str, default='2', help='index of GPU to use')
    args = parser.parse_args()
    return args

def weighted_average_weights(w, client_dataset_len):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], client_dataset_len[0])  # w[0][key] * client_dataset_len[0]
        for i in range(1, len(w)):
            w_avg[key] += torch.mul((w[i][key]), client_dataset_len[i])  # w[i][key] * client_dataset_len[i]
        w_avg[key] = torch.div(w_avg[key], sum(client_dataset_len))
    return w_avg

def save_npy_record(npy_path, record, name=None):
    if name == None:
        name = "record"
    max_index = 0
    for filename in os.listdir(npy_path):
        if filename.startswith(name) and filename.endswith(".npy"):
            max_index +=1
    if max_index==0:
        np.save(npy_path+'/{}.npy'.format(name), record)
    else:
        np.save(npy_path+'/{}_{}.npy'.format(name, max_index), record)

def make_model(args):
    if args.model == 'bisenetv2':
        global_model = set_model_bisenetv2(args=args,num_classes=args.num_classes)
    else:
        exit('Error: unrecognized model')
    return global_model


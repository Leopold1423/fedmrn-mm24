import os
import copy
import torch
import random
import argparse
import numpy as np

from models.lstm import Dev_CharLSTM, SRN_CharLSTM
from models.shakespeare import ShakeSpeare
from models.update import test, fedavg, LocalUpdate
from utils.logger import get_log
from utils.tool import save_npy_record, get_device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', type=str, default='uniform_0.1', help=' ')
    parser.add_argument('--mask_type', type=str, default='binary', help='binary/signed')
    parser.add_argument('--com_type', type=str, default='fedsrn', help='com type')
    parser.add_argument('--model', type=str, default='lstm', help='model name')
    parser.add_argument('--dataset', type=str, default='shakespeare', help="name of dataset")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--rounds', type=int, default=50, help="rounds of training")
    parser.add_argument('--epochs', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=3.0, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum (default: 0.5)")
    parser.add_argument('--gpu', type=int, default=0, help="gpu id, -1 for cpu")
    parser.add_argument("--log_dir", type=str, default="./log/debug/", help="dir")
    parser.add_argument("--log_name", type=str, default="debug", help="log")
    args = parser.parse_args()
    # seed
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    # log
    logger = get_log(args.log_dir, args.log_name)
    pt_path = os.path.join(args.log_dir, args.log_name)
    os.makedirs(pt_path, exist_ok=True)
    # dataset
    dataset_train = ShakeSpeare(train=True)
    dataset_test = ShakeSpeare(train=False)
    dict_users = dataset_train.get_client_dic()
    args.num_users = len(dict_users)        # add num_users
    img_size = dataset_train[0][0].shape
    # model
    args.device = get_device(args.gpu)      # add device
    if args.model == "lstm":
        net_glob = Dev_CharLSTM().to(args.device)
    elif args.model == "srn_lstm":
        net_glob = SRN_CharLSTM(args).to(args.device)
    w_glob = net_glob.state_dict()
    # training
    logger.info(args)
    logger.info(net_glob)
    record = {"accuracy":[], "loss":[]}

    acc_test = []
    clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) for idx in range(args.num_users)]
    m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)
    for iter in range(args.rounds):
        w_locals, loss_locals, weight_locols= [], [], []
        idxs_users = np.random.choice(clients_index_array, m, replace=False)
        for idx in idxs_users:
            w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))
        # update global weights
        w_glob = fedavg(w_locals, weight_locols)
        net_glob.load_state_dict(w_glob)
        # test accuracy
        acc_t, loss_t = test(net_glob, dataset_test, args)
        record["loss"].append(loss_t)
        record["accuracy"].append(acc_t)
        logger.info("round {:3d}, test loss: {:.4f}, test accuracy: {:.4f}".format(iter, loss_t, acc_t))
    
    save_npy_record(pt_path, record)
    best_round = np.argmax(np.array(record["accuracy"]))
    best_acc = record["accuracy"][best_round]
    best_loss = record["loss"][best_round]
    logger.info("* best round: %d; best acc: %.4f; best loss: %.4f" %(best_round, best_acc, best_loss))    


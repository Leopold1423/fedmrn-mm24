import os
import copy
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from utils.logger import get_log
from models.srn import srn_replace_modules
from models.update import LocalUpdate, test_inference
from utils.utils import weighted_average_weights, save_npy_record, make_model, args_parser
from myseg.datasplit import get_dataset_ade20k


if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    # logger
    args = args_parser()
    logger = get_log(args.log_dir, args.log_name)
    pt_path = os.path.join(args.log_dir, args.log_name)
    os.makedirs(pt_path, exist_ok=True)
    logger.info(args)
    # device
    torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dataset
    if args.dataset == 'voc':
        train_dataset, test_dataset, user_groups = get_dataset_ade20k(args)
    else:
        exit('Error: unrecognized dataset')
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    # model
    global_model = make_model(args)
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    # compress part
    key_to_compress = [k for k in global_weights.keys() if "weight" in k and global_weights[k].dim()>1]
    logger.info(key_to_compress)
    # srn replace
    if args.com_type == "fedsrn":
        srn_replace_modules(global_model, args)
        srn_weights = copy.deepcopy(global_model.state_dict())
        for k, v in global_weights.items():
            srn_weights[k] = v
        global_model.load_state_dict(srn_weights, strict=True)
        global_model.to(device)
        global_weights = copy.deepcopy(global_model.state_dict())
    # training
    start_time = time.time()
    logger.info('train global model on {} of {} users locally for {} epochs'.format(args.frac_num, args.num_users, args.epochs))
    IoU_record, Acc_record = [], []
    record = {"accuracy": [0.0], "iou": [0.0]}

    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        client_dataset_len = [] # for non-IID weighted_average_weights
        global_model.train()
        idxs_users = np.random.choice(range(args.num_users), int(args.frac_num), replace=False)
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])            
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch, com_type=args.com_type)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            client_dataset_len.append(len(user_groups[idx])) # for non-IID weighted_average_weights

        loss_avg = sum(local_losses) / len(local_losses)
        logger.info('round {}, avg train loss {:.4f}'.format(epoch+1, loss_avg))

        # fedavg: average_weights
        global_weights = weighted_average_weights(local_weights, client_dataset_len)
        global_model.load_state_dict(global_weights)

        # evaluate
        if (epoch+1) % args.test_frequency == 0 or epoch == args.epochs-1:
            global_model.eval()
            test_acc, test_iou, confmat = test_inference(args, global_model, test_loader)
            # logger.info(confmat)
            logger.info('* test round: {}, test accuracy: {:.2f}%, test iou: {:.2f}%, time: {:.1f} min'.format(epoch+1, test_acc, test_iou, (time.time()-start_time)/60))
            if test_acc >= np.max(np.array(record["accuracy"])) or test_iou >= np.max(np.array(record["iou"])):
                torch.save(global_model.state_dict(), os.path.join(pt_path, "best.pt"))
                logger.info('** global model weights save to checkpoint')
            record["iou"].append(test_iou)
            record["accuracy"].append(test_acc)

    save_npy_record(pt_path, record)
    best_round = np.argmax(np.array(record["accuracy"]))
    best_acc = record["accuracy"][best_round]
    best_iou = record["iou"][best_round]
    logger.info("* best round: %d; best accuracy: %.4f; best iou: %.4f" %(best_round, best_acc, best_iou))
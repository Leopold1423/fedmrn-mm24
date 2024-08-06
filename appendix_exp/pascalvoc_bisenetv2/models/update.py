import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.eval_utils import evaluate
import myseg.bisenet_utils
from myseg.magic import MultiEpochsDataLoader
from myseg.bisenet_utils import OhemCELoss, BackCELoss
from models.srn import srn_generate_noise, srn_push_noise


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach().float(), label.clone().detach()

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader, self.testloader, self.trainloader_eval = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_val_test(self, dataset, idxs):
        # split indexes for train, and test (100%, 50%)
        idxs_train = idxs[:]
        idxs_test = idxs[:int(0.5*len(idxs))]
        # use MultiEpochsDataLoader to speed up training
        trainloader = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
                                            batch_size=self.args.local_bs, num_workers=self.args.num_workers,
                                            shuffle=True, drop_last=True, pin_memory=True)
        trainloader_eval = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
                                            batch_size=1, num_workers=self.args.num_workers,
                                            shuffle=False, drop_last=False, pin_memory=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=1, num_workers=self.args.num_workers,
                                shuffle=False)
        return trainloader, testloader, trainloader_eval

    def update_weights(self, model, global_round, com_type="fedavg"):
        if com_type == "fedsrn":
            srn_generate_noise(model)
        model.train()
        epoch_loss = []
        args = self.args
        # optimizer and loss
        if args.model == 'bisenetv2':
            optimizer = myseg.bisenet_utils.set_optimizer(model, args)
            if args.losstype=='ohem':
                criteria_pre = OhemCELoss(0.7)
                criteria_aux = [OhemCELoss(0.7) for _ in range(4)]  # num_aux_heads=4
            elif args.losstype=='ce':
                criteria_pre = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
                criteria_aux = [nn.CrossEntropyLoss(ignore_index=255, reduction='mean') for _ in range(4)]  # num_aux_heads=4
            elif args.losstype =='back':
                criteria_pre = BackCELoss(args)
                criteria_aux = [BackCELoss(args) for _ in range(4)]  # num_aux_heads=4
            else:
                raise ValueError('loss type is not defined')
        else:
            exit('Error: unrecognized model')

        # training
        for iter in range(args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if args.model == 'bisenetv2':   
                    logits, feat_head, *logits_aux = model(images)
                    labels_ = labels
                    loss_pre = criteria_pre(logits, labels_)
                    loss_aux = [crit(lgt, labels_) for crit, lgt in zip(criteria_aux, logits_aux)]
                    loss = loss_pre + sum(loss_aux)
                else:
                    exit('Error: unrecognized model')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        print('| global round : {} | local epochs : {} | {} images \t loss: {:.4f}'.format(
            global_round, args.local_ep, len(self.trainloader.dataset), loss.item()))     
        
        if com_type == "fedsrn":
            srn_push_noise(model)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

def test_inference(args, model, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    confmat = evaluate(model, testloader, device, args.num_classes)
    return confmat.acc_global, confmat.iou_mean, str(confmat)
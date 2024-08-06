import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.batch_size, shuffle=True)

    def train(self, net):
        if hasattr(net, "generate_noise"):
            net.generate_noise()
        net = net.to(self.args.device)
        net.train()
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        if hasattr(net, "push_noise"):
            net.push_noise()
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def fedavg(w, size):
    totalSize = sum(size)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        w_avg[k] = torch.div(w_avg[k], totalSize)
    return w_avg

def test(net_g, datatest, args):
    net_g.eval()
    test_loss, correct = 0, 0
    data_loader = DataLoader(datatest, batch_size=args.batch_size)
    for data, target in data_loader:
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    return accuracy, test_loss


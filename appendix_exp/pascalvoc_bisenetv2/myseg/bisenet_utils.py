import torch
from torch import nn
from myseg.bisenetv2 import BiSeNetV2


def set_model_bisenetv2(args,num_classes):
    net = BiSeNetV2(args,num_classes) 
    return net

def set_optimizer(model, args):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': args.lr * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': args.lr * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optim

class BackCELoss(nn.Module):
    def __init__(self, args, ignore_lb=255):
        super(BackCELoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.class_num = args.num_classes
        self.criteria = nn.NLLLoss(ignore_index=ignore_lb, reduction='mean')
    def forward(self, logits, labels):
        total_labels = torch.unique(labels)
        new_labels = labels.clone()
        probs = torch.softmax(logits,1)
        fore_ = []
        back_ = []
        
        for l in range(self.class_num):
            if l in total_labels:
                fore_.append(probs[:,l,:,:].unsqueeze(1))
            else: 
                back_.append(probs[:,l,:,:].unsqueeze(1))
        Flag=False
        if not  len(fore_)==self.class_num:
            fore_.append(sum(back_))
            Flag=True
        
        for i,l in enumerate(total_labels):
            if Flag :
                new_labels[labels==l]=i
            else: 
                if l!=255:
                    new_labels[labels==l]=i
        probs  =torch.cat(fore_,1)
        logprobs = torch.log(probs+1e-7)
        return self.criteria(logprobs,new_labels.long())

class OhemCELoss(nn.Module):
    '''
    Feddrive: We apply OHEM (Online Hard-Negative Mining) [56], selecting 25%
    of the pixels having the highest loss for the optimization.
    '''

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        # n_min = labels[labels != self.ignore_lb].numel() // 16
        n_min = int(labels[labels != self.ignore_lb].numel() * 0.25)
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)



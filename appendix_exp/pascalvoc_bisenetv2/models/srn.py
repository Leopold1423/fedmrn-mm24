import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from myseg.bisenetv2 import *


class masking(torch.autograd.Function):
    @staticmethod
    def forward(ctx, update, noise, mask_type):
        if mask_type == "binary":
            mask = torch.floor(update/(noise+1e-8)+torch.rand_like(update)).clamp(0, 1)
        elif mask_type == "signed":
            zero2one = (update/(noise+1e-8)+1)/2
            mask = torch.floor(zero2one + torch.rand_like(update)).clamp(0, 1)
            mask = mask*2 - 1
        final_noise = noise*mask
        return final_noise

    @staticmethod
    def backward(ctx, dy):
        return dy, None, None

class SRN_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, args=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.noise = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.update = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        self.mask_type = args.mask_type
        self.noise_type = args.noise_type
    
    def generate_noise(self):
        noise_type = self.noise_type.split("_")[0]
        alpha = float(self.noise_type.split("_")[1])

        seed = random.randint(0, 1000)
        generator = torch.Generator(device=self.noise.device).manual_seed(seed)
        if noise_type == "gauss":
            noise = torch.normal(mean=0, std=1.0, size=self.noise.shape, generator=generator, device=self.noise.device)
        elif noise_type == "uniform":
            noise = 2*torch.rand(size=self.noise.shape, generator=generator, device=self.noise.device)-1
        elif noise_type == "bernoulli":
            noise = 2*torch.randint(2, size=self.noise.shape, generator=generator, dtype=torch.float32, device=self.noise.device)-1

        self.noise.data = alpha * noise
        self.update.data = torch.zeros_like(self.weight)
    
    def push_noise(self):
        with torch.no_grad():
            update = masking.apply(self.update, self.noise, self.mask_type)
            self.weight.data = self.weight.data + update
            self.update.data = torch.zeros_like(self.weight)
            self.noise.data = torch.zeros_like(self.weight)

    def forward(self, x):
        update = masking.apply(self.update, self.noise, self.mask_type)
        return F.conv2d(x, self.weight + update, None, self.stride, self.padding, groups=self.groups)

target_modules = (Sequential, ConvBNReLU, DetailBranch, StemBlock, CEBlock, GELayerS1, GELayerS2, SegmentBranch, BGALayer)
head_modules = (SegmentHead, ProjectionHead)


def srn_replace_modules(model, args):
    for n, m in model._modules.items():
        if isinstance(m, nn.Conv2d):
            setattr(model, n, SRN_Conv2d(m.in_channels, m.out_channels, m.kernel_size[0], m.stride[0], m.padding, m.groups, args=args))
        if isinstance(m, target_modules):
            srn_replace_modules(m, args)
        if isinstance(m, head_modules):
            srn_replace_modules(m, args)

def srn_generate_noise(model):
    for n, m in model._modules.items():
        if isinstance(m, SRN_Conv2d):
            m.generate_noise()
        if isinstance(m, target_modules):
            srn_generate_noise(m)
        if isinstance(m, head_modules):
            srn_generate_noise(m)

def srn_push_noise(model):
    for n, m in model._modules.items():
        if isinstance(m, SRN_Conv2d):
            m.push_noise()
        if isinstance(m, target_modules):
            srn_push_noise(m)
        if isinstance(m, head_modules):
            srn_push_noise(m)


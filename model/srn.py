import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential


class masking(torch.autograd.Function):
    @staticmethod
    def forward(ctx, update, noise, mask_type):
        if mask_type == "binary":
            mask = torch.floor(update/noise + torch.rand_like(update)).clamp(0, 1)
        elif mask_type == "signed":
            mask = torch.floor((update/noise+1)/2 + torch.rand_like(update)).clamp(0, 1)
            mask = mask * 2 - 1
        else:
            raise ValueError("not supproted mask type.")
        return noise*mask

    @staticmethod
    def backward(ctx, dy):
        return dy, None, None

class clipping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, update, noise, mask_type):
        if mask_type == "binary":
            clipped_update = torch.clamp(update/noise, min=0, max=1) * noise
        elif mask_type == "signed":
            clipped_update = torch.clamp(update/noise, min=-1, max=1) * noise
        else:
            raise ValueError("not supproted mask type.")
        return clipped_update

    @staticmethod
    def backward(ctx, dy):
        return dy, None, None
    
class SRN_Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SRN_Linear, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.zeros((self.out_features, self.in_features)), requires_grad=False)
        self.noise = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.update = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        nn.init.normal_(self.weight, mean=0, std=0.01)
        nn.init.normal_(self.noise, mean=0, std=0.01)
        self.probability = torch.tensor(1.0)        

    def set_parameters(self, config):
        self.update.data = torch.zeros_like(self.weight)

        seed = random.randint(0, 1000)
        generator = torch.Generator().manual_seed(seed)
        self.mask_type, self.noise_type = config["mask_type"], config["noise_type"]
        noise_type, noise_magnitude = self.noise_type.split("_")
        if noise_type == "gauss":
            self.noise.data = torch.normal(mean=0, std=1.0, size=self.noise.shape, generator=generator)
        elif noise_type == "uniform":
            self.noise.data = 2*torch.rand(size=self.noise.shape, generator=generator)-1
        elif noise_type == "bernoulli":
            self.noise.data = 2*torch.randint(2, size=self.noise.shape, generator=generator, dtype=torch.float32)-1
        self.noise.data *= float(noise_magnitude)
        zero_position = (self.noise.data > -1e-8).float() * (self.noise.data < 1e-8).float()
        self.noise.data = zero_position * torch.tensor(1e-8) + (1-zero_position) * self.noise.data

    def get_weight(self):
        with torch.no_grad():
            weight = self.weight + masking.apply(self.update, self.noise, self.mask_type)
            return weight
        
    def forward(self, x):
        masked_update = masking.apply(self.update, self.noise, self.mask_type)      # SM
        
        if self.probability < 1:    # PM
            pm_mask = torch.bernoulli(self.probability*torch.ones_like(self.weight))
            clipped_update = clipping.apply(self.update, self.noise, self.mask_type)
            masked_update = masked_update * pm_mask + clipped_update * (1-pm_mask)  

        return F.linear(x, self.weight + masked_update, None)

class SRN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SRN_Conv2d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.weight = nn.Parameter(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), requires_grad=False)
        self.noise = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.update = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        nn.init.normal_(self.weight, mean=0, std=0.01)
        nn.init.normal_(self.noise, mean=0, std=0.01)
        self.probability = torch.tensor(1.0)

    def set_parameters(self, config):
        self.update.data = torch.zeros_like(self.weight)

        seed = random.randint(0, 1000)
        generator = torch.Generator().manual_seed(seed)
        self.mask_type, self.noise_type = config["mask_type"], config["noise_type"]
        noise_type, noise_magnitude = self.noise_type.split("_")
        if noise_type == "gauss":
            self.noise.data = torch.normal(mean=0, std=1.0, size=self.noise.shape, generator=generator)
        elif noise_type == "uniform":
            self.noise.data = 2*torch.rand(size=self.noise.shape, generator=generator)-1
        elif noise_type == "bernoulli":
            self.noise.data = 2*torch.randint(2, size=self.noise.shape, generator=generator, dtype=torch.float32)-1
        self.noise.data *= float(noise_magnitude)
        zero_position = (self.noise.data > -1e-8).float() * (self.noise.data < 1e-8).float()
        self.noise.data = zero_position * torch.tensor(1e-8) + (1-zero_position) * self.noise.data

    def get_weight(self):
        with torch.no_grad():
            weight = self.weight + masking.apply(self.update, self.noise, self.mask_type)
            return weight
        
    def forward(self, x):
        masked_update = masking.apply(self.update, self.noise, self.mask_type)      # SM
        
        if self.probability < 1:    # PM
            pm_mask = torch.bernoulli(self.probability*torch.ones_like(self.weight))
            clipped_update = clipping.apply(self.update, self.noise, self.mask_type)
            masked_update = masked_update * pm_mask + clipped_update * (1-pm_mask)  
        
        return F.conv2d(x, self.weight + masked_update, None, self.stride, self.padding)


def srn_replace_modules(model):
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, SRN_Conv2d(module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.padding))
        if isinstance(module, nn.Linear):
            setattr(model, name, SRN_Linear(module.in_features, module.out_features))
        if isinstance(module, Sequential):
            srn_replace_modules(module)

def srn_set_parameters(model, config):
    for name, module in model._modules.items():
        if isinstance(module, (SRN_Linear, SRN_Conv2d)):
            module.set_parameters(config)    
        if isinstance(module, Sequential):
            srn_set_parameters(module, config)

def srn_get_parameters(model):
    for name, module in model._modules.items():
        if isinstance(module, (SRN_Linear, SRN_Conv2d)):
            module.weight.data = module.get_weight()
        if isinstance(module, Sequential):
            srn_get_parameters(module)

def srn_set_probability(model, probability):
    for name, module in model._modules.items():
        if isinstance(module, (SRN_Linear, SRN_Conv2d)):
            module.probability = torch.tensor(float(probability)) 
        if isinstance(module, Sequential):
            srn_set_probability(module, probability)



import torch
import random
from torch import nn
import torch.nn.functional as F


class Dev_Embedding(nn.Module):
    def __init__(self, num_embeddings, embeddding_dim):
        super().__init__()
        self.num_embeddings, self.embeddding_dim = num_embeddings, embeddding_dim
        self.weight = nn.Parameter(torch.randn(self.num_embeddings, self.embeddding_dim))

    def forward(self, x):
        return F.embedding(x, self.weight)

class Dev_Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features))

    def forward(self, x):
        return F.linear(x, self.weight, None)

class Dev_LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.randn(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.randn(hidden_sz, hidden_sz * 4))

    def forward(self, x, init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class Dev_CharLSTM(nn.Module):
    def __init__(self, init_std=0.01):
        super().__init__()
        self.embed = Dev_Embedding(80, 8)
        self.lstm = Dev_LSTM(8, 256)
        self.drop = nn.Dropout()
        self.out = Dev_Linear(256, 80, bias=False)
        self.weight_init(init_std)
    
    def weight_init(self, std=0.01):
        for weight in self.parameters():
            nn.init.normal_(weight, mean=0, std=std)

    def forward(self, x):
        x = self.embed(x)
        x, hidden = self.lstm(x)
        x = self.drop(x)
        return self.out(x[:, -1, :])


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

class SRN_Embedding(nn.Module):
    def __init__(self, num_embeddings, embeddding_dim):
        super().__init__()   # the embedding table is not compressed
        self.num_embeddings, self.embeddding_dim = num_embeddings, embeddding_dim
        self.weight = nn.Parameter(torch.randn(self.num_embeddings, self.embeddding_dim))
        nn.init.normal_(self.weight, mean=0, std=0.01)

    def forward(self, x):
        return F.embedding(x, self.weight)

class SRN_Linear(nn.Module):
    def __init__(self, in_features, out_features, args, bias=False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.randn(self.out_features, self.in_features), requires_grad=True)
        nn.init.normal_(self.weight, mean=0, std=0.01)   

        self.noise = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.update = nn.Parameter(torch.zeros_like(self.weight), requires_grad=True)
        self.mask_type = args.mask_type
        self.noise_type = args.noise_type
    
    def generate_noise(self):
        noise_type = self.noise_type.split("_")[0]
        alpha = float(self.noise_type.split("_")[1])

        seed = random.randint(0, 1000)
        generator = torch.Generator().manual_seed(seed)
        if noise_type == "gauss":
            noise = torch.normal(mean=0, std=1.0, size=self.noise.shape, generator=generator)
        elif noise_type == "uniform":
            noise = 2*torch.rand(size=self.noise.shape, generator=generator)-1
        elif noise_type == "bernoulli":
            noise = 2*torch.randint(2, size=self.noise.shape, generator=generator, dtype=torch.float32)-1

        self.noise.data = alpha * noise
        self.update.data = torch.zeros_like(self.weight)
    
    def push_noise(self):
        with torch.no_grad():
            update = masking.apply(self.update, self.noise, self.mask_type)
            self.weight.data = self.weight.data + update
            self.update.data = torch.zeros_like(self.weight)
            self.noise.data = torch.zeros_like(self.weight)

    def forward(self, x):
        update = masking.apply(self.update, self.noise, self.mask_type)     # for simplicity, only SM is used
        return F.linear(x, self.weight + update, None)

class SRN_LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, args):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        nn.init.normal_(self.W, mean=0, std=0.01)   
        nn.init.normal_(self.U, mean=0, std=0.01)   

        self.noise_W = nn.Parameter(torch.zeros_like(self.W), requires_grad=False)
        self.update_W = nn.Parameter(torch.zeros_like(self.W), requires_grad=True)
        self.noise_U = nn.Parameter(torch.zeros_like(self.U), requires_grad=False)
        self.update_U = nn.Parameter(torch.zeros_like(self.U), requires_grad=True)
        self.mask_type = args.mask_type
        self.noise_type = args.noise_type
    
    def generate_noise(self):
        noise_type = self.noise_type.split("_")[0]
        alpha = float(self.noise_type.split("_")[1])

        seed = random.randint(0, 1000)
        generator = torch.Generator().manual_seed(seed)
        if noise_type == "gauss":
            noise_W = torch.normal(mean=0, std=1.0, size=self.noise_W.shape, generator=generator)
            noise_U = torch.normal(mean=0, std=1.0, size=self.noise_U.shape, generator=generator)
        elif noise_type == "uniform":
            noise_W = 2*torch.rand(size=self.noise_W.shape, generator=generator)-1
            noise_U = 2*torch.rand(size=self.noise_U.shape, generator=generator)-1
        elif noise_type == "bernoulli":
            noise_W = 2*torch.randint(2, size=self.noise_W.shape, generator=generator, dtype=torch.float32)-1
            noise_U = 2*torch.randint(2, size=self.noise_U.shape, generator=generator, dtype=torch.float32)-1

        self.noise_W.data = alpha * noise_W
        self.update_W.data = torch.zeros_like(self.W)
        self.noise_U.data = alpha * noise_U
        self.update_U.data = torch.zeros_like(self.U)
    
    def push_noise(self):
        with torch.no_grad():
            update_W = masking.apply(self.update_W, self.noise_W, self.mask_type)
            self.W.data = self.W.data + update_W
            self.update_W.data = torch.zeros_like(self.W)
            self.noise_W.data = torch.zeros_like(self.W)

            update_U = masking.apply(self.update_U, self.noise_U, self.mask_type)
            self.U.data = self.U.data + update_U
            self.update_U.data = torch.zeros_like(self.U)
            self.noise_U.data = torch.zeros_like(self.U)

    def forward(self, x, init_states=None):
        update_W = masking.apply(self.update_W, self.noise_W, self.mask_type)
        actual_W = self.W + update_W
        update_U = masking.apply(self.update_U, self.noise_U, self.mask_type)
        actual_U = self.U + update_U
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ actual_W + h_t @ actual_U
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

class SRN_CharLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = SRN_Embedding(80, 8)
        self.lstm = SRN_LSTM(8, 256, args)
        self.drop = nn.Dropout()
        self.out = SRN_Linear(256, 80, args, bias=False)

    def generate_noise(self):
        self.lstm.generate_noise()
        self.out.generate_noise()
    
    def push_noise(self):
        self.lstm.push_noise()
        self.out.push_noise()

    def forward(self, x):
        x = self.embed(x)
        x, hidden = self.lstm(x)
        x = self.drop(x)
        return self.out(x[:, -1, :])


import torch
import torch.nn as nn


class CNN4(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN4, self).__init__()
        c=[32,64,128,256]
        self.features = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(c[3]*4, num_classes, bias=False)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN4_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN4_MNIST, self).__init__()
        c=[32,64,128,256]
        self.features = nn.Sequential(
            nn.Conv2d(1, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),            
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(c[3], num_classes, bias=False)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CNN8(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN8, self).__init__()
        c=[32,32,64,64,128,128,256,256]
        self.features = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[0], c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[1], c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[2], c[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[3], c[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[4], c[5], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[5]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c[5], c[6], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[6]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c[6], c[7], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[7]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(c[7]*4, num_classes, bias=False)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = CNN8()
    x=torch.rand((1,3,32,32))
    model(x)
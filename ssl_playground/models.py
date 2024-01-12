from torch import flatten, diagonal
from torch.nn import Conv2d, Linear, Identity, Module
from torchvision.models import resnet18

BT_LAMBD = 0.0051

def create_supervised_resnet18():
    model = resnet18()
    model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = Linear(in_features=512, out_features=10, bias=True)
    return model

def create_self_supervised_resnet18():
    model = resnet18()
    model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = Identity()
    return model

def create_resnet18(config):
    configs = ["supervised", "self-supervised"]
    assert config in configs, "Resnet18 config not supported!"
    if config == "supervised":
        return create_supervised_resnet18()
    return create_self_supervised_resnet18()

def shape_trace_resnet(model, images):
    x = images
    print("Initial:", x.size())
    x = model.conv1(x)
    print("Conv1:", x.size())
    x = model.maxpool(x)
    print("Maxpool:", x.size())
    x = model.layer1(x)
    print("Layer1:", x.size())
    x = model.layer2(x)
    print("Layer2:", x.size())
    x = model.layer3(x)
    print("Layer3:", x.size())
    x = model.layer4(x)
    print("Layer4:", x.size())
    x = model.avgpool(x)
    print("AvgPool:", x.size())
    x = flatten(x, 1)
    print("Flatten:", x.size())
    x = model.fc(x)
    print("fc:", x.size())

def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss:

    def __init__(self, lambd):
        self.lambd = lambd

    def apply(self, outputs1, outputs2):
        
        c = outputs1.T @ outputs2

        on_diag = diagonal(c).add_(-1).pow_(2).sum()
        off_diag = BarlowTwinsLoss.off_diagonal(c).pow_(2).sum()
        return on_diag + self.lambd * off_diag

class AugmentedResnet(Module):

    def __init__(self, model_class="r18", injection_block=1):
        assert model_class in ["r18"], "Only ResNet 18 is currently supported!"
        assert injection_block in range(6), "Can only inject at the end of a single block: 0, 1, 2, 3, 4, or 5."
        self.resnet = create_self_supervised_resnet18()
    
    def partial_forward(self, x, end_block):
        assert end_block in range(6)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        if end_block == 0: return x
        x = self.layer1(x)
        if end_block == 1: return x
        x = self.layer2(2)
        if end_block == 2: return x
        x = self.layer3(x)
        if end_block == 3: return x
        x = self.layer4(x)
        x = self.avgpool(x)
        if end_block == 4: return x
        x = self.fc(x)
        return x

    def forward(self, x):
        return self.resnet(x) 

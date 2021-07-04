import torch
import torch.nn as nn
import torch.optim as optim
class One_Layer_Net(nn.Module):
    def __init__(self):
        super(One_Layer_Net, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200),)
        self.fc2 = nn.Sequential(
            nn.Linear(200, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Sigmoid_Net1(nn.Module):
    def __init__(self):
        super(Sigmoid_Net1, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200),
                                 nn.Sigmoid())
        self.fc2 = nn.Sequential(
            nn.Linear(200, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ReLU_Net1(nn.Module):
    def __init__(self):
        super(ReLU_Net1, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(200, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Batch_Net1(nn.Module):
    def __init__(self):
        super(Batch_Net1, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200),nn.BatchNorm1d(200),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(200, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Drop_Net1(nn.Module):
    def __init__(self):
        super(Batch_Net1, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200),torch.nn.Dropout(0.3),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(200, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Three_Layer_Net(nn.Module):
    def __init__(self):
        super(Three_Layer_Net, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200),nn.Linear(200, 100),nn.Linear(100, 150))
        self.fc2 = nn.Sequential(
            nn.Linear(150, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Sigmoid_Net2(nn.Module):
    def __init__(self):
        super(Sigmoid_Net2, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200),
                                 nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(200,150),
                                 nn.Sigmoid(),
                                 nn.Linear(150, 100),
                                 nn.Sigmoid()
                                 )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class ReLU_Net2(nn.Module):
    def __init__(self):
        super(ReLU_Net2, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(200,150),
                                 nn.ReLU(),
                                 nn.Linear(150, 100),
                                 nn.ReLU()
                                 )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class Batch_Net2(nn.Module):
    def __init__(self):
        super(Batch_Net2, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200), nn.BatchNorm1d(200),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(200,150), nn.BatchNorm1d(150),
                                 nn.ReLU(),
                                 nn.Linear(150, 100), nn.BatchNorm1d(100),
                                 nn.ReLU()
                                 )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
class Drop_Net2(nn.Module):
    def __init__(self):
        super(Drop_Net2, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(28*28, 200), torch.nn.Dropout(0.3),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(200,150), torch.nn.Dropout(0.3),
                                 nn.ReLU(),
                                 nn.Linear(150, 100), torch.nn.Dropout(0.3),
                                 nn.ReLU()
                                 )
        self.fc3 = nn.Sequential(
            nn.Linear(100, 10))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5, 1, 2), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))
        	# 最后的结果一定要变为 10，因为数字的选项是 0 ~ 9

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class L1(torch.nn.Module):
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)
import torchvision
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import argparse
import torchvision.models as models
import torch.nn.functional as F
train_data = torchvision.datasets.ImageFolder('./sample',
                                              transform=transforms.Compose([
                                                # transforms.Resize(256),
                                                #
                                                #   transforms.RandomCrop(64, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
                                                  transforms.Resize(160),

                                                  transforms.RandomCrop(144, padding=4),  # 先四周填充0，在吧图像随机裁剪成112
                                                   # transforms.RandomResizedCrop(144),


                                                transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                                                transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                              )
BATCH_SIZE = 64
epoch = 40
n = len(train_data)  # total number of examples
lista=[i for i in range(0,n,5)]
random.shuffle(lista)
listb=[i for i in range(n) if i not in lista]
random.shuffle(listb)
# print(listb)
# print(lista)
test_loader = torch.utils.data.Subset(train_data,lista)  # take first 10%
test_loader = torch.utils.data.DataLoader(test_loader, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=2)
train_loader = torch.utils.data.Subset(train_data,listb)  # take the rest
train_loader = torch.utils.data.DataLoader(train_loader, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

# def show_batch(imgs):
#     grid = utils.make_grid(imgs,nrow=5)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))
#     plt.title('Batch from train_loader')
# #
#
# for i, (batch_x, batch_y) in enumerate(data_loader):
#     if(i<4):
#         print(i, batch_x.size(), batch_y.size())
#
#         show_batch(batch_x)
#         plt.axis('off')
#         plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Simple_CNN_Net(nn.Module):
    def __init__(self):
        super(Simple_CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(17424, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 17424)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGG15(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG15, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 10
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 12
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(8192, 100),
            nn.ReLU(True),
            nn.Dropout(),
            # 15
            nn.Linear(100, num_classes),
        )
        # self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out

def simple_train_cifar10():


    LR = 0.0007
    # l2的默认WEIGHT_DECAY默认为0
    WEIGHT_DECAY = 5e-4

    # 模型

    net = Simple_CNN_Net().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    # 定义两个数组
    Loss_list = []
    Accuracy_list = []
    train_correct_list=[]
    a2_list = []
    a3_list = []
    a4_list = []


    # net.load_state_dict(torch.load('./parameterForme/parameter1.pkl'))
    for _ in range(epoch):
        net.train()
        sum_loss = 0.0
        train_correct=0
        total_train=0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # print(i)
            # print(labels)
            # print(labels)
            # print('type(images) = ', type(inputs))
            # print('type(labels) = ', type(labels))
            # labels=torch.from_numpy(labels)
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            optimizer.zero_grad()  # 将梯度归零
            outputs = net(inputs)  # 将数据传入网络进行前向运算
            loss = criterion(outputs, labels)  # 得到损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 通过梯度做一步参数更新
            # print(loss)
            sum_loss += loss.item()
            __, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum()
            if i % 35 == 34:
                print('[%d,%d] loss:%.03f,test_acc::%.03f' %
                      (_ + 1, i + 1, sum_loss / 35,train_correct/total_train))
                Loss_list.append(sum_loss /35)
                sum_loss = 0.0
                train_correct_list.append(train_correct/total_train)

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            TP=0
            TN=0
            FN=0
            FP=0
            for data in test_loader:

                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                TP += ((predicted == labels) & (labels == torch.ones_like(labels))).sum().item()
                TN += ((predicted == labels) & (labels == torch.zeros_like(labels))).sum().item()
                FN += ((predicted != labels) & (labels == torch.ones_like(labels))).sum().item()
                FP += ((predicted != labels) & (labels == torch.zeros_like(labels))).sum().item()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            a1 = (TP + TN) / (FN + FP + TP + TN)
            a2 = TP / (TP + FP)
            a3 = TP / (TP + FN)
            a4 = 2 * a2 * a3 / (a2 + a3)
            print("准确率={},精确率={},召回率={},F值={}".format(a1, a2, a3, a4))
            acc = correct / total
            Accuracy_list.append(acc)
            a2_list.append(a2)
            a3_list.append(a3)
            a4_list.append(a4)

    net.eval()  # 将模型变换为测试模式
    correct = 0
    total = 0
    for data_test in test_loader:
        images, labels = data_test
        images, labels = Variable(images).to(device), Variable(labels).to(device)
        output_test = net(images)
        _, predicted = torch.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("correct: ", correct)
    print("Test acc: {0}".format(correct.item() / len(test_loader.dataset)))
    # 保存
    if not os.path.exists("./parameterForme"):
        os.mkdir("./parameterForme")
    torch.save(net.state_dict(), './parameterForme/parameter1.pkl')

    # 作图
    # x1 = range(0, len(Accuracy_list))
    # x2 = range(0, len(Loss_list))
    # y1 = Accuracy_list
    # y2 = Loss_list
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-')
    # plt.title('Test accuracy vs. epoches')
    # plt.ylabel('Test accuracy')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '.-')
    # plt.xlabel('Test loss vs. 100')
    # plt.ylabel('Test loss')
    # plt.text(x=1,y=1,s="acc={}".format(((correct.item() / len(test_loader.dataset)) )))
    # plt.savefig("./parameterForme/pic_1.jpg")
    # plt.show()

    plt.subplot(2, 1, 1)

    plt.plot(Loss_list, label='Loss_list')
    plt.plot(train_correct_list, label='train_correct_list')
    plt.title("Train")

    plt.subplot(2, 1, 2)
    plt.plot(Accuracy_list, label='Accuracy_list')
    plt.plot(a2_list, label='Precision')
    plt.plot(a3_list, label='Recall')
    plt.plot(a4_list, label='F1')
    plt.title("test")
    plt.legend()
    plt.savefig("./parameterForme/pic_1_0.jpg")

if __name__ == '__main__':
    simple_train_cifar10()
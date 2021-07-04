import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import model_assignment1 as mymodel
import matplotlib.pyplot as plt

# 下载训练集
BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=True, download=False,
                   transform=transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.1037,), (0.3081,))])), batch_size=BATCH_SIZE,
    shuffle=True)

# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.0007
epoch = 35
# l2的默认WEIGHT_DECAY默认为0
WEIGHT_DECAY = 0

choice = int(input("choose the model\n"
                   "1:One_Layer_Net\n"
                   "2:Sigmoid_Net1\n"
                   "3:ReLU_Net1\n"
                   "4:Batch_Net1\n"
                   "5:Drop_Net1\n"
                   "6:Two_Layer_Net\n"
                   "7:Sigmoid_Net2\n"
                   "8:ReLU_Net2\n"
                   "9:Batch_Net2\n"
                   "10:Drop_Net2\n"
                   "11:Drop_Net2+L1\n"
                   "12:LeNet\n:"))
if choice == 1:
    net = mymodel.One_Layer_Net().to(device)
elif choice == 2:
    net = mymodel.Sigmoid_Net1().to(device)
elif choice == 3:
    net = mymodel.ReLU_Net1().to(device)
elif choice == 4:
    net = mymodel.Batch_Net1().to(device)
elif choice == 5:
    net = mymodel.Drop_Net1().to(device)
elif choice == 6:
    net = mymodel.Three_Layer_Net().to(device)
elif choice == 7:
    net = mymodel.Sigmoid_Net2().to(device)
elif choice == 8:
    net = mymodel.ReLU_Net2().to(device)
elif choice == 9:
    net = mymodel.Batch_Net2().to(device)
elif choice == 10:
    net = mymodel.Drop_Net2().to(device)
elif choice == 11:
    net = mymodel.L1(mymodel.LeNet(), 0.04).to(device)
elif choice == 12:
    net = mymodel.LeNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    net.parameters(),
    lr=LR,weight_decay=WEIGHT_DECAY
)


if __name__ == '__main__':

    # 定义两个数组
    Loss_list = []
    Accuracy_list = []

    for epoch in range(epoch):
        net.train()
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            optimizer.zero_grad()  # 将梯度归零
            outputs = net(inputs)  # 将数据传入网络进行前向运算
            loss = criterion(outputs, labels)  # 得到损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 通过梯度做一步参数更新

            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                Loss_list.append(sum_loss / 100)
                sum_loss = 0.0
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
        Accuracy_list.append(correct.item() / len(test_loader.dataset))

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
    if not os.path.exists("./parameterForMnistFashion"):
        os.mkdir("./parameterForMnistFashion")
    torch.save(net.state_dict(), './parameterForMnistFashion/parameter{}.pkl'.format(choice))
    # # 加载
    # model = TheModelClass(...)
    # model.load_state_dict(torch.load('\parameter.pkl'))


    #作图
    x1 = range(0, len(Accuracy_list))
    x2 = range(0, len(Loss_list))
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. 100')
    plt.ylabel('Test loss')
    plt.savefig("./parameterForMnistFashion/fashionmnist_accuracy_loss{0}.jpg".format(choice))
    plt.show()

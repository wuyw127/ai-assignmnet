import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import model_assignment2 as mymodel
import matplotlib.pyplot as plt
import argparse
import torchvision.models as models
BATCH_SIZE = 128
epoch =80
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def simple_train_cifar10():
    choice=1
    # 数据集
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=False, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

    LR = 0.0007
    # l2的默认WEIGHT_DECAY默认为0
    WEIGHT_DECAY = 0

    # 模型

    net = mymodel.Simple_CNN_Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    # 定义两个数组
    Loss_list = []
    Accuracy_list = []


    # net.load_state_dict(torch.load('./parameterForCifar10/parameter{}.pkl'.format(choice)))
    for _ in range(epoch):
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
                      (_ + 1, i + 1, sum_loss / 100))
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
    if not os.path.exists("./parameterForCifar10"):
        os.mkdir("./parameterForCifar10")
    torch.save(net.state_dict(), './parameterForCifar10/parameter{}.pkl'.format(choice))
    # # 加载
    # model = TheModelClass(...)
    # model.load_state_dict(torch.load('\parameter.pkl'))

    # 作图
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
    plt.text(x=1,y=1,s="acc={}".format(((correct.item() / len(test_loader.dataset)) )))
    plt.savefig("./parameterForCifar10/Cifar10_accuracy_loss{0}.jpg".format(choice))
    plt.show()
def image_enhancement():
    # torchvision.transforms包括所有图像增强的方法：
    # 第一个函数是
    # Scale，对图片的尺寸进行缩小或者放大；第二个函数是
    # CenterCrop，对图像正中心进行给定大小的裁剪；第三个函数是
    # RandomCrop，对图片进行给定大小的随机裁剪；第四个函数是
    # RandomHorizaontalFlip，对图片进行概率为0
    # .5
    # 的随机水平翻转；第五个函数是
    # RandomSizedCrop，首先对图片进行随机尺寸的裁剪，然后再对裁剪的图片进行一个随机比例的缩放，最后将图片变成给定的大小，
    # 这在InceptionNet中比较流行；最后一个是
    # pad，对图片进行边界零填充；
    #
    # 上面介绍了PyTorch内置的一些图像增强的方法，还有更多的增强方法，可以使用OpenCV或者PIL等第三方图形库实现。
    # 在网络的训练的过程中图形增强是一种常见、默认的做法，对多任务进行图像增强之后能够在一定程度上提升任务的准确率。
    choice=2
    # 数据集
    train_transform = transforms.Compose([transforms.Resize(40),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=False, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

    LR = 0.0007
    # l2的默认WEIGHT_DECAY默认为0
    WEIGHT_DECAY = 0

    # 模型

    net = mymodel.Simple_CNN_Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    # 定义两个数组
    Loss_list = []
    Accuracy_list = []


    # net.load_state_dict(torch.load('./parameterForCifar10/parameter{}.pkl'.format(choice)))
    for _ in range(epoch):
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
                      (_ + 1, i + 1, sum_loss / 100))
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
    if not os.path.exists("./parameterForCifar10"):
        os.mkdir("./parameterForCifar10")
    torch.save(net.state_dict(), './parameterForCifar10/parameter{}.pkl'.format(choice))
    # # 加载
    # model = TheModelClass(...)
    # model.load_state_dict(torch.load('\parameter.pkl'))

    # 作图
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
    plt.text(x=1,y=1,s="acc={}".format(((correct.item() / len(test_loader.dataset)) )))
    plt.savefig("./parameterForCifar10/Cifar10_accuracy_loss{0}.jpg".format(choice))
    plt.show()
def regularization():
    choice=3
    # 数据集
    train_transform = transforms.Compose([transforms.Resize(40),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=False, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

    LR = 0.000001
    # l2的默认WEIGHT_DECAY默认为0

    # 模型

    net = mymodel.Simple_CNN_Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR, weight_decay=5e-4,
    )

    # 定义两个数组
    Loss_list = []
    Accuracy_list = []


    net.load_state_dict(torch.load('./parameterForCifar10/parameter{}.pkl'.format(choice)))
    for _ in range(epoch):
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
                      (_ + 1, i + 1, sum_loss / 100))
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
    if not os.path.exists("./parameterForCifar10"):
        os.mkdir("./parameterForCifar10")
    torch.save(net.state_dict(), './parameterForCifar10/parameter{}.pkl'.format(choice))
    # # 加载
    # model = TheModelClass(...)
    # model.load_state_dict(torch.load('\parameter.pkl'))

    # 作图
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
    plt.text(x=1,y=1,s="acc={}".format(((correct.item() / len(test_loader.dataset)) )))
    plt.savefig("./parameterForCifar10/Cifar10_accuracy_loss{0}_2.jpg".format(choice))
    plt.show()
def resnet():
    choice=4
    # 数据集

    train_transform = transforms.Compose([transforms.Resize(40),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=False, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)

    LR = 0.0007
    # l2的默认WEIGHT_DECAY默认为0
    WEIGHT_DECAY = 0

    # 模型

    net = mymodel.ResNet18().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    # 定义两个数组
    Loss_list = []
    Accuracy_list = []


    # net.load_state_dict(torch.load('./parameterForCifar10/parameter{}.pkl'.format(choice)))
    for _ in range(epoch):
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
            print(i)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (_ + 1, i + 1, sum_loss / 100))
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
        print('acc:%.03f' %
              (correct.item() / len(test_loader.dataset)))

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
    if not os.path.exists("./parameterForCifar10"):
        os.mkdir("./parameterForCifar10")
    torch.save(net.state_dict(), './parameterForCifar10/parameter{}.pkl'.format(choice))
    # # 加载
    # model = TheModelClass(...)
    # model.load_state_dict(torch.load('\parameter.pkl'))

    # 作图
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
    plt.text(x=1,y=1,s="acc={}".format(((correct.item() / len(test_loader.dataset)) )))
    plt.savefig("./parameterForCifar10/Cifar10_accuracy_loss{0}.jpg".format(choice))
    plt.show()
def resnet_change():
    choice=5
    # 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
    parser.add_argument('--net', default='./model/Resnet18.pth',
                        help="path to net (to continue training)")  # 恢复训练时的模型路径
    args = parser.parse_args()

    # 超参数设置
    EPOCH = 4  # 遍历数据集次数
    pre_epoch = 0  # 定义已经遍历数据集的次数
    LR = 0.1  # 学习率

    # 准备数据集并预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=False,
                                            transform=train_transform)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 模型定义-ResNet
    net = mymodel.ResNet18().to(device)
    # 取得之前的参数
    # net.load_state_dict(torch.load('./parameterForCifar10/parameter5.pkl'))

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    # 训练
    best_acc = 85  # 2 初始化best test accuracy
    Loss_list = []
    Accuracy_list = []
    for epoch in range(pre_epoch, EPOCH):
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            # 准备数据
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            Loss_list.append(sum_loss / (i + 1))
            Accuracy_list.append(100. * correct / total)
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total
    print("Training Finished, TotalEPOCH=%d" % EPOCH)

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
    if not os.path.exists("./parameterForCifar10"):
        os.mkdir("./parameterForCifar10")
    torch.save(net.state_dict(), './parameterForCifar10/parameter{}.pkl'.format(choice))
    # # 加载
    # model = TheModelClass(...)
    # model.load_state_dict(torch.load('\parameter.pkl'))

    # 作图
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
    plt.text(x=1,y=1,s="acc={}".format(((correct.item() / len(test_loader.dataset)) )))
    plt.savefig("./parameterForCifar10/Cifar10_accuracy_loss{0}.jpg".format(choice))
    plt.show()

def VGG16():
    choice=6
    # 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
    parser.add_argument('--net', default='./model/Resnet18.pth',
                        help="path to net (to continue training)")  # 恢复训练时的模型路径
    args = parser.parse_args()

    # 超参数设置
    EPOCH = 10  # 遍历数据集次数
    pre_epoch = 0  # 定义已经遍历数据集的次数
    LR = 0.001  # 学习率

    # 准备数据集并预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=False,
                                            transform=train_transform)  # 训练数据集
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 模型定义-ResNet
    net = mymodel.VGG16().to(device)
    # net=models.vgg16(pretrained=True).to(device)
    # 取得之前的参数
    net.load_state_dict(torch.load('./parameterForCifar10/parameter6.pkl'))

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    # 训练
    best_acc = 85  # 2 初始化best test accuracy
    Loss_list = []
    Accuracy_list = []
    for epoch in range(pre_epoch, EPOCH):
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            # 准备数据
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            Loss_list.append(sum_loss / (i + 1))
            Accuracy_list.append(100. * correct / total)
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total
            # 将每次测试结果实时写入acc.txt文件中
            if acc > best_acc:
                best_acc = acc
    print("Training Finished, TotalEPOCH=%d" % EPOCH)

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
    if not os.path.exists("./parameterForCifar10"):
        os.mkdir("./parameterForCifar10")
    torch.save(net.state_dict(), './parameterForCifar10/parameter{}.pkl'.format(choice))
    # # 加载
    # model = TheModelClass(...)
    # model.load_state_dict(torch.load('\parameter.pkl'))

    # 作图
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
    plt.text(x=1,y=1,s="acc={}".format(((correct.item() / len(test_loader.dataset)) )))
    plt.savefig("./parameterForCifar10/Cifar10_accuracy_loss{0}.jpg".format(choice))
    plt.show()


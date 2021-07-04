 # 人工智能作业
- [人工智能作业](#人工智能作业)
  - [1. assignment1作业需求描述](#1-assignment1作业需求描述)
  - [2. assignment2作业需求描述](#2-assignment2作业需求描述)
  - [3. assignment3作业需求描述](#3-assignment3作业需求描述)
  - [4. assignment4作业需求描述](#4-assignment4作业需求描述)
  - [5. Requirements](#5-requirements)
  - [6.作者](#6作者)

## 1. assignment1作业需求描述
1. 使用简单全连接网络作为分类器，完成MNIST分类任务
2. 使用深度全连接网络（2个隐藏层以上）作为分类器，完成MNIST分类任务
3. 使用深度全连接网络（2个隐藏层以上）作为分类器，添加dropout，正则化等技巧，完成MNIST分类任务
4. 使用LetNet5，完成MNIST分类任务
5. 重复任务3、4，完成MNIST-fashion分类任务

## 2. assignment2作业需求描述
1. 使用简单CNN作为分类器完成Cifar10分类任务

2. 使用简单CNN作为分类器，增加数据增强、正则化等技巧，完成Cifar10分类任务

3. 使用ResNet作为分类器，完成Cifar10分类任务

4. 使用VGG或ResNet预训练模型，微调后作为分类器，完成Cifar10分类任务

## 3. assignment3作业需求描述
1. 使用原始GAN，DCGAN实现手写数字生成

## 4. assignment4作业需求描述
  1. 在集成电路板（PCB）的生产过程中，需要在PCB表面上焊接不同类型不同数量的贴片电阻，焊点检测（solder joint detection）是PCB（print circuit board）生产中的关键环节，主要目的是从全部焊点图片中识别出焊点异常的图片。但是在实际的焊点检测过程中，焊点检测往往会误检（将合格的焊点检测成不合格的焊点）。此时就需要人工进行二次检测，对第一次的焊点检测中不合格的焊点进行筛选，并从中挑选出合格的焊点，这样会增加一定的人力成本以及焊点检测的时间。为了提高检测的准确率，减少误判率，现需要根据提供的焊点图片数据集，利用机器学习或深度学习算法，完成焊点图片数据集的分类识别任务。异常焊点可以大概分为三种情况：1.缺元器件;2.焊接位置错误；3.焊锡太多或太少导致颜色不对。给出的数据集中的数据为贴片电阻的焊点图片。数据集按照电阻的类型分为三类（101K,102K,331K）。每类又分为正常焊点（normal）和异常焊点（abnormal）。要求对电阻的类型进行分类，同时要求对正常焊点和异常焊点进行分类。需要注意的是，数据集（包括样本集和测试集）中的图片尺寸不完全相同，主要分布在[175,100]和[112,64]这两个尺寸附近。
2. 训练数据在3300，正常图片2800，异常图片480

## 5. Requirements

- **Development Environment:** 

  Win 10 

- **Development Software:**

  **PyCharm** *2020.3.5.PC-191.6605.12*

- **Development Language:**

  ​	Python

 - **Mainly Reference Count:**

  1. torchvision 
  2. matplotlib 
  3. os
  4. torch
  5. numpy


## 6.作者

| ID      | Name                |
| ------- | ------------------- |
| ****** | 吴杨婉婷     |


​	**联系方式**	email: 1852824@tongji.edu.cn


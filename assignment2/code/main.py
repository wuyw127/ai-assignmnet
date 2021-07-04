import train_funcs

if __name__ == '__main__':
    choice = int(input("choose the model\n"
                   "1:简单cnn\n"
                   "2:数据增强\n"
                   "3:正则化\n"
                   "4:resnet\n"
                   "5:resnet微调\n"
                   "6:VGG微调\n:\n"
                   ))
    if choice == 1:
        train_funcs.simple_train_cifar10()
    elif choice == 2:
        train_funcs.image_enhancement()
    elif choice == 3:
        train_funcs.regularization()
    elif choice == 4:
        train_funcs.resnet()
    elif choice == 5:
        train_funcs.resnet_change()
    elif choice == 6:
        train_funcs.VGG16()



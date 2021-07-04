import os
import shutil



def copyImg():
    sourcePath = './sample/0/'
    # 指定图片原始路径A
    targetPath = './sample/0/'
    # 指定图片存放目录B
    for i in range(480):
        shutil.copy(sourcePath + '{}.png'.format(i) , targetPath +'{}.png'.format(i+480) )



if __name__ == '__main__':
    copyImg()

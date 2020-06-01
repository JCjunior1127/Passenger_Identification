# -*- coding: utf-8 -*-
'''
 @Time  :  2020/5/30 1:41
 @Author:  wu xia
 @File  :  split_train_val_test.py
@2：划分测试集、验证集、训练集
'''

import os
import random


def _main():
    xml_filepath = './datasets/annotations'  # 相对目录：工程文件所在目录
    txt_savepath = './datasets/imagesets'  # 相对目录：工程文件所在目录

    ftest = open('./datasets/imagesets/main/test.txt', 'w')  # 划分测试集
    ftrain = open('./datasets/imagesets/main/train.txt', 'w')  # 划分训练集---->最后仅 train.txt里面有数据，为什么？
    ftrainval = open('./datasets/imagesets/main/trainval.txt', 'w')  # ？
    fval = open('./datasets/imagesets/main/val.txt', 'w')  # 划分验证集

    test_percent = 0  # 测试集比例？
    train_percent = 1  # 训练集比例？
    trainval_percent = 0  #？
    val_percent = 0  # 验证集比例？

    total_xml = os.listdir(xml_filepath)
    num = len(total_xml)  # 标注文件数量
    list = range(num)
    tv = int(num * trainval_percent)  # trainval比例?
    tr = int(tv * train_percent)  # 训练集比例
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    for i in list:
        name = total_xml[i][:-4] + '\n'  # 获取第i个xml文件的文件名，不包括最后“,jpg”后缀
        if i in trainval:  # 从trainval里划分测试集和验证集
            ftrainval.write(name)
            if i in train:
                ftest.write(name)
            else:
                fval.write(name)
        else:  # 训练集
            ftrain.write(name)

    ftest.close()
    ftrain.close()
    ftrainval.close()
    fval.close()


if __name__ == "__main__":
    _main()
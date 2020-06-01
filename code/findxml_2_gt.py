# -*- coding: utf-8 -*-
'''
 @Time  :  2020/6/1 1:46
 @Author:  wu xia
 @File  :  findxml_2_gt.py
 从datasets/annotations文件夹中找到所有图片对应的xml文件，然后粘到ground_truth文件夹
'''

import os
import shutil

testfilepath = './test_images'  # 根据测试图片名字从txt_annotations里面将xml文件拷贝到ground_truth
xmlfilepath = './datasets/annotations/'
xmlsavepath = './model_AP/ground_truth/'
test_jpg = os.listdir(testfilepath)

num = len(test_jpg)
list = range(num)
L = []

for i in list:
    name = test_jpg[i][:-4] + '.xml'
    L.append(name)

for filename in L:
    shutil.copy(os.path.join(xmlfilepath, filename), xmlsavepath)


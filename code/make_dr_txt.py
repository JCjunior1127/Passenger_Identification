# -*- coding: utf-8 -*-
'''
 @Time  :  2020/6/1 2:09
 @Author:  wu xia
 @File  :  make_dr_txt.py
 将test_results文件夹下的test_results.txt文件转为每张图片对应的一个个独立的txt,并保存到model_AP/detection_results/
'''

multi_txt_path = './model_AP/detection_results/'
single_txt_path = './test_results/test_result.txt'
f = open(single_txt_path, encoding='utf8')
s = f.readlines()

for i in range(len(s)):  # 中按行存放的检测内容，为列表的形式
    r = s[i].split('.jpg ')
    file = open(multi_txt_path + r[0] + '.txt', 'w')
    if len(r[1]) > 5:
        t = r[1].split(';')
        # print('len(t):',len(t))
        if len(t) == 3:
            file.write(t[0] + '\n' + t[1] + '\n')  # 有两个对象被检测出
        elif len(t) == 4:
            file.write(t[0] + '\n' + t[1] + '\n' + t[2] + '\n')  # 有三个对象被检测出
        elif len(t) == 5:
            file.write(t[0] + '\n' + t[1] + '\n' + t[2] + '\n' + t[3] + '\n')  # 有四个对象被检测出
        elif len(t) == 6:
            file.write(t[0] + '\n' + t[1] + '\n' + t[2] + '\n' + t[3] + '\n' + t[4] + '\n')  # 有五个对象被检测出
        elif len(t) == 7:
            file.write(t[0] + '\n' + t[1] + '\n' + t[2] + '\n' + t[3] + '\n' + t[4] + '\n' + t[5] + '\n')  # 有六个对象被检测出

        else:
            file.write(t[0] + '\n')  # 有一个对象
    else:
        file.write('')  # 没有检测出来对象，创建一个空白的对象

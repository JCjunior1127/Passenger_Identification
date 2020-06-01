# -*- coding: utf-8 -*-
'''
 @Time  :  2020/5/30 1:56
 @Author:  wu xia
 @File  :  annotation_xml2txt.py
@3：将标注文件.xml转化为.txt格式
'''

import xml.etree.ElementTree as ET  # Python 下可用的 XML 处理工具
from os import getcwd


# 转换标注文件xml格式为txt格式(yolo)
def convert_annotation_xml2txt(year, image_id, txt_file, classes):
    '''
    type: =station
    image_id:图像名字，不包括后缀名
    list_file: 要保存的txt文件
    '''

    wd = getcwd()  # os.getcwd()返回当前工作目录(当前工程文件所在目录)
    xml_file = open('%s/datasets/annotations/%s.xml' % (wd, image_id), encoding='UTF-8')  # 标注文件输入路径
    tree = ET.parse(xml_file)  # 加载并且解析这个 XML。 ElementTree 将整个 XML 解析为一棵树，Element 将单个结点解析为树
    root = tree.getroot()  # 抓根结点元素，root 是一个 Element 元素

    # Element 对象有一个 iter 方法可以对子结点进行深度优先遍历。 ElementTree 对象也有 iter 方法
    for obj in root.iter('object'):  # iter 方法接受一个标签名字，然后只遍历那些有指定标签的元素
        difficult = obj.find('difficult').text  # 进入一个指定的子结点
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        b = (int(xml_box.find('xmin').text),
             int(xml_box.find('ymin').text),
             int(xml_box.find('xmax').text),
             int(xml_box.find('ymax').text))
        txt_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def _main():
    # 测试集、验证集、训练集数据划分文件
    sets = [('station', 'train'), ('station', 'val'), ('station', 'test')]  # 关联集合，单位pair
    classes = ["person"]  # 要检测的类别：仅检测person

    wd = getcwd()  # os.getcwd()返回当前工作目录(当前工程文件所在目录)
    for type, image_set in sets:  # sets：测试集、验证集、训练集数据划分文件
        image_ids = open('%s/datasets/imagesets/main/%s.txt' % (wd, image_set), encoding='UTF-8').read().strip().split()  # 遍历训练集下所有txt文件
        txt_file = open('%s/txt_annotations/%s_%s.txt' % (wd, type, image_set), 'w')  # type=station. txt文件保存目录：保存到当前工程目录所在目录，即是voc_annotation.py所在目录

        # 循环处理每个标注文件
        for image_id in image_ids:
            txt_file.write('%s/datasets/images/%s.jpg' % (wd, image_id))  # wds是当前工作目录
            convert_annotation_xml2txt(type, image_id, txt_file, classes)
            txt_file.write('\n')
        txt_file.close()  # f.close() 缩进错误h会导致 ValueError: I/O operation on closed file


if __name__ == "__main__":
    _main()

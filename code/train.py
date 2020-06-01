# -*- coding: utf-8 -*-
'''
 @Time  :  2020/5/29 21:38
 @Author:  wu xia
 @File  :  train.py
 @5：设置验证集与训练集的比例，batchsize大小，epoch数量
'''


import numpy as np
import os
from keras import backend
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from LossHistory import LossHistory

from yolov3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss, mobile_body
from yolov3.utils import get_random_data


def gpu_select(gpu_count):  # 自动选择gpu
    """
    Select the top k free memory GPU for working

    Args:
        gpu_count: int - number of gpu for selecting
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print("GPUs free memory: {}".format(memory_gpu))
    gpu = [(item, i) for i, item in enumerate(memory_gpu)]
    gpu.sort(key=lambda x: x[0])
    gpu_selected = ([str(item[1]) for item in gpu])[-gpu_count:]
    print("GPUs : {} has been selected.".format(gpu_selected))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_selected)
    os.system('rm tmp')
    return True


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, yolo_weights_path, load_pretrained=False, freeze_body=False):  # 默认的函数参数不用管
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    print("num_anchors=", num_anchors)
    print("num_classes", num_classes)

    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for l in range(3)]
    model_body = mobile_body(image_input, num_anchors // 3, num_classes)
    print('Create yolov3 model with {} anchors and {} classes'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(yolo_weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}...'.format(yolo_weights_path))
        if freeze_body:
            # Do not freeze 3 output layers(不冻结3个输出层)
            num = len(model_body.layers) - 7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # 模型损失
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)  # 图片数量
    if n == 0 or batch_size <= 0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


def train(model, annotation_path, input_shape, anchors, num_classes, batch_size, val_split, train_epochs, log_dir='logs/'):
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)  # 随机保存标注数据

    num_val = int(len(lines) * val_split)  # num_val:验证集图片数量 = 3012*0.1=301.2
    num_train = len(lines) - num_val  # num_train:训练集图片数量 = 总图片数量-验证集数量
    print('Train on {} samples, val on {} samples, with batchsize {}.'.format(num_train, num_val, batch_size))

    # lr=0.001 (学习率:learn_rate)
    model.compile(optimizer='adam', loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    # TensorBoard
    logging = TensorBoard(log_dir=log_dir)  # 记录数据并保存数据    checkpoint：检验点
    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5", monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    model.fit_generator(data_generator_wrap(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrap(lines[num_train:], batch_size, input_shape, anchors,num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=train_epochs,
                        initial_epoch=0,
                        callbacks=[logs_loss])
    model.save_weights(log_dir + 'trained_weights.h5')  # 模型权重



'''
因为容易出现这样的场景（常见）：当你在进行keras的交叉验证时，例如你用5折，
对于fold_0,fold_1…一直到fold_4.，都应该有一个独立的模型，所以在每折的开头都需要加上clear_session()，
否则上一折的训练集成了这一折的验证集，会导致数据泄露。
同时，链接提到，不清空的话，那么graph上的node越来越多，内存问题，时间问题都会变得严峻。
'''
backend.clear_session()  # 清除一个session，session就是tensorflow中我们常见的会话
logs_loss = LossHistory()  # 全局对象,引用另外一个文件里的类
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
# gpu_select(1)


def _main():
    backend.clear_session()

    log_dir = './logs/loss_weights_result/'  # 模型训练后损失和权重的保存目录

    annotation_path = './txt_annotations/station_train.txt'  # 数据集的txt格式标注文件路径
    classes_path = './model_data/detection_classes.txt'  # 要检测的对象的名字
    anchors_path = './model_data/yolo_anchors.txt'  # 聚类结果路径
    yolo_weights_path = './model_data/yolo_weights.h5'  # yolo预训练模型权重路径

    input_shape = (416, 416)  # multiple of 32, h-w  输入图片尺寸
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    # 创建模型
    model = create_model(input_shape, anchors, len(class_names), yolo_weights_path)

    batch_size = 16  # 模型计算一次损失要处理的图片的数量
    val_split = 0.1  # 验证集中包含的图像数量/所有图像数量 = 0.1，即验证集所占比例
    train_epochs = 2  # 将所有数据集处理一遍称为一个epoch，故epoch数表示迭代次数
    train(model, annotation_path, input_shape, anchors, len(class_names), batch_size, val_split, train_epochs, log_dir=log_dir)

if __name__ == '__main__':
    _main()
    logs_loss.end_draw()
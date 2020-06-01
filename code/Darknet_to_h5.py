# -*- coding: utf-8 -*-
'''
 @Time  :  2020/5/29 17:41
 @Author:  wu xia
 @File  :  Darknet_to_h5.py
 Reads Darknet config and weights and creates Keras model with TF backend.
 @1：将tensorflow格式的yolov3权重转换为keras需要的格式
'''
import os
import keras
import argparse
import configparser
import io
from collections import defaultdict
import numpy as np
from keras import backend as K
from keras.layers import (Conv2D, Input, ZeroPadding2D, Add, UpSampling2D, MaxPooling2D, Concatenate)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot
from os import getcwd


os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 在所有访问GPU的代码之前
# GpuNum = keras.cuda.device_count()

# ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息
parser = argparse.ArgumentParser(description='Convert Darknet To Keras')  # 在参数帮助文档之前显示的文本（默认值：无）
parser.add_argument('config_path', help='Path to Darknet cfg file')  # 读取路径： ./yolov3_weights/yolov3.weights  添加参数
parser.add_argument('weights_path', help='Path to Darknet weights file')  # 读取路径： ./yolov3_cfg/yolov3.cfg
parser.add_argument('output_path', help='Path to output Keras model file')  # 保存路径：./model_data/yolo.h5
parser.add_argument('-p', '--plot_model', help='Plot generated Keras model and save as image', action='store_true')
parser.add_argument('-w', '--weights_only', help='Save as Keras weights file instead of model file', action='store_true')

def unique_config_sections(config_file):
    '''将所有配置节转换为具有唯一名称，向配置节添加唯一后缀,提高配置解析器的可兼容性'''
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
        output_stream.seek(0)
        return output_stream

def _main(args):
    config_path = os.path.expanduser(args.config_path)
    weights_path = os.path.expanduser(args.weights_path)
    output_path = os.path.expanduser(args.output_path)  # keras模型及其权重的保存路径

    # wd = getcwd()  # os.getcwd()返回当前工作目录(当前工程文件所在目录)
    # config_path = "%s/yolov3_cfg/yolov3.cfg" % (wd)
    # weights_path = "%s/yolov3_weights/yolov3.weights" % (wd)
    # output_path = "%s/model" % (wd)  # keras模型及其权重的保存路径

    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weights_path)
    assert output_path.endswith('.h5'), 'output path {} is not a .h5 file'.format(output_path)

    output_root = os.path.splitext(output_path)[0]  # keras模型结构图的保存路径  assert 宏的原型定义在 assert.h 中，其作用是如果它的条件返回错误，则终止程序执行

    # 加载weights和config
    print('Loading weights...')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(shape=(3,), dtype='int32', buffer=weights_file.read(12))
    if (major * 10 + minor)>= 2 and major < 1000 and minor < 1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config...')  # 解析Darknet配置
    cfg_parser = configparser.ConfigParser()
    unique_config_file = unique_config_sections(config_path)
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras model...')
    input_layer = Input(shape=(None, None, 3))  # 输入层
    prev_layer = input_layer  # ？
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4  # 权重衰变

    count = 0
    out_index = []
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))  # 解析section
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])  # 滤波器
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])  # 步
            pad = int(cfg_parser[section]['pad'])  # 填充
            activation = cfg_parser[section]['activation']  # 激活
            batch_normalize = 'batch_normalize' in cfg_parser[section]  # ？
            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # 设置权重
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]

            prev_layer_shape = K.int_shape(prev_layer)
            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)

            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)
            conv_bias = np.ndarray(shape=(filters,), dtype='float32', buffer=weights_file.read(filters * 4))
            count += filters

            # 1.batch_normalize层
            if batch_normalize:
                bn_weights = np.ndarray(shape=(3, filters), dtype='float32', buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta？
                    bn_weights[1],  # running mean（滑动平均）
                    bn_weights[2]  # running var
                ]

                conv_weights = np.ndarray(shape=darknet_w_shape, dtype='float32', buffer=weights_file.read(weights_size * 4))
                count += weights_size

                # DarkNet conv_weights are serialized Caffe-style:(out_dim, in_dim, height, width)
                # We would like to set these to Tensorflow order:(height, width, in_dim, out_dim)
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                conv_weights = [conv_weights] if batch_normalize else [conv_weights, conv_bias]

                # Handle activation（激活）
                act_fn = None
                if activation == 'leaky':
                    pass  # Add advanced activation later
                elif activation != 'linear':
                    raise ValueError('Unknown activation function `{}` in section {}'.format(activation, section))

                # Create Conv2D layer
                if stride > 1:
                    # Darknet uses left and top padding instead of 'same' mode
                    prev_layer = ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)

                conv_layer = (Conv2D(filters, (size, size),
                    strides=(stride, stride),
                    kernel_regularizer=l2(weight_decay),
                    use_bias=not batch_normalize,
                    weights=conv_weights,
                    activation=act_fn,
                    padding=padding))(prev_layer)

                if batch_normalize:
                    conv_layer = (BatchNormalization(weights=bn_weight_list))(conv_layer)
                prev_layer = conv_layer

                if activation == 'linear':
                    all_layers.append(prev_layer)
                elif activation == 'leaky':
                    act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                    prev_layer = act_layer
                    all_layers.append(act_layer)

            # 2.路由层
            elif section.startswith('route'):
                ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
                layers = [all_layers[i] for i in ids]
                if len(layers) > 1:
                    print('Concatenating route layers...', layers)  # 连接路由层
                    concatenate_layer = Concatenate()(layers)
                    all_layers.append(concatenate_layer)
                    prev_layer = concatenate_layer
                else:
                    skip_layer = layers[0]  # only one layer to route(只有一层要走)
                    all_layers.append(skip_layer)
                    prev_layer = skip_layer

            # 3.最大池化层
            elif section.startswith('maxpool'):
                size = int(cfg_parser[section]['size'])
                stride = int(cfg_parser[section]['stride'])
                all_layers.append(MaxPooling2D(
                        pool_size=(size, size),
                        strides=(stride, stride),
                        padding='same')(prev_layer))
                prev_layer = all_layers[-1]

            # 4.
            elif section.startswith('shortcut'):
                index = int(cfg_parser[section]['from'])
                activation = cfg_parser[section]['activation']
                assert activation == 'linear', 'Only  support linear activation'
                all_layers.append(Add()([all_layers[index], prev_layer]))
                prev_layer = all_layers[-1]

            # yolo层
            elif section.startswith('yolo'):
                out_index.append(len(all_layers) - 1)
                all_layers.append(None)
                prev_layer = all_layers[-1]

            # net层
            elif section.startswith('net'):
                pass

            else:
                raise ValueError('Unsupported section header type: {}'.format(section))

        # Create and save model
        if len(out_index) == 0: out_index.append(len(all_layers) - 1)
        model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])

        print(model.summary())  #model.summary() 打印出模型概述信息,是 utils.print_summary 的简捷调用.
        if args.weights_only:
            model.save_weights('{}'.format(output_path))
            print('Saved Keras weights to {}'.format(output_path))
        else:
            model.save('{}'.format(output_path))
            print('Saved Keras model to {}'.format(output_path))

            # Check to see if all weights have been read
            remaining_weights = len(weights_file.read()) / 4
            weights_file.close()
            print('Read {} of {} from Darknet weights.'.format(count, count + remaining_weights))

            if remaining_weights > 0:
                print('Warning: {} unused weights'.format(remaining_weights))

            if args.plot_model:
                plot(model, to_file='{}.png'.format(output_root), show_shapes=True)
                print('Saved model plot to {}.png'.format(output_root))  # keras模型结构图的保存路径


if __name__ == '__main__':
    _main(parser.parse_args())
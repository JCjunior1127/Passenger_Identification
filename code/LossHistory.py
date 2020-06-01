# -*- coding: utf-8 -*-
'''
 @Time  :  2020/5/29 22:27
 @Author:  wu xia
 @File  :  LossHistory.py
 实时绘制模型loss和accuracy的曲线
'''


import keras
import matplotlib.pyplot as plt
import time

fig_save_path = './fig_save/'  # 模型训练集损失值和验证集准确率曲线图的保存路径
class LossHistory(keras.callbacks.Callback):  # 计算模型的loss与accuracy


    def on_train_begin(self, logs={}):  # 创建盛放loss与accuracy的容器
        self.losses = {'batch': [], 'epoch': []}  # loss
        self.val_loss = {'batch': [], 'epoch': []}  # ccuracy


    def draw_loss_50(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 50))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)

        plt.legend(loc="upper right")
        plt.savefig(fig_save_path + type + '_' + label + '.jpg')
        plt.close()

    def draw_loss_100(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 100))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path + type + '_' + label + '.jpg')
        plt.close()

    def draw_loss_200(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 200))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path + type + '_' + label + '.jpg')
        plt.close()

    def draw_loss_500(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 500))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path + type + '_' + label + '.jpg')
        plt.close()

    def draw_loss_1000(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 1000))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path + type + '_' + label + '.jpg')
        plt.close()

    # 绘图：把每一种曲线都单独绘图（保存损失曲线图），若想把各种曲线绘制在一张图上的话可修改此法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(fig_save_path + type + '_' + label + '.jpg')
        plt.close()

    def on_batch_end(self, batch, logs={}):  # 按照batch来追加数据，一个batch指计算一次损失处理多少张图片
        # 每个batch结束后追加一个值
        self.losses['batch'].append(logs.get('loss'))  # 每一个batch完成后向容器里面追加loss，accuracy
        self.val_loss['batch'].append(logs.get('val_loss'))

        if int(time.time()) % 3600 == 0:  # 每五秒按照当前容器里的值来绘图 返回当前时间的时间戳（1970纪元后经过的浮点秒数）如：time.time(): 1234892919.655932
            self.draw_loss_50(self.losses['batch'], 'loss', 'train_batch_50')
            self.draw_loss_100(self.losses['batch'], 'loss', 'train_batch_100')
            self.draw_loss_200(self.losses['batch'], 'loss', 'train_batch_200')
            self.draw_loss_500(self.losses['batch'], 'loss', 'train_batch_500')
            self.draw_loss_1000(self.losses['batch'], 'loss', 'train_batch_1000')

            self.draw_p(self.losses['batch'], 'loss', 'train_batch')
            self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))  # 每一个epoch完成后向容器里面追加loss，accuracy
        self.val_loss['epoch'].append(logs.get('val_loss'))

        if int(time.time()) % 3600 == 0:  # 每五秒按照当前容器里的值来绘图
            self.draw_loss_50(self.losses['batch'], 'loss', 'train_batch_50')
            self.draw_loss_100(self.losses['batch'], 'loss', 'train_batch_100')
            self.draw_loss_200(self.losses['batch'], 'loss', 'train_batch_200')
            self.draw_loss_500(self.losses['batch'], 'loss', 'train_batch_500')
            self.draw_loss_1000(self.losses['batch'], 'loss', 'train_batch_1000')

            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')

    # 由于on_batch_end和on_epoch_end的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程
    # 所以最后一次绘图结束，又训练了0-5秒的时间，所以这里的方法会在整个训练结束以后调用
    def end_draw(self):  # 调用了
        self.draw_loss_50(self.losses['batch'], 'loss', 'train_batch_50')
        self.draw_loss_100(self.losses['batch'], 'loss', 'train_batch_100')
        self.draw_loss_200(self.losses['batch'], 'loss', 'train_batch_200')
        self.draw_loss_500(self.losses['batch'], 'loss', 'train_batch_500')
        self.draw_loss_1000(self.losses['batch'], 'loss', 'train_batch_1000')

        self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        #  self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')


'''




import keras
import matplotlib.pyplot as plt
import time


class LossHistory(keras.callbacks.Callback):  # 计算模型的loss与accuracy
    def on_train_begin(self, logs={}):  # 创建盛放loss与accuracy的容器
        self.losses = {'batch': [], 'epoch': []}  # loss
        self.val_loss = {'batch': [], 'epoch': []}  # ccuracy

    def draw_loss_50(self, lists, label, type, fig_save_path):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 50))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)

        plt.legend(loc="upper right")
        plt.savefig(fig_save_path + type + '_' + label + '.jpg')
        plt.close()

    def draw_loss_100(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 100))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')
        plt.close()

    def draw_loss_200(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 200))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')
        plt.close()

    def draw_loss_500(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 500))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')
        plt.close()

    def draw_loss_1000(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylim((0, 1000))  # 损失值的范围
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')
        plt.close()

    # 绘图：把每一种曲线都单独绘图（保存损失曲线图），若想把各种曲线绘制在一张图上的话可修改此法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')
        plt.close()

    def on_batch_end(self, batch, logs={}):  # 按照batch来追加数据，一个batch指计算一次损失处理多少张图片
        # 每个batch结束后追加一个值
        self.losses['batch'].append(logs.get('loss'))  # 每一个batch完成后向容器里面追加loss，accuracy
        self.val_loss['batch'].append(logs.get('val_loss'))

        if int(time.time()) % 3600 == 0:  # 每五秒按照当前容器里的值来绘图 返回当前时间的时间戳（1970纪元后经过的浮点秒数）如：time.time(): 1234892919.655932
            self.draw_loss_50(self.losses['batch'], 'loss', 'train_batch_50')
            self.draw_loss_100(self.losses['batch'], 'loss', 'train_batch_100')
            self.draw_loss_200(self.losses['batch'], 'loss', 'train_batch_200')
            self.draw_loss_500(self.losses['batch'], 'loss', 'train_batch_500')
            self.draw_loss_1000(self.losses['batch'], 'loss', 'train_batch_1000')

            self.draw_p(self.losses['batch'], 'loss', 'train_batch')
            self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))  # 每一个epoch完成后向容器里面追加loss，accuracy
        self.val_loss['epoch'].append(logs.get('val_loss'))

        if int(time.time()) % 3600 == 0:  # 每五秒按照当前容器里的值来绘图
            self.draw_loss_50(self.losses['batch'], 'loss', 'train_batch_50')
            self.draw_loss_100(self.losses['batch'], 'loss', 'train_batch_100')
            self.draw_loss_200(self.losses['batch'], 'loss', 'train_batch_200')
            self.draw_loss_500(self.losses['batch'], 'loss', 'train_batch_500')
            self.draw_loss_1000(self.losses['batch'], 'loss', 'train_batch_1000')

            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')

    # 由于on_batch_end和on_epoch_end的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程
    # 所以最后一次绘图结束，又训练了0-5秒的时间，所以这里的方法会在整个训练结束以后调用
    def end_draw(self):  # 调用了
        self.draw_loss_50(self.losses['batch'], 'loss', 'train_batch_50')
        self.draw_loss_100(self.losses['batch'], 'loss', 'train_batch_100')
        self.draw_loss_200(self.losses['batch'], 'loss', 'train_batch_200')
        self.draw_loss_500(self.losses['batch'], 'loss', 'train_batch_500')
        self.draw_loss_1000(self.losses['batch'], 'loss', 'train_batch_1000')

        self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        #  self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
'''
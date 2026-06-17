from cProfile import label
import sys
import argparse
import datetime
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dsets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset
from model import UNetWrapper, SegmentationAugmentation
from util import enumerateWithEstimate

# 指标索引常量
METRICS_LOSS_NDX = 0
METRICS_TP_NDX = 1
METRICS_FN_NDX = 2
METRICS_FP_NDX = 3
METRICS_SIZE = 4

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)

class LunaSegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        # 解析参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers', help='数据加载线程数', default=4, type=int)
        parser.add_argument('--batch-size', help='批次大小', default=16, type=int)
        parser.add_argument('--epochs', help='训练周期数', default=1, type=int)
        parser.add_argument('--limit', help='限制样本数用于快速调试', default=None, type=int)
        parser.add_argument('--augmented', help='启用全部数据增强', action='store_true', default=False)
        parser.add_argument('--augment-flip', help='翻转增强', action='store_true', default=False)
        parser.add_argument('--augment-offset', help='平移增强', action='store_true', default=False)
        parser.add_argument('--augment-scale', help='缩放增强', action='store_true', default=False)
        parser.add_argument('--augment-rotate', help='旋转增强', action='store_true', default=False)
        parser.add_argument('--augment-noise', help='噪点增强', action='store_true', default=False)
        parser.add_argument('comment', nargs='?', default='', help='运行名称后缀')
        self.cli_args = parser.parse_args(sys_argv)

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        # 整理数据增强参数
        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.1
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        # 设备配置
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 初始化模型与优化器
        self.segmentation_model, self.augmentation_model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.totalTrainingSamples_count = 0

    def initModel(self):
        # 初始化 2D U-Net 模型
        segmentation_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv'
        )

        # 数据增强模型
        if self.augmentation_dict:
            augmentation_model = SegmentationAugmentation(**self.augmentation_dict)
        else:
            augmentation_model = None

        if self.use_cuda:
            log.info(f"使用 CUDA GPU 训练，检测到 {torch.cuda.device_count()} 个设备")
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            if augmentation_model:
                augmentation_model = augmentation_model.to(self.device)
        else:
            log.info("使用 CPU 训练")

        return segmentation_model, augmentation_model


    def initOptimizer(self):
        return Adam(self.segmentation_model.parameters(), lr=0.001)

    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=False,
            limit=self.cli_args.limit,
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        return DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            limit=self.cli_args.limit,
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        return DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

    def diceLoss(self, prediction_g, label_g, epsilon=1.0):
        # 骰子损失计算
        diceLabel_g = label_g.sum(dim=[1,2,3])
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) / (dicePrediction_g + diceLabel_g + epsilon)
        return 1.0 - diceRatio_g

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g, classificationThreshold=0.5):
        input_t, label_t, _series_list, _slice_ndx_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        # 数据增强 (仅在训练模式且有配置时)
        if self.segmentation_model.training and self.augmentation_model:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        label_g = label_g.to(torch.float32)

        # 前向传播
        prediction_g = self.segmentation_model(input_g)

        # 混合损失计算：整体 Dice + 假阴性罚项 (fnLoss * 8)
        diceLoss_g = self.diceLoss(prediction_g, label_g)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        # 记录指标到指标矩阵
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)
        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] >classificationThreshold).to(torch.float32)
            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1.0 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (1.0 - label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = diceLoss_g.detach()
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp.detach()
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn.detach()
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp.detach()

        # 返回加权损失
        return diceLoss_g.mean() + fnLoss_g.mean() * 8.0

    def doTraining(self, epoch_ndx, train_dl):
        self.segmentation_model.train()
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        for batch_ndx, batch_tup in enumerateWithEstimate(
            train_dl,
            f"Epoch {epoch_ndx} 训练中",
            start_ndx=0,
            print_ndx=4,
        ):
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()
            self.totalTrainingSamples_count += len(batch_tup[0])

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        self.segmentation_model.eval()
        valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

        with torch.no_grad():
            for batch_ndx, batch_tup in enumerateWithEstimate(
                val_dl,
                f"Epoch {epoch_ndx} 验证中",
                start_ndx=0,
                print_ndx=4,
            ):
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        # 汇总各项指标
        loss_mean = metrics_t[METRICS_LOSS_NDX].mean().item()
        tp_sum = metrics_t[METRICS_TP_NDX].sum().item()
        fn_sum = metrics_t[METRICS_FN_NDX].sum().item()
        fp_sum = metrics_t[METRICS_FP_NDX].sum().item()

        # 计算精度、召回率、F1分数
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        log.info(
            f"Epoch {epoch_ndx} {mode_str:<5} | Loss: {loss_mean:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1_score:.4f} "
            f"(TP: {tp_sum:.0f}, FN: {fn_sum:.0f}, FP: {fp_sum:.0f})"
        )

        return f1_score

    def main(self):
        log.info(f"开始训练分割任务，配置: {self.cli_args}")
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info(f"Epoch {epoch_ndx}/{self.cli_args.epochs} 训练开始...")
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            log.info(f"Epoch {epoch_ndx}/{self.cli_args.epochs} 验证开始...")
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        log.info("训练完成！")

if __name__ == '__main__':
    LunaSegmentationTrainingApp().main()

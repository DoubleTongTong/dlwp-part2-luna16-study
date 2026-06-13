import os
import sys
import argparse
import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dsets import LunaDataset
from model import LunaModel
from util import enumerateWithEstimate

METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        # 解析命令行参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        parser.add_argument('--limit',
            help='Limit the number of samples for quick debugging',
            default=None,
            type=int,
        )
        parser.add_argument('--balanced',
            help="Balance the training data to half positive, half negative.",
            action='store_true',
            default=False,
        )
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        # 硬件加速检查与配置
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 初始化模型与优化器
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                # 若有多张显卡，则进行并行配置包装
                model = nn.DataParallel(model)
            model = model.to(self.device)
        else:
            log.info("Using CPU.")
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=int(self.cli_args.balanced),
            limit=self.cli_args.limit
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

        return train_dl

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
            limit=self.cli_args.limit
        )
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )

        return val_dl

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        if self.cli_args.limit:
            log.info("Debugging mode enabled: Dataset size limited to {}".format(self.cli_args.limit))

        log.info("Successfully initialized training DataLoader with {} samples ({} batches)".format(
            len(train_dl.dataset), len(train_dl)))
        log.info("Successfully initialized validation DataLoader with {} samples ({} batches)".format(
            len(val_dl.dataset), len(val_dl)))
        log.info("Setup complete! Model: {}, Optimizer: {}".format(
            type(self.model).__name__, type(self.optimizer).__name__))


        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, training...".format(epoch_ndx, self.cli_args.epochs))
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            log.info("Epoch {} of {}, validating...".format(epoch_ndx, self.cli_args.epochs))
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if self.trn_writer is not None:
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffleSamples()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        for batch_ndx, batch_tup in enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=0,
            print_ndx=16,
            backoff=2,
        ):
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )
            loss_var.backward()
            self.optimizer.step()
            self.totalTrainingSamples_count += len(batch_tup[0])

            # Batch logging
            if batch_ndx >= 16 and (batch_ndx & (batch_ndx - 1)) == 0:
                log.info("Epoch {} Training Batch {}/{} (loss: {:.4f})".format(
                    epoch_ndx,
                    batch_ndx,
                    len(train_dl),
                    loss_var.item()
                ))

        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            for batch_ndx, batch_tup in enumerateWithEstimate(
                val_dl,
                "E{} Validation".format(epoch_ndx),
                start_ndx=0,
                print_ndx=16,
                backoff=2,
            ):
                loss_var = self.computeBatchLoss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    valMetrics_g,
                )

                # Batch logging
                if batch_ndx >= 16 and (batch_ndx & (batch_ndx - 1)) == 0:
                    log.info("Epoch {} Validation Batch {}/{} (loss: {:.4f})".format(
                        epoch_ndx,
                        batch_ndx,
                        len(val_dl),
                        loss_var.item()
                    ))

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:, 1])

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.time_str)
            self.trn_writer = SummaryWriter(log_dir=log_dir + '-trn_cls')
            self.val_writer = SummaryWriter(log_dir=log_dir + '-val_cls')

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t,
        classificationThreshold=0.5
    ):
        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold
        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        trueNeg_count = neg_correct
        truePos_count = pos_correct
        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean() if neg_count > 0 else 0.0
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean() if pos_count > 0 else 0.0

        metrics_dict['correct/all'] = (pos_correct + neg_correct) \
            / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100 if neg_count > 0 else 0.0
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100 if pos_count > 0 else 0.0

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float32(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = \
            truePos_count / np.float32(truePos_count + falseNeg_count)
        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        self.initTensorboardWriters()
        writer = self.trn_writer if mode_str == 'trn' else self.val_writer
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             + "{pr/precision:.4f} precision, "
             + "{pr/recall:.4f} recall, "
             + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

if __name__ == '__main__':
    LunaTrainingApp().main()

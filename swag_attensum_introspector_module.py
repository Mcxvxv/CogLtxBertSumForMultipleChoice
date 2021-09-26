import collections
import os
import json
import logging
from argparse import ArgumentParser
import random
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from transformers import BertTokenizer

from memreplay import _score_blocks
from optimization import WarmupLinearLR
from models import SwagIntrospector, BertSumIntrospector, AttenSumIntrospector
from buffer import buffer_collate_for_intro, buffer_collate_for_bertsumintro, buffer_collate_for_attensumintro


class AttenSumIntrospectorModule(pl.LightningModule):
    def __init__(self, config):
        super(AttenSumIntrospectorModule, self).__init__()
        self.config = config
        self.save_hyperparameters()
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        self.introspector = AttenSumIntrospector.from_pretrained(config.model_name, intro_config=config)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.introspector.parameters(),
            lr=self.config.lr1,
            weight_decay=self.config.weight_decay1
        )
        scheduler = WarmupLinearLR(optimizer, self.trainer.max_epochs * (
                    len(self.train_dataset) // self.config.intro_accumulate_grad_batches // self.config.batch_size_intro_per_gpu))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def set_dataset(self, dataset, mode='train'):
        if mode == 'train':
            self.train_dataset = dataset
        elif mode == 'val':
            self.val_dataset = dataset
        elif mode == 'test':
            self.test_dataset = dataset
        else:
            raise ValueError('No such dataset')

    def train_dataloader(self):
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = DistributedSampler(self.train_dataset)
        # train_sampler = RandomSampler(self.train_dataset)
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size_intro_per_gpu,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            collate_fn=buffer_collate_for_attensumintro,
            persistent_workers=True
        )
        logging.info('train_dataset reloaded in Introspector.')
        return loader

    def on_train_epoch_start(self):
        self.cuda_device = next(self.introspector.parameters()).device
        self._file = open(os.path.join(self.config.tmp_dir, 'estimations_{}.txt'.format(self.cuda_device)), 'w')

    def on_train_epoch_end(self, **kwargs):
        self._file.close()

    def training_step(self, batch, batch_idx):
        inputs, labels, pbufs, question_lens, chioce_lens, passage_lens, clses, mask_cls = batch
        loss_introspector, logits = self.introspector(*inputs, labels, question_lens, chioce_lens, passage_lens, clses, mask_cls)
        for i, buf in enumerate(pbufs):
            self._write_estimation(buf, logits[i])
        self.log("loss", loss_introspector)
        return {'loss': loss_introspector}

    def _write_estimation(self, buf, relevance_blk):
        for i, blk in enumerate(buf):
            self._file.write(f'{blk.pos} {relevance_blk[i].item()}\n')

    @staticmethod
    def add_specific_args(parser):
        # encoder
        parser.add_argument("--ff_size", default=512, type=int, help='encoder feedforward size')
        parser.add_argument("--heads", default=4, type=int, help='encoder num_attention_heads')
        parser.add_argument("--inter_layers", default=2, type=int, help='encoder num_transformer_encoder_layer')
        parser.add_argument("--dropout", default=0.1, type=float, help='encoder dropout')

        parser.add_argument('--lr1', type=float, default=4e-5, help='learning rate of introspector')
        parser.add_argument('--weight_decay1', type=float, default=1e-4, help='weight decay of introspector')
        parser.add_argument('--batch_size_intro_per_gpu', type=int, default=12, help='intro batch_size')
        parser.add_argument('--intro_accumulate_grad_batches', type=int, default=4,
                            help='num of intro accumulate_grad_batches')

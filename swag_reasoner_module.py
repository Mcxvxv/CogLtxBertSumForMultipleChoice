import os
import json
import logging
from argparse import ArgumentParser
import random
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel, BertForMultipleChoice, get_cosine_schedule_with_warmup

from data_helper import AverageMeter
from optimization import WarmupLinearLR
from models import *
from utils import CAPACITY, ForkedPdb
from buffer import buffer_collate_for_reason


class SwagReasonerModule(pl.LightningModule):
    def __init__(self, config):
        super(SwagReasonerModule, self).__init__()
        self.config = config
        self.save_hyperparameters()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.reasoner = BertForMultipleChoice.from_pretrained(config.model_name)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

        self.train_accs = AverageMeter()

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.reasoner.parameters(),
            lr=self.config.lr2,
            weight_decay=self.config.weight_decay2
        )
        scheduler = WarmupLinearLR(optimizer, self.trainer.max_epochs * (
                len(self.train_dataset) // self.config.reason_accumulate_grad_batches // self.config.batch_size_reason_per_gpu))  # return [optimizer], [scheduler]
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

    # @pl.LightningDataModule.train_dataloader
    def train_dataloader(self):
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = DistributedSampler(self.train_dataset)
        # train_sampler = RandomSampler(self.train_dataset)
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size_reason_per_gpu,
            shuffle=False,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            collate_fn=buffer_collate_for_reason,
            persistent_workers=True
        )
        logging.info('train_dataset reloaded in Reasoner.')
        return loader

    def on_train_epoch_start(self):
        self.train_accs.reset()
        self.cuda_device = next(self.reasoner.parameters()).device
        self.change_file = open(os.path.join(self.config.tmp_dir, 'changes_{}.txt'.format(self.cuda_device)), 'w')
        self.compare_file = open(os.path.join(self.config.tmp_dir, 'compare_{}.txt'.format(self.cuda_device)), 'a+')
        self.rel_file = open(os.path.join(self.config.tmp_dir, 'rel_{}.txt'.format(self.cuda_device)), 'a+')

    def on_train_epoch_end(self, **kwargs):
        self.change_file.close()
        self.compare_file.write("\n")
        self.compare_file.close()
        self.rel_file.write("\n")
        self.rel_file.close()

    def training_step(self, batch, batch_idx):
        inputs, labels, pbufs, question_lens_tensor, choice_lens_tensor, passage_lens_tensor = batch
        for pbuf in pbufs:
            res = {
                "pos": [],
                "rel": []
            }
            for blk in pbuf:
                res["pos"].append(blk.pos)
                res["rel"].append(blk.relevance)
            self.rel_file.write(json.dumps(res, ensure_ascii=False) + '\n')

        logits = self.reasoner(*inputs).logits
        result = self.criterion(logits, labels)
        train_acc = torch.eq(logits.argmax(1), labels).sum().float().item() / labels.size(0)
        self.train_accs.update(train_acc, labels.size(0))

        loss_reasoner = result.mean()

        # Label the relevance by the current reasoner
        if self.config.latent:
            self._intervention(batch, result)
        self.log_dict({'loss': loss_reasoner, 'acc': self.train_accs.avg}, prog_bar=True)
        return {'loss': loss_reasoner}

    def _write_changes(self, blk, key, value):
        # print("write change {} {} {}".format(blk.pos, key, value))
        self.change_file.write('{} {} {}\n'.format(blk.pos, key, value))

    def _intervention(self, batch, result):
        loss_reasoner = result.detach()
        with torch.no_grad():
            max_bs = self.config.batch_size_reason_per_gpu * 4
            inputs, labels, pbufs, question_lens_tensor, choice_lens_tensor, passage_lens_tensor = batch
            input_ids, attention_mask, token_type_ids = inputs
            batch_size = input_ids.size(0)
            for b_idx in range(batch_size):
                # (4, 512)
                ids, attn_mask, type_ids = input_ids[b_idx], attention_mask[b_idx], token_type_ids[b_idx]
                pbuf, question_len, choice_len = pbufs[b_idx], question_lens_tensor[b_idx], choice_lens_tensor[b_idx]
                assert len(pbuf) > 1
                bs = len(pbuf)
                # Make inputs by expand with different attention masks
                ids = ids.view(1, 4, -1).repeat(bs, 1, 1)
                type_ids = type_ids.view(1, 4, -1).repeat(bs, 1, 1)
                attn_mask = attn_mask.view(1, 4, -1).repeat(bs, 1, 1)
                label = labels[b_idx].view(1, -1).repeat(bs, 1)

                t = 0
                dblk_length = 0
                for blk_idx, blk in enumerate(pbuf.blocks):
                    tmp_blk_length = len(blk) - 2
                    for i in range(len(question_len)):
                        qc_length = question_len[i] + choice_len[i]
                        dblk_start = qc_length + dblk_length
                        dblk_end = dblk_start + tmp_blk_length
                        attn_mask[t, i, dblk_start:dblk_end].zero_()
                    dblk_length += tmp_blk_length
                    t += 1
                assert t == bs

                losses = []
                for j in range((bs - 1) // max_bs + 1):
                    l, r = max_bs * j, min(bs, max_bs * (j + 1))
                    # print(attn_masks[l:r])
                    logits = self.reasoner(ids[l:r], attn_mask[l:r], type_ids[l:r]).logits
                    result = self.criterion(logits, label[l:r].view(-1))
                    losses.append(result)
                losses_delta = torch.cat(losses, dim=0) - loss_reasoner[b_idx]

                relevances = []
                # Label relevance
                t = 0
                for blk in pbuf.blocks:
                    relevances.append(blk.relevance)
                    if losses_delta[t] >= self.config.levelup_threshold and blk.relevance < 2:  # TODO topk
                        self._write_changes(blk, 'relevance', blk.relevance + 1)
                    elif losses_delta[t] <= self.config.leveldown_threshold and blk.relevance > -1:
                        self._write_changes(blk, 'relevance', blk.relevance - 1)
                    t += 1
                w_data = {
                    "losses": torch.cat(losses, dim=0).tolist(),
                    "loss_reasoner": loss_reasoner[b_idx].tolist(),
                    "losses_delta": losses_delta.tolist(),
                    "relevances": relevances
                }
                self.compare_file.write(json.dumps(w_data, ensure_ascii=False) + '\n')
                assert t == bs

    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--lr2', type=float, default=1e-4, help='learning rate of reasoner')
        parser.add_argument('--weight_decay2', type=float, default=1e-4, help='weight decay of reasoner')
        parser.add_argument('--levelup_threshold', type=float, default=0.2, help='levelup_threshold')
        parser.add_argument('--leveldown_threshold', type=float, default=-0.05, help='leveldown_threshold')
        parser.add_argument('--batch_size_reason_per_gpu', type=int, default=3, help='reasoner batch_size')
        parser.add_argument('--reason_accumulate_grad_batches', type=int, default=16,
                            help='num of reasoner accumulate_grad_batches')

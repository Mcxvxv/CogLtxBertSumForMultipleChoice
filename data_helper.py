import copy
import json
import pickle
import os
import re
import logging
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from buffer import Buffer
from utils import CAPACITY, BLOCK_SIZE


class SimpleListDataset(Dataset):
    def __init__(self, source):
        if isinstance(source, str):
            with open(source, 'rb') as fin:
                logging.info('Loading dataset...')
                self.dataset = pickle.load(fin)[0:]
        elif isinstance(source, list):
            self.dataset = source
        if not isinstance(self.dataset, list):
            raise ValueError('The source of SimpleListDataset is not a list.')

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class SwagBlkPosInterface:
    def __init__(self, dataset):
        assert isinstance(dataset, SimpleListDataset)
        self.d = {}
        self.dataset = dataset
        for data in dataset:
            qbuf, cbuf_list, dbuf, label = data
            for blk in qbuf:
                if blk.pos not in self.d:
                    self.d[blk.pos] = blk
            for cbuf in cbuf_list:
                for blk in cbuf:
                    if blk.pos not in self.d:
                        self.d[blk.pos] = blk
            for blk in dbuf:
                if blk.pos not in self.d:
                    self.d[blk.pos] = blk

    def set_property(self, pos, key, value=None):
        blk = self.d[pos]
        if value is not None:
            setattr(blk, key, value)
        elif hasattr(blk, key):
            delattr(blk, key)

    def apply_changes_from_file(self, filename):
        with open(filename, 'r') as fin:
            for line in fin:
                tmp = [
                    int(s) if s.isdigit() or s[0] == '-' and s[1:].isdigit() else s
                    for s in line.split()
                ]
                self.set_property(*tmp)

    def apply_changes_from_dir(self, tmp_dir):
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('changes_'):
                self.apply_changes_from_file(filename)
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def collect_estimations_from_dir(self, tmp_dir):
        ret = []
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('estimations_'):
                logging.info("collect_estimations_from {}".format(shortname))
                with open(filename, 'r') as fin:
                    for line in fin:
                        l = line.split()
                        pos, estimation = int(l[0]), float(l[1])
                        self.d[pos].estimation = estimation
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def build_swag_random_buffer(self, num_samples):
        n0, n1 = [int(s) for s in num_samples.split(',')][:2]
        ret = []
        # 最多block数
        max_blk_num = CAPACITY // (BLOCK_SIZE + 2)
        logging.info('building swag random buffers for introspection...')
        for qbuf, cbuf_list, dbuf, label in tqdm(self.dataset):
            # 1. continous
            lb = max_blk_num
            st = random.randint(0, max(0, len(dbuf) - lb * n0))
            for i in range(n0):
                buf = Buffer()
                buf.blocks = dbuf.blocks[st + i * lb:st + (i + 1) * lb]
                ret.append(buf)
            # 2. pos + neg
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            for i in range(n1):
                selected_pblks = random.sample(pbuf.blocks, min(lb, len(pbuf)))
                selected_nblks = random.sample(nbuf.blocks, min(lb - len(selected_pblks), len(nbuf)))
                buf = Buffer()
                buf.blocks = selected_pblks + selected_nblks
                ret.append(buf.sort_())
        return SimpleListDataset(ret)

    def build_attensum_random_buffer(self, num_samples):
        n0, n1 = [int(s) for s in num_samples.split(',')][:2]
        ret = []
        # 最多block数
        max_blk_num = CAPACITY // (BLOCK_SIZE + 2)
        logging.info('building attensum random buffers for introspection...')
        for qbuf, cbuf_list, dbuf, swag_label in tqdm(self.dataset):
            cbuf_block_max_num = 0
            for cbuf in cbuf_list:
                cbuf_block_max_num = max(len(cbuf), cbuf_block_max_num)
            lb = max_blk_num - len(qbuf) - cbuf_block_max_num
            # 1. continous
            st = random.randint(0, max(0, len(dbuf) - lb * n0))
            for i in range(n0):
                pbuf = Buffer()
                pbuf.blocks = dbuf.blocks[st + i * lb:st + (i + 1) * lb]
                buf_list, qblk_num, cblk_num, pblk_num = [], [], [], []
                for cbuf in cbuf_list:
                    qcp_buf = Buffer()
                    qcp_buf.blocks = qbuf.blocks + cbuf.blocks + pbuf.blocks
                    buf_list.append(qcp_buf)
                    qblk_num.append(len(qbuf))
                    cblk_num.append(len(cbuf))
                    pblk_num.append(len(pbuf))
                ret.append((buf_list, pbuf, qblk_num, cblk_num, pblk_num))
            # 2. pos + neg
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            for i in range(n1):
                selected_pblks = random.sample(pbuf.blocks, min(lb, len(pbuf)))
                selected_nblks = random.sample(nbuf.blocks, min(lb - len(selected_pblks), len(nbuf)))
                pbuf = Buffer()
                pbuf.blocks = selected_pblks + selected_nblks
                pbuf = pbuf.sort_()
                buf_list, qblk_num, cblk_num, pblk_num = [], [], [], []
                for cbuf in cbuf_list:
                    qcp_buf = Buffer()
                    qcp_buf.blocks = qbuf.blocks + cbuf.blocks + pbuf.blocks
                    buf_list.append(qcp_buf)
                    qblk_num.append(len(qbuf))
                    cblk_num.append(len(cbuf))
                    pblk_num.append(len(pbuf))
                ret.append((buf_list, pbuf, qblk_num, cblk_num, pblk_num))
        return SimpleListDataset(ret)

    def build_swag_promising_buffer(self, num_samples):
        n2, n3 = [int(x) for x in num_samples.split(',')][2:]
        ret = []
        max_blk_num = CAPACITY // (BLOCK_SIZE + 2)
        logging.info('building swag promising buffers for reasoning...')
        for qbuf, cbuf_list, dbuf, label in tqdm(self.dataset):
            # 1. retrieve top n2*(max-len(pos)) estimations into buf 2. cut
            pbuf, nbuf = dbuf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            cbuf_block_max_num = 0
            for cbuf in cbuf_list:
                cbuf_block_max_num = max(len(cbuf), cbuf_block_max_num)
            if len(pbuf) >= max_blk_num - len(qbuf) - cbuf_block_max_num:
                pbuf = pbuf.random_sample(max_blk_num - len(qbuf) - cbuf_block_max_num)
            lb = max_blk_num - len(qbuf) - cbuf_block_max_num - len(pbuf)
            estimations = torch.tensor([blk.estimation for blk in nbuf], dtype=torch.long)
            keeped_indices = estimations.argsort(descending=True)[:n2 * lb]
            selected_nblks = [blk for i, blk in enumerate(nbuf) if i in keeped_indices]
            while 0 < len(selected_nblks) < n2 * lb:
                selected_nblks = selected_nblks * (n2 * lb // len(selected_nblks) + 1)
            for i in range(n2):
                buf = Buffer()
                buf.blocks = pbuf.blocks + selected_nblks[i * lb: (i + 1) * lb]
                s_buf = buf.sort_()
                buf_list, qblk_num, cblk_num, pblk_num = [], [], [], []
                for cbuf in cbuf_list:
                    qcp_buf = Buffer()
                    qcp_buf.blocks = qbuf.blocks + cbuf.blocks + s_buf.blocks
                    buf_list.append(qcp_buf)
                    qblk_num.append(len(qbuf))
                    cblk_num.append(len(cbuf))
                    pblk_num.append(len(s_buf))
                ret.append((buf_list, s_buf, qblk_num, cblk_num, pblk_num, label))
            for i in range(n3):
                buf = Buffer()
                buf.blocks = pbuf.blocks + random.sample(nbuf.blocks, min(len(nbuf), lb))
                s_buf = buf.sort_()
                buf_list, qblk_num, cblk_num, pblk_num = [], [], [], []
                for cbuf in cbuf_list:
                    qcp_buf = Buffer()
                    qcp_buf.blocks = qbuf.blocks + cbuf.blocks + s_buf.blocks
                    buf_list.append(qcp_buf)
                    qblk_num.append(len(qbuf))
                    cblk_num.append(len(cbuf))
                    pblk_num.append(len(s_buf))
                ret.append((buf_list, s_buf, qblk_num, cblk_num, pblk_num, label))
        return SimpleListDataset(ret)


def find_lastest_checkpoint(checkpoints_dir, epoch=False):
    lastest = (-1, '')
    if os.path.exists(checkpoints_dir):
        for shortname in os.listdir(checkpoints_dir):
            # m = re.match(r'_ckpt_epoch_(\d+).+', shortname)
            m = re.match(r'wepoch=(\d+)-epoch=(\d+)-step=(\d+).+', shortname)
            if m is not None and int(m.group(1)) > lastest[0]:
                lastest = (int(m.group(1)), shortname)
    return os.path.join(checkpoints_dir, lastest[-1]) if not epoch else lastest[0]


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

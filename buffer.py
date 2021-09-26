# coding=utf8
import json

import torch
from copy import copy
from transformers import AutoTokenizer, BertTokenizer
from utils import CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME, CHINSESE_MODEL_PATH
import random
from bisect import bisect_left
from itertools import chain


class Block:
    def __init__(self, tokenizer, ids, pos, blk_type=2, **kwargs):
        self.tokenizer = tokenizer
        self.ids = ids
        self.pos = pos
        # 0 q, 1 c, 2, p
        self.blk_type = blk_type
        self.relevance = 0
        # 块分数
        self.estimation = 0
        self.__dict__.update(kwargs)

    def __lt__(self, rhs):
        return self.blk_type < rhs.blk_type or (self.blk_type == rhs.blk_type and self.pos < rhs.pos)

    def __ne__(self, rhs):
        return self.pos != rhs.pos or self.blk_type != rhs.blk_type

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(self.ids))


class Buffer:
    @staticmethod
    def split_chinese_document_into_blocks(d, tokenizer, cnt=0, dtype='d', properties=None):
        '''
            d: [['word', '##piece'], ...] # a document of tokenized sentences
            properties: [
                            [
                                (name: str, value: any), # len(2) tuple, sentence level property
                                (name: str, position: int, value: any) # len(3) tuple, token level property
                            ],
                            []... # len(d) lists
                        ]
        '''
        ret = Buffer()
        # d is only a list of tokens, not split.
        # properties are also a list of tuples.
        end_tokens = {'\n': 0, '。': 1, '？': 1, '！': 1, '，': 2}
        sen_cost, break_cost = 4, 8
        poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
        poses.insert(0, (-1, 0))
        if poses[-1][0] < len(d) - 1:
            poses.append((len(d) - 1, 0))
        x = 0
        while x < len(poses) - 1:
            if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
                poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
            x += 1
        # simple dynamic programming
        best = [(0, 0)]
        for i, (p, cost) in enumerate(poses):
            if i == 0:
                continue
            best.append((-1, 100000))
            for j in range(i - 1, -1, -1):
                if p - poses[j][0] > BLOCK_SIZE:
                    break
                value = best[j][1] + cost + sen_cost
                if value < best[i][1]:
                    best[i] = (j, value)
            assert best[i][0] >= 0
        intervals, x = [], len(poses) - 1
        while x > 0:
            l = poses[best[x][0]][0]
            intervals.append((l + 1, poses[x][0] + 1))
            x = best[x][0]
        if properties is None:
            properties = []
        for idx, (st, en) in enumerate(reversed(intervals)):
            cnt += 1
            tmp = [tokenizer.cls_token] + d[st: en] + [tokenizer.sep_token]
            # inject properties into blks
            tmp_kwargs = {}
            for p in properties:
                if len(p) == 2:
                    tmp_kwargs[p[0]] = p[1]
                elif len(p) == 3:
                    if st <= p[1] < en:
                        tmp_kwargs[p[0]] = (p[1] - st, p[2])
                else:
                    raise ValueError('Invalid property {}'.format(p))
            ret.insert(Block(
                tokenizer,
                tokenizer.convert_tokens_to_ids(tmp),
                cnt,
                **tmp_kwargs)
            )

        return ret, cnt

    def __init__(self):
        self.blocks = []

    def __add__(self, buf):
        ret = Buffer()
        ret.blocks = self.blocks + buf.blocks
        return ret

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, key):
        return self.blocks[key]

    def __str__(self):
        return ''.join([str(b) + '\n' for b in self.blocks])

    def clone(self):
        ret = Buffer()
        ret.blocks = self.blocks.copy()
        return ret

    def calc_size(self):
        return sum([len(b) for b in self.blocks])

    def block_ends(self):
        t, ret = 0, []
        for b in self.blocks:
            t += len(b)
            ret.append(t)
        return ret

    def insert(self, b, reverse=True):
        if not reverse:
            for index in range(len(self.blocks) + 1):
                if index >= len(self.blocks) or b < self.blocks[index]:
                    self.blocks.insert(index, b)
                    break
        else:
            for index in range(len(self.blocks), -1, -1):
                if index == 0 or self.blocks[index - 1] < b:
                    self.blocks.insert(index, b)
                    break

    def merge(self, buf):
        ret = Buffer()
        t1, t2 = 0, 0
        while t1 < len(self.blocks) or t2 < len(buf):
            if t1 < len(self.blocks) and (t2 >= len(buf) or self.blocks[t1] < buf.blocks[t2]):
                ret.blocks.append(self.blocks[t1])
                t1 += 1
            else:
                ret.blocks.append(buf.blocks[t2])
                t2 += 1
        return ret

    def filtered(self, fltr: 'function blk, index->bool', need_residue=False):
        ret, ret2 = Buffer(), Buffer()
        for i, blk in enumerate(self.blocks):
            if fltr(blk, i):
                ret.blocks.append(blk)
            else:
                ret2.blocks.append(blk)
        if need_residue:
            return ret, ret2
        else:
            return ret

    def random_sample(self, size):
        assert size <= len(self.blocks)
        index = sorted(random.sample(range(len(self.blocks)), size))
        ret = Buffer()
        ret.blocks = [self.blocks[i] for i in index]
        return ret

    def sort_(self):
        self.blocks.sort()
        return self

    # <cls>ppp<sep>
    # def fill(self, buf):
    #     ret, tmp_buf, tmp_size, blk_count = [], self.clone(), self.calc_size(), 0
    #     for blk in buf:
    #         if tmp_size + len(blk) - 2*blk_count > CAPACITY:
    #             ret.append(tmp_buf)
    #             tmp_buf, tmp_size, blk_count = self.clone(), self.calc_size(), 0
    #         tmp_buf.blocks.append(blk)
    #         tmp_size += len(blk)
    #         blk_count += 1
    #     ret.append(tmp_buf)
    #     return ret

    # <cls><sep>p<sep>p<sep>p<sep>
    # def fill(self, buf):
    #     ret, tmp_buf, tmp_size, blk_count = [], self.clone(), self.calc_size(), 0
    #     for blk in buf:
    #         if tmp_size + len(blk) - (blk_count + 1) + 2 > CAPACITY:
    #             ret.append(tmp_buf)
    #             tmp_buf, tmp_size, blk_count = self.clone(), self.calc_size(), 0
    #         tmp_buf.blocks.append(blk)
    #         tmp_size += len(blk)
    #         blk_count += 1
    #     ret.append(tmp_buf)
    #     return ret

    # # <cls>p<sep><cls>p<sep><cls>p<sep>
    # def fill(self, buf):
    #     ret, tmp_buf, tmp_size, blk_count = [], self.clone(), self.calc_size(), 0
    #     for blk in buf:
    #         if tmp_size + len(blk) > CAPACITY:
    #             ret.append(tmp_buf)
    #             tmp_buf, tmp_size, blk_count = self.clone(), self.calc_size(), 0
    #         tmp_buf.blocks.append(blk)
    #         tmp_size += len(blk)
    #         blk_count += 1
    #     ret.append(tmp_buf)
    #     return ret

    # <cls>q<sep><cls>c<sep><cls>p<sep><cls>p<sep><cls>p<sep>
    def fill(self, qbuf, cbufs, dbuf, qc_max_trans_length):
        pbufs, tmp_buf, tmp_size = [], self.clone(), self.calc_size()
        for blk in dbuf:
            if qc_max_trans_length + tmp_size + len(blk) > CAPACITY:
                pbufs.append(tmp_buf)
                tmp_buf, tmp_size = self.clone(), self.calc_size()
            tmp_buf.blocks.append(blk)
            tmp_size += len(blk)
        pbufs.append(tmp_buf)
        ret = []
        for pbuf in pbufs:
            buf_list, qblk_num, cblk_num, pblk_num = [], [], [], []
            for cbuf in cbufs:
                qcp_buf = Buffer()
                qcp_buf.blocks = qbuf.blocks + cbuf.blocks + pbuf.blocks
                buf_list.append(qcp_buf)
                qblk_num.append(len(qbuf))
                cblk_num.append(len(cbuf))
                pblk_num.append(len(pbuf))
            ret.append((buf_list, pbuf, qblk_num, cblk_num, pblk_num))
        return ret

    def export_relevance(self, out=None, blks=None, blk_ends=None):
        relevance = out
        t = 2
        for idx, blk_end in enumerate(blk_ends):
            w = blk_end
            if blks[idx].relevance >= 1:
                relevance[t: w] = 1
            t = w
        return relevance

    # def export_intro(self, out=None):
    #     ids, att_masks, type_ids = out
    #     att_masks.zero_()
    #     t = 0
    #     blk_ends = []
    #     # blk_strs = []
    #     blk_num = len(self.blocks)
    #     for idx, b in enumerate(self.blocks):
    #         # 默认去cls去sep
    #         tmp_length = len(b) - 2
    #         ids[t:t + tmp_length] = torch.tensor(b.ids[1:tmp_length + 1], dtype=torch.long)  # id
    #         if idx == 0:
    #             # 首block只去sep
    #             tmp_length = len(b) - 1
    #             ids[t:t + tmp_length] = torch.tensor(b.ids[0:tmp_length], dtype=torch.long)  # id
    #         if idx == blk_num - 1:
    #             # 尾block只去cls
    #             tmp_length = len(b) - 1
    #             ids[t:t + tmp_length] = torch.tensor(b.ids[1:], dtype=torch.long)  # id
    #         if blk_num == 1:
    #             # 只有一个block保持cls保持sep
    #             tmp_length = len(b)
    #             ids[t:t + tmp_length] = torch.tensor(b.ids, dtype=torch.long)  # id
    #         # blk_strs.append(b.tokenizer.convert_tokens_to_string(b.tokenizer.convert_ids_to_tokens(ids[t:t + tmp_length])))
    #         if b.blk_type == 2:
    #             type_ids[t:t + tmp_length] = 1
    #         att_masks[t:t + tmp_length] = 1  # attention_mask
    #         t += tmp_length
    #         blk_ends.append(t)
    #     return ids, att_masks, type_ids, blk_ends

    # <cls><sep>p<sep>p<sep>p<sep>
    def export_intro(self, out=None):
        ids, att_masks, type_ids = out
        att_masks.zero_()
        blk_ends = []
        ids[0:2] = torch.tensor([101, 102], dtype=torch.long)
        t = 2
        for idx, b in enumerate(self.blocks):
            # 默认去cls
            tmp_length = len(b) - 1
            ids[t:t + tmp_length] = torch.tensor(b.ids[1:], dtype=torch.long)  # id
            # print(b.tokenizer.convert_tokens_to_string(b.tokenizer.convert_ids_to_tokens(ids[t:t + tmp_length])))
            if b.blk_type == 2:
                type_ids[t:t + tmp_length] = 1
            att_masks[t:t + tmp_length] = 1  # attention_mask
            t += tmp_length
            blk_ends.append(t)
        return ids, att_masks, type_ids, blk_ends

    # <cls>p<sep><cls>p<sep><cls>p<sep>
    def export_bertsum_intro(self, out=None):
        ids, att_masks, type_ids = out
        att_masks.zero_()
        blk_ends = []
        clses = []
        t = 0
        for idx, b in enumerate(self.blocks):
            clses.append(t)
            tmp_length = len(b)
            ids[t:t + tmp_length] = torch.tensor(b.ids, dtype=torch.long)  # id
            # print(b.tokenizer.convert_tokens_to_string(b.tokenizer.convert_ids_to_tokens(ids[t:t + tmp_length])))
            type_ids[t:t + tmp_length] = idx % 2
            att_masks[t:t + tmp_length] = 1  # attention_mask
            t += tmp_length
            blk_ends.append(t)
        return ids, att_masks, type_ids, blk_ends, clses

    # <cls>q<sep><cls>c<sep><cls>p<sep><cls>p<sep><cls>p<sep>
    def export_attensum_intro(self, q_num, c_num, p_num, out=None):
        ids, att_masks, type_ids = out
        question_len, choice_len, passage_len = 0, 0, 0
        att_masks.zero_()
        cls, label = [], []
        t = 0
        for idx, b in enumerate(self.blocks):
            if idx >= q_num + c_num:
                cls.append(t)
                label.append(b.relevance)
            tmp_length = len(b)
            ids[t:t + tmp_length] = torch.tensor(b.ids, dtype=torch.long)  # id
            if idx == 0 or idx == q_num:
                # q,c首block只去sep
                tmp_length = len(b) - 1
                ids[t:t + tmp_length] = torch.tensor(b.ids[0:tmp_length], dtype=torch.long)  # id
            if idx == q_num - 1 or idx == q_num + c_num - 1:
                # q,c的尾block只去cls
                tmp_length = len(b) - 1
                ids[t:t + tmp_length] = torch.tensor(b.ids[1:], dtype=torch.long)  # id
            if (idx > 0 and idx < q_num - 1) or (idx > q_num and idx < q_num + c_num - 1):
                # q,c首尾间的block去cls和sep
                tmp_length = len(b) - 2
                ids[t:t + tmp_length] = torch.tensor(b.ids[1:tmp_length + 1], dtype=torch.long)  # id
            if (idx == 0 and q_num == 1) or (idx == q_num and c_num == 1):
                # 如果q,c只有一个block保持cls保持sep
                tmp_length = len(b)
                ids[t:t + tmp_length] = torch.tensor(b.ids, dtype=torch.long)  # id
            # print(b.tokenizer.convert_tokens_to_string(b.tokenizer.convert_ids_to_tokens(ids[t:t + tmp_length])))
            type_ids[t:t + tmp_length] = idx % 2
            att_masks[t:t + tmp_length] = 1  # attention_mask
            t += tmp_length
            if idx == q_num - 1:
                question_len = t
            if idx == q_num + c_num - 1:
                choice_len = t - question_len
            if idx == q_num + c_num + p_num - 1:
                passage_len = t - question_len - choice_len
        first_cls = cls[0]
        cls = [c - first_cls for c in cls]
        return ids, att_masks, type_ids, question_len, choice_len, passage_len, cls, label

    def export_reason(self, q_num, c_num, p_num, out=None):
        ids, att_masks, type_ids = out
        question_len, choice_len, passage_len = 0, 0, 0
        att_masks.zero_()
        t = 0
        # blk_strs = []
        for idx, b in enumerate(self.blocks):
            # 默认去cls去sep
            tmp_length = len(b) - 2
            ids[t:t + tmp_length] = torch.tensor(b.ids[1:tmp_length + 1], dtype=torch.long)  # id
            if idx == 0:
                # 首block只去sep
                tmp_length = len(b) - 1
                ids[t:t + tmp_length] = torch.tensor(b.ids[0:tmp_length], dtype=torch.long)  # id
            if idx == q_num - 1 or idx == q_num + c_num - 1 or idx == q_num + c_num + p_num - 1:
                # q,c,p的尾block只去cls
                tmp_length = len(b) - 1
                ids[t:t + tmp_length] = torch.tensor(b.ids[1:], dtype=torch.long)  # id
            if idx == 0 and q_num == 1:
                # 如果q只有一个block保持cls保持sep
                tmp_length = len(b)
                ids[t:t + tmp_length] = torch.tensor(b.ids, dtype=torch.long)  # id
            # blk_strs.append(
            #     b.tokenizer.convert_tokens_to_string(b.tokenizer.convert_ids_to_tokens(ids[t:t + tmp_length])))
            if b.blk_type == 2:
                type_ids[t:t + tmp_length] = 1
            att_masks[t:t + tmp_length] = 1  # attention_mask
            t += tmp_length
            if idx == q_num - 1:
                question_len = t
            if idx == q_num + c_num - 1:
                choice_len = t - question_len
            if idx == q_num + c_num + p_num - 1:
                passage_len = t - question_len - choice_len
        return ids, att_masks, type_ids, question_len, choice_len, passage_len

    # def export_reason(self, q_num, c_num, p_num, out=None):
    #     ids, att_masks, type_ids = out
    #     question_len, choice_len, passage_len = 0, 0, 0
    #     att_masks.zero_()
    #     t = 0
    #     for idx, b in enumerate(self.blocks):
    #         # 默认去cls去sep
    #         tmp_length = len(b) - 2
    #         ids[t:t + tmp_length] = torch.tensor(b.ids[1:tmp_length + 1], dtype=torch.long)  # id
    #         if idx == 0:
    #             # 首block只去sep
    #             tmp_length = len(b) - 1
    #             ids[t:t + tmp_length] = torch.tensor(b.ids[0:tmp_length], dtype=torch.long)  # id
    #         if idx == q_num - 1 or idx == q_num + c_num - 1 or idx >= q_num + c_num:
    #             # q,c的尾block只去cls
    #             tmp_length = len(b) - 1
    #             ids[t:t + tmp_length] = torch.tensor(b.ids[1:], dtype=torch.long)  # id
    #         if idx == 0 and q_num == 1:
    #             # 如果q只有一个block保持cls保持sep
    #             tmp_length = len(b)
    #             ids[t:t + tmp_length] = torch.tensor(b.ids, dtype=torch.long)  # id
    #         if b.blk_type == 2:
    #             type_ids[t:t + tmp_length] = 1
    #         att_masks[t:t + tmp_length] = 1  # attention_mask
    #         t += tmp_length
    #         if idx == q_num - 1:
    #             question_len = t
    #         if idx == q_num + c_num - 1:
    #             choice_len = t - question_len
    #         if idx == q_num + c_num + p_num - 1:
    #             passage_len = t - question_len - choice_len
    #     return ids, att_masks, type_ids, question_len, choice_len, passage_len


def buffer_collate_for_intro(bufs):
    inputs = torch.zeros(4, len(bufs), CAPACITY, dtype=torch.long)
    blk_ends_list = []
    for i, buf in enumerate(bufs):
        blk_ends = buf.export_intro(out=(inputs[0, i], inputs[1, i], inputs[2, i]))[3]
        blk_ends_list.append(blk_ends)
    inputs[2].zero_()
    # Train the introspector after labeling
    for i, buf in enumerate(bufs):
        buf.export_relevance(out=inputs[3, i], blks=buf.blocks, blk_ends=blk_ends_list[i])
    return inputs, blk_ends_list, bufs


def buffer_collate_for_bertsumintro(bufs):
    inputs = torch.zeros(3, len(bufs), CAPACITY, dtype=torch.long)
    blk_ends_list = []
    clses_list = []
    labels_list = []
    for i, buf in enumerate(bufs):
        blk_ends, clses = buf.export_bertsum_intro(out=(inputs[0, i], inputs[1, i], inputs[2, i]))[3:5]
        blk_ends_list.append(blk_ends)
        clses_list.append(clses)
        labels = []
        for blk in buf.blocks:
            if blk.relevance >= 1:
                labels.append(1)
            else:
                labels.append(0)
        labels_list.append(labels)
    clses_list = torch.tensor(_pad(clses_list, -1), dtype=torch.long)
    labels_list = torch.tensor(_pad(labels_list, -1), dtype=torch.float)
    mask_cls = torch.ones_like(clses_list)
    mask_cls[clses_list == -1] = 0
    clses_list[clses_list == -1] = 0
    return inputs, blk_ends_list, clses_list, labels_list, mask_cls, bufs


def buffer_collate_for_attensumintro(data):
    inputs = torch.zeros(3, len(data), 4, CAPACITY, dtype=torch.long)
    pbufs, question_lens_list, choice_lens_list, passage_lens_list, clses_list, labels_list = [], [], [], [], [], []
    for i, (buf_list, pbuf, qblk_num, cblk_num, pblk_num) in enumerate(data):
        question_lens, choice_lens, passage_lens, clses, labels = [], [], [], [], []
        for j, buf in enumerate(buf_list):
            question_len, choice_len, passage_len, cls, label = buf.export_attensum_intro(
                qblk_num[j],
                cblk_num[j],
                pblk_num[j],
                out=(
                    inputs[0, i, j],
                    inputs[1, i, j],
                    inputs[2, i, j]
                )
            )[3:]
            question_lens.append(question_len)
            choice_lens.append(choice_len)
            passage_lens.append(passage_len)
            clses.append(cls)
            labels.append(label)
        pbufs.append(pbuf)
        question_lens_list.append(question_lens)
        choice_lens_list.append(choice_lens)
        passage_lens_list.append(passage_lens)
        clses_list.append(clses[0])
        labels_list.append(labels[0])
    question_lens_tensor = torch.tensor(question_lens_list, dtype=torch.long)
    chioce_lens_tensor = torch.tensor(choice_lens_list, dtype=torch.long)
    passage_lens_tensor = torch.tensor(passage_lens_list, dtype=torch.long)
    clses_tensor = torch.tensor(_pad(clses_list, -1), dtype=torch.long)
    labels_tensor = torch.tensor(_pad(labels_list, -1), dtype=torch.float)
    mask_cls = torch.ones_like(clses_tensor)
    mask_cls[clses_tensor == -1] = 0
    clses_tensor[clses_tensor == -1] = 0
    labels_tensor[labels_tensor == -1] = 0
    return inputs, labels_tensor, pbufs, question_lens_tensor, chioce_lens_tensor, passage_lens_tensor, clses_tensor, mask_cls


def buffer_collate_for_reason(data):  # does not collate
    # Make inputs for reasoner
    inputs = torch.zeros(3, len(data), 4, 512, dtype=torch.long)
    labels = torch.zeros(len(data), dtype=torch.long)
    pbufs, question_lens_list, choice_lens_list, passage_lens_list = [], [], [], []
    for i, (buf_list, pbuf, qblk_num, cblk_num, pblk_num, label) in enumerate(data):
        question_lens, choice_lens, passage_lens = [], [], []
        for j, buf in enumerate(buf_list):
            question_len, choice_len, passage_len = buf.export_reason(
                qblk_num[j],
                cblk_num[j],
                pblk_num[j],
                out=(
                    inputs[0, i, j],
                    inputs[1, i, j],
                    inputs[2, i, j]
                )
            )[3:]
            question_lens.append(question_len)
            choice_lens.append(choice_len)
            passage_lens.append(passage_len)
        pbufs.append(pbuf)
        question_lens_list.append(question_lens)
        choice_lens_list.append(choice_lens)
        passage_lens_list.append(passage_lens)
        labels[i] = label
    question_lens_tensor = torch.tensor(question_lens_list, dtype=torch.long)
    chioce_lens_tensor = torch.tensor(choice_lens_list, dtype=torch.long)
    passage_lens_tensor = torch.tensor(passage_lens_list, dtype=torch.long)
    return inputs, labels, pbufs, question_lens_tensor, chioce_lens_tensor, passage_lens_tensor


def _pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    return rtn_data


if __name__ == "__main__":
    s = "材料一：日前，中国科学院在京召开新闻发布会对外宣布“墨子号”进行科学实验卫星提前并圆满实现全部既定科学目标，为我国在未来继续引领世界量子通过研究奠定了坚实的基础。通信安全是国家信息安全和人类经济社会生活的基本需求。千百年来，人们对于通信安全的追求从未停止。然而，基于计算复杂性的传统加密技术，在原理上存在着被破译的可能性，随着数学和计算能力的不断提升，经典密码被破译的可能性与日俱增，中国科学技术大学潘建伟教授说：“通过量子通信可以解决这个问题，把量子物理与信息技术相结合，利用量子调控技术，用一种革命性的方式对信息进行编码、存储、传输和操纵，从而在确保信息安全，提高运算速度，提升测量精度等方面突破经典信息技术的瓶颈。”量子通信主要研究内容包括量子密匙分发和量子隐形的传态，量子密匙分发通过量子态的传输，使遥远两地的用户可以共享无条件安全的密匙，利用该密匙对信息进行一次一密的严格加密，这是目前人类唯一已知的不可窃听，不可破译的无条件安全的通信方式。量子通信的另一重要内容量子隐形传态，是利用量子纠缠特性讲物质的未知量子态精确传送到遥远地点，而不用传送物质本身。通过隐形传输实现信息传送。        （摘编自吴月辉《墨子号”，抢占量子科技创新制高点》，《人民日报》2017年8月10日）材料二：潘建伟的导师安东·蔡林格说，潘建伟的团队在量子互联网的发展方面冲到了领先地位，量子互联网是由卫星和地面设备构成的能够在全球范围分享量子信息的网络。这将使不可破解的全球加密通信成为可能，同时也使我们可以开展一些新的控制远距离量子联系的实验。目前，潘建伟的团队计划发射第二颗卫星，他们还在中国的天宫二号空间站上进行着一项太空量子实验，潘建伟说，未来五年“还会取得很多精彩的成果，一个新的时代已经到来”。潘建伟是一个有着无穷热情的乐观主义者，他低调地表达了自己的信心，称中国政府将会支持下一个宏伟计划——一项投资20亿美元的量子通信、量子计量和量子计划的五年计划，与此形成对照的是欧洲2016年宣布的旗舰项目，投资额为12亿美元。（摘编自伊丽莎白·吉布尼《一位把量子通信带到太空又带回地球的物理学家》、《自然》2017年12月）材料三：日本《读卖新闻》5月2日报道：中国实验设施瞄准一流（记者：莳田一彦、船越翔）在中国南部广东省东莞市郊外的丘陵地带，中国刚才刚建成了大型实验设施，“中国散裂中子源”，该实验设施建设费用达到23亿元人民币，3月正式投入运行。中国是继美国、英国、日本之后第四个拥有同样设施的国家。日本的J-PARC加速器设施中心主任齐藤直人说：“虽然日本在技术和经验上领先，但中国发展得实在太快，亚洲的中心正在从日本向中国转移。”中国推进的这类大型工程还有很多。3月，中国人民政治协商会议开幕。政协委员潘建伟被媒体记者团团围住。潘建伟是利用2016年发射的“墨子号”人造卫星进行量子通信研究的研究团队负责人，其团队2017年以后相继发布了多项世界首创的实验成果。潘建伟今年当选美国《时代》杂志“全球百大最具影响力人物”。使用人造卫星的实验要耗费巨额资金，欧洲和日本还在犹豫不决、日本的研究人员还在犹豫不决，日本的研究人员认为，“在基础科学领域，中国正在踏入他国难以涉足的领域，领先世界”。"
    from transformers import AutoModel, AutoTokenizer

    # tokenizer = AutoTokenizer.from_pretrained(r"./ft_model/roberta-base")
    tokenizer = BertTokenizer.from_pretrained(r"./ft_model/chinese-roberta-wwm-ext")

    for blk in Buffer.split_chinese_document_into_blocks(tokenizer.tokenize(s), tokenizer, hard=False)[0]:
        print(blk)

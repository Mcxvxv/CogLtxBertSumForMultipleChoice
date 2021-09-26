import torch
import torch.nn.functional as F

from utils import CAPACITY
from buffer import Buffer, _pad, buffer_collate_for_attensumintro


def _score_blocks(buf, blk_ends, relevance_token):
    relevance_blk = torch.ones(len(blk_ends), device='cpu')
    for i in range(len(blk_ends)):
        if buf[i].blk_type == 2:
            start = 2 if i == 0 else blk_ends[i - 1]
            # end = blk_ends[i] - 1 if i == len(blk_ends) - 1 else blk_ends[i]
            end = blk_ends[i]
            relevance_blk[i] = (relevance_token[start:end]).mean()
    return relevance_blk


def positional_smoothing(buf, relevance_blk, factor_forward=0.1, factor_backward=0.3):
    ret = torch.zeros_like(relevance_blk)
    for i, blk in enumerate(buf):
        rest = 1.
        if i > 0 and buf[i - 1].pos == blk.pos - 1:
            rest -= factor_forward
            ret[i] += relevance_blk[i - 1] * factor_forward
        if i < len(buf) - 1 and buf[i + 1].pos == blk.pos + 1:
            rest -= factor_backward
            ret[i] += relevance_blk[i + 1] * factor_backward
        ret[i] += relevance_blk[i] * rest
        ret[i] = max(ret[i], relevance_blk[i])
    return ret


def mem_replay(introspector, fill_buf, dbuf, device, times='3,5', batch_size_inference=16, qc_max_trans_length=0):
    '''
        times: increased number of blocks each replay.
    '''
    times = [int(x) for x in times.split(',')]
    inputs = torch.zeros(3, batch_size_inference, CAPACITY, dtype=torch.long, device=device)
    B_set = []  # the poses of B blks in qbuf
    for k, inc in enumerate(times):
        num_to_keep = len(fill_buf) + inc
        # stage one: continuous
        estimations = torch.zeros(len(dbuf), device='cpu')
        bufs, t = fill_buf.fill(dbuf), 0
        for i in range((len(bufs) - 1) // batch_size_inference + 1):
            l, r = batch_size_inference * i, min(len(bufs), batch_size_inference * (i + 1))
            blk_ends_list = []
            for j, buf in enumerate(bufs[l:r]):
                blk_ends = buf.export_intro(out=(inputs[0, j], inputs[1, j], inputs[2, j]))[3]
                blk_ends_list.append(blk_ends)
            logits = introspector(*inputs[:, :r - l]).sigmoid_()
            for j, buf in enumerate(bufs[l:r]):
                estimation = _score_blocks(buf, blk_ends_list[j], logits[j])[len(fill_buf):]
                estimations[t: t + len(estimation)] = estimation
                t += len(estimation)
        assert t == len(dbuf)

        # estimations = positional_smoothing(dbuf, estimations)
        # fill the buffer up
        # 按照reasoner的输入格式限制长度
        indices = estimations.argsort(descending=True)
        if fill_buf.calc_size() == 0:
            fill_buf_size = 0
        else:
            fill_buf_size = fill_buf.calc_size() - (len(fill_buf) - 1) * 2 - 1
        for idx in indices:
            tmp_length = qc_max_trans_length + fill_buf_size + len(dbuf[idx]) - 2
            if fill_buf_size == 0:
                tmp_length = qc_max_trans_length + len(dbuf[idx]) - 1
            if tmp_length > CAPACITY:
                break
            if dbuf[idx] in B_set:
                continue
            if fill_buf_size == 0:
                fill_buf_size = fill_buf_size + len(dbuf[idx]) - 1
            else:
                fill_buf_size = fill_buf_size + len(dbuf[idx]) - 2
            fill_buf.insert(dbuf[idx])

        # 按照intro的输入格式限制长度
        tmp_length = 0
        blk_count = 0
        break_flag = False
        for blk in fill_buf.blocks:
            blk_count += 1
            if tmp_length + len(blk) - blk_count + 2 > CAPACITY:
                break_flag = True
                break
            tmp_length += len(blk)
        if break_flag:
            fill_buf.blocks = fill_buf.blocks[0:blk_count - 1]
        else:
            fill_buf.blocks = fill_buf.blocks[0:blk_count]

        # keep only num_to_keep blks
        blk_ends = fill_buf.export_intro(out=(inputs[0, 0], inputs[1, 0], inputs[2, 0]))[3]
        relevance_token = torch.sigmoid(introspector(*inputs[:, :1]).view(-1))

        relevance_blk = _score_blocks(fill_buf, blk_ends, relevance_token)
        keeped_indices = relevance_blk.argsort(descending=True)
        if len(keeped_indices) > num_to_keep and k < len(times) - 1:
            keeped_indices = keeped_indices[0:num_to_keep]
        else:
            return fill_buf, relevance_blk
        # manually filtering
        filtered_qbuf, filtered_relevance_blk = Buffer(), []
        for i, blk in enumerate(fill_buf):
            if i in keeped_indices:
                filtered_qbuf.blocks.append(blk)
                filtered_relevance_blk.append(relevance_blk[i])
        fill_buf = filtered_qbuf
        # record the blocks already in the qbuf
        B_set = [blk for blk in fill_buf if blk.blk_type == 2]
    return filtered_qbuf, torch.tensor(filtered_relevance_blk)


def bertsum_mem_replay(introspector, fill_buf, dbuf, device, times='3,5', batch_size_inference=16,
                       qc_max_trans_length=0):
    '''
        times: increased number of blocks each replay.
    '''
    times = [int(x) for x in times.split(',')]
    inputs = torch.zeros(3, batch_size_inference, CAPACITY, dtype=torch.long, device=device)
    B_set = []  # the poses of B blks in qbuf
    for k, inc in enumerate(times):
        num_to_keep = len(fill_buf) + inc
        # stage one: continuous
        estimations = torch.zeros(len(dbuf), device='cpu')
        bufs, t = fill_buf.fill(dbuf), 0
        for i in range((len(bufs) - 1) // batch_size_inference + 1):
            l, r = batch_size_inference * i, min(len(bufs), batch_size_inference * (i + 1))
            blk_ends_list = []
            clses_list = []
            for j, buf in enumerate(bufs[l:r]):
                blk_ends, clses = buf.export_bertsum_intro(out=(inputs[0, j], inputs[1, j], inputs[2, j]))[3:5]
                blk_ends_list.append(blk_ends)
                clses_list.append(clses)
            clses_list = torch.tensor(_pad(clses_list, -1), dtype=torch.long).to(device)
            mask_cls = torch.ones_like(clses_list).to(device)
            mask_cls[clses_list == -1] = 0
            clses_list[clses_list == -1] = 0
            logits = introspector(*inputs[:, :r - l], None, clses_list, mask_cls)
            for j, buf in enumerate(bufs[l:r]):
                estimation = logits[j][mask_cls[j] == 1][len(fill_buf):]
                estimations[t: t + len(estimation)] = estimation
                t += len(estimation)
        assert t == len(dbuf)

        # estimations = positional_smoothing(dbuf, estimations)
        # fill the buffer up
        indices = estimations.argsort(descending=True)
        if fill_buf.calc_size() == 0:
            fill_buf_size = 0
        else:
            fill_buf_size = fill_buf.calc_size() - (len(fill_buf) - 1) * 2 - 1
        for idx in indices:
            tmp_length = qc_max_trans_length + fill_buf_size + len(dbuf[idx]) - 2
            if fill_buf_size == 0:
                tmp_length = qc_max_trans_length + len(dbuf[idx]) - 1
            if tmp_length > CAPACITY:
                break
            if dbuf[idx] in B_set:
                continue
            if fill_buf_size == 0:
                fill_buf_size = fill_buf_size + len(dbuf[idx]) - 1
            else:
                fill_buf_size = fill_buf_size + len(dbuf[idx]) - 2
            fill_buf.insert(dbuf[idx])

        tmp_length = 0
        blk_count = 0
        break_flag = False
        for blk in fill_buf.blocks:
            blk_count += 1
            if tmp_length + len(blk) > CAPACITY:
                break_flag = True
                break
            tmp_length += len(blk)
        if break_flag:
            fill_buf.blocks = fill_buf.blocks[0:blk_count - 1]
        else:
            fill_buf.blocks = fill_buf.blocks[0:blk_count]

        # keep only num_to_keep blks
        blk_ends, clses = fill_buf.export_bertsum_intro(out=(inputs[0, 0], inputs[1, 0], inputs[2, 0]))[3:5]
        clses = torch.tensor(clses, dtype=torch.long).to(device).unsqueeze(0)
        mask_cls = torch.ones_like(clses).to(device)
        relevance_blk = torch.sigmoid(introspector(*inputs[:, :1], None, clses, mask_cls).view(-1))
        keeped_indices = relevance_blk.argsort(descending=True)
        if len(keeped_indices) > num_to_keep and k < len(times) - 1:
            keeped_indices = keeped_indices[0:num_to_keep]
        else:
            return fill_buf, relevance_blk
        # manually filtering
        filtered_qbuf, filtered_relevance_blk = Buffer(), []
        for i, blk in enumerate(fill_buf):
            if i in keeped_indices:
                filtered_qbuf.blocks.append(blk)
                filtered_relevance_blk.append(relevance_blk[i])
        fill_buf = filtered_qbuf
        # record the blocks already in the qbuf
        B_set = [blk for blk in fill_buf if blk.blk_type == 2]
    return filtered_qbuf, torch.tensor(filtered_relevance_blk)


def attensum_mem_replay(introspector, fill_buf, qbuf, cbufs, dbuf, device, times='3,5', batch_size_inference=16,
                        qc_max_trans_length=0):
    '''
        times: increased number of blocks each replay.
    '''
    times = [int(x) for x in times.split(',')]
    B_set = []  # the poses of B blks in qbuf
    for k, inc in enumerate(times):
        num_to_keep = len(fill_buf) + inc
        # stage one: continuous
        estimations = torch.zeros(len(dbuf), device='cpu')
        data = fill_buf.fill(qbuf, cbufs, dbuf, qc_max_trans_length)
        t = 0
        for i in range((len(data) - 1) // batch_size_inference + 1):
            l, r = batch_size_inference * i, min(len(data), batch_size_inference * (i + 1))
            capacity_data = data[l:r]
            inputs, labels, pbufs, question_lens, chioce_lens, passage_lens, clses, mask_cls = buffer_collate_for_attensumintro(capacity_data)
            inputs = inputs.to(device)
            question_lens = inputs.to(device)
            chioce_lens = inputs.to(device)
            passage_lens = inputs.to(device)
            clses = inputs.to(device)
            mask_cls = inputs.to(device)
            logits = introspector(*inputs, None, question_lens, chioce_lens, passage_lens, clses, mask_cls)
            for j, data in enumerate(data[l:r]):
                estimation = logits[j][mask_cls[j] == 1][len(fill_buf):]
                estimations[t: t + len(estimation)] = estimation
                t += len(estimation)
        assert t == len(dbuf)

        # estimations = positional_smoothing(dbuf, estimations)
        # fill the buffer up
        indices = estimations.argsort(descending=True)
        tmp_length = fill_buf.calc_size()
        for idx in indices:
            if qc_max_trans_length + tmp_length + len(dbuf[idx]) > CAPACITY:
                break
            if dbuf[idx] in B_set:
                continue
            fill_buf.insert(dbuf[idx])
            tmp_length += len(dbuf[idx])


        buf_list, qblk_num, cblk_num, pblk_num = [], [], [], []
        for cbuf in cbufs:
            qcp_buf = Buffer()
            qcp_buf.blocks = qbuf.blocks + cbuf.blocks + fill_buf.blocks
            buf_list.append(qcp_buf)
            qblk_num.append(len(qbuf))
            cblk_num.append(len(cbuf))
            pblk_num.append(len(fill_buf))
        data = [(buf_list, fill_buf, qblk_num, cblk_num, pblk_num)]

        # keep only num_to_keep blks
        inputs, labels, pbufs, question_lens, chioce_lens, passage_lens, clses, mask_cls = buffer_collate_for_attensumintro(data)
        inputs = inputs.to(device)
        question_lens = inputs.to(device)
        chioce_lens = inputs.to(device)
        passage_lens = inputs.to(device)
        clses = inputs.to(device)
        mask_cls = inputs.to(device)
        relevance_blk = introspector(*inputs, None, question_lens, chioce_lens, passage_lens, clses, mask_cls).view(-1)
        keeped_indices = relevance_blk.argsort(descending=True)
        if len(keeped_indices) > num_to_keep and k < len(times) - 1:
            keeped_indices = keeped_indices[0:num_to_keep]
        else:
            return fill_buf, relevance_blk
        # manually filtering
        filtered_qbuf, filtered_relevance_blk = Buffer(), []
        for i, blk in enumerate(fill_buf):
            if i in keeped_indices:
                filtered_qbuf.blocks.append(blk)
                filtered_relevance_blk.append(relevance_blk[i])
        fill_buf = filtered_qbuf
        # record the blocks already in the qbuf
        B_set = [blk for blk in fill_buf if blk.blk_type == 2]
    return filtered_qbuf, torch.tensor(filtered_relevance_blk)

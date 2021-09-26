import copy
import json
import os
from argparse import ArgumentParser


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForMultipleChoice

from buffer import Buffer, buffer_collate_for_intro, buffer_collate_for_reason, buffer_collate_for_bertsumintro, \
    buffer_collate_for_attensumintro
from data_helper import SimpleListDataset, SwagBlkPosInterface
from memreplay import mem_replay, attensum_mem_replay
from models import SwagIntrospector, BertSumIntrospector, AttenSumIntrospector

def test_intro():
    train_source = "./hh_data/hh_train.pkl"
    dataset = SimpleListDataset(train_source)
    interface = SwagBlkPosInterface(dataset)
    dataset = interface.build_swag_random_buffer(num_samples='1,1,1,1')
    for data in dataset:
        print(data)
        break
    # dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=buffer_collate_for_intro, shuffle=False)
    # introspector = SwagIntrospector.from_pretrained("../pretrain_model/chinese-bert-wwm-ext")
    # for batch in tqdm(dataloader):
    #     inputs, blk_ends_list, bufs = batch
    #     # for blk_strs in blk_strs_list:
    #     #     blk_str = ''.join(blk_strs)
    #     #     if blk_str.count("[CLS]") > 1 or blk_str.count("[SEP]") > 1:
    #     #         print("[CLS]:", blk_str.count("[CLS]"), "[SEP]:", blk_str.count("[SEP]"))
    #     # loss_introspector, logits = introspector(*inputs[:3], labels=inputs[3])
    #     # for i, buf in enumerate(bufs):
    #     #     blk_ends = blk_ends_list[i]
    #     #     _write_estimation(buf, _score_blocks(inputs[0][i], buf, blk_ends, torch.sigmoid(logits[i])))
    #     break


def _write_estimation(buf, relevance_blk):
    for i, blk in enumerate(buf):
        print(f'{blk.pos} {relevance_blk[i].item()}\n')


def _score_blocks(ids, buf, blk_ends, relevance_token):
    relevance_blk = torch.ones(len(blk_ends), device='cpu')
    for i in range(len(blk_ends)):
        if buf[i].blk_type == 2:
            start = 1 if i == 0 else blk_ends[i - 1]
            end = blk_ends[i] - 1 if i == len(blk_ends) - 1 else blk_ends[i]
            print("=" * 20)
            print(buf[i].tokenizer.convert_tokens_to_string(buf[i].tokenizer.convert_ids_to_tokens(ids[start:end])))
            relevance_blk[i] = (relevance_token[start:end]).mean()
    return relevance_blk


def test_reason():
    train_source = "./hh_data/1024_v2_shuffle/hh_train.pkl"
    dataset = SimpleListDataset(train_source)
    interface = SwagBlkPosInterface(dataset)
    dataset = interface.build_swag_promising_buffer(num_samples='1,1,1,1')
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=buffer_collate_for_reason, shuffle=False)
    reasoner = BertForMultipleChoice.from_pretrained("../pretrain_model/chinese-bert-wwm-ext")
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for batch in tqdm(dataloader):
        inputs, labels, pbufs, question_lens_tensor, choice_lens_tensor, passage_lens_tensor = batch
        # for blk_strses in blk_strses_list:
        #     for blk_strs in blk_strses:
        #         blk_str = ''.join(blk_strs)
        #         if blk_str.count("[CLS]") != 1 and blk_str.count("[SEP]") != 3:
        #             print("[CLS]:", blk_str.count("[CLS]"), "[SEP]:", blk_str.count("[SEP]"))
        logits = reasoner(*inputs).logits
        result = criterion(logits, labels)
        print(result)
        train_acc = torch.eq(logits.argmax(1), labels).sum().float().item() / labels.size(0)

        loss_reasoner = result.mean()

        # Label the relevance by the current reasoner
        _intervention(batch, result, reasoner, criterion)
        break


def _intervention(batch, result, reasoner, criterion):
    loss_reasoner = result.detach()
    with torch.no_grad():
        max_bs = 1 * 4
        inputs, labels, pbufs, question_lens_tensor, chioce_lens_tensor, passage_lens_tensor = batch
        input_ids, attention_mask, token_type_ids = inputs
        batch_size = input_ids.size(0)
        for b_idx in range(batch_size):
            # (4, 512)
            ids, attn_mask, type_ids = input_ids[b_idx], attention_mask[b_idx], token_type_ids[b_idx]
            pbuf, question_len, choice_len = pbufs[b_idx], question_lens_tensor[b_idx], chioce_lens_tensor[b_idx]
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
                print("blk:", blk)
                tmp_blk_length = len(blk) - 2
                for i in range(len(question_len)):
                    qc_length = question_len[i] + choice_len[i]
                    dblk_start = qc_length + dblk_length
                    dblk_end = dblk_start + tmp_blk_length
                    # print(input_ids[t, i, dblk_start:dblk_end])
                    print(blk.tokenizer.convert_tokens_to_string(
                        blk.tokenizer.convert_ids_to_tokens(ids[t, i, dblk_start:dblk_end])))
                    attn_mask[t, i, dblk_start:dblk_end].zero_()
                dblk_length += tmp_blk_length
                t += 1
            assert t == bs

            losses = []
            for j in range((bs - 1) // max_bs + 1):
                l, r = max_bs * j, min(bs, max_bs * (j + 1))
                # print(attn_masks[l:r])
                logits = reasoner(ids[l:r], attn_mask[l:r], type_ids[l:r]).logits
                result = criterion(logits, label[l:r].view(-1))
                losses.append(result)
            losses_delta = torch.cat(losses, dim=0) - loss_reasoner[b_idx]
            print(losses_delta)

            relevances = []
            # Label relevance
            t = 0
            for blk in pbuf.blocks:
                relevances.append(blk.relevance)
                if losses_delta[t] >= 0.2 and blk.relevance < 2:  # TODO topk
                    print('{} {}\n'.format(blk.pos, blk.relevance + 1))
                elif losses_delta[t] <= -0.05 and blk.relevance > -1:
                    print('{} {}\n'.format(blk.pos, blk.relevance - 1))
                t += 1
            assert t == bs


def test_bertsum_intro():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--ff_size", default=512, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--inter_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    intro_config = parser.parse_args()

    train_source = "./hh_data/1024_v2_shuffle/hh_train.pkl"
    dataset = SimpleListDataset(train_source)
    interface = SwagBlkPosInterface(dataset)
    dataset = interface.build_swag_random_buffer(num_samples='1,1,1,1')
    dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=buffer_collate_for_bertsumintro, shuffle=False)
    introspector = BertSumIntrospector.from_pretrained("../pretrain_model/chinese-bert-wwm-ext",
                                                       intro_config=intro_config)
    for batch in tqdm(dataloader):
        inputs, blk_ends_list, clses_list, labels_list, mask_cls, bufs = batch
        loss, logits = introspector(*inputs, labels_list, clses_list, mask_cls)
        for idx, buf in enumerate(bufs):
            print(loss)
            # print(logits[idx])
            print(logits[idx][[mask_cls[idx] == 1]])
        break


def test_memreplay():
    test_source = "./hh_data/1024_v2_shuffle/hh_test.pkl"
    device = f'cpu'

    parser = ArgumentParser(add_help=True)
    parser.add_argument("--ff_size", default=512, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--inter_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    intro_config = parser.parse_args()

    introspector = BertSumIntrospector.from_pretrained("../pretrain_model/chinese-bert-wwm-ext",
                                                       intro_config=intro_config).to(device).eval()
    reasoner = BertForMultipleChoice.from_pretrained("../pretrain_model/chinese-bert-wwm-ext").to(device).eval()
    qd_dataset = SimpleListDataset(test_source)

    with torch.no_grad():
        for qbuf, cbuf_list, dbuf, label in tqdm(qd_dataset):
            _id = qbuf[0]._id
            c_max_trans_length = 0
            for cbuf in cbuf_list:
                c_trans_length = cbuf.calc_size() - 2 * (len(cbuf) - 1) - 1
                c_max_trans_length = max(c_max_trans_length, c_trans_length)
            qc_max_trans_length = qbuf.calc_size() - 2 * (len(qbuf) - 1) + c_max_trans_length
            fill_buf = Buffer()
            pbuf, relevance_score = mem_replay(
                introspector,
                fill_buf,
                dbuf,
                times="3,5",
                device=device,
                qc_max_trans_length=qc_max_trans_length
            )  # TODO times hyperparam
            buf_list, qblk_num, cblk_num, pblk_num = [], [], [], []
            for cbuf in cbuf_list:
                qcp_buf = Buffer()
                qcp_buf.blocks = qbuf.blocks + cbuf.blocks + pbuf.blocks
                buf_list.append(qcp_buf)
                qblk_num.append(len(qbuf))
                cblk_num.append(len(cbuf))
                pblk_num.append(len(pbuf))
            inputs = torch.zeros(3, 1, 4, 512, dtype=torch.long, device=device)
            for i, buf in enumerate(buf_list):
                buf.export_reason(
                    qblk_num[i],
                    cblk_num[i],
                    pblk_num[i],
                    out=(
                        inputs[0, 0, i],
                        inputs[1, 0, i],
                        inputs[2, 0, i]
                    )
                )
            output = reasoner(*inputs)
            print(output)
            break


def test_attensum_intro():
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--ff_size", default=512, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--inter_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    intro_config = parser.parse_args()

    train_source = "./hh_data/1024_v2_shuffle/hh_train.pkl"
    dataset = SimpleListDataset(train_source)
    interface = SwagBlkPosInterface(dataset)
    dataset = interface.build_attensum_random_buffer(num_samples='1,1,1,1')
    dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=buffer_collate_for_attensumintro, shuffle=False)
    introspector = AttenSumIntrospector.from_pretrained("../pretrain_model/chinese-bert-wwm-ext",
                                                        intro_config=intro_config)
    for batch in tqdm(dataloader):
        inputs, labels, pbufs, question_lens, chioce_lens, passage_lens, clses, mask_cls = batch
        loss_introspector, logits = introspector(*inputs, labels, question_lens, chioce_lens, passage_lens, clses,
                                                 mask_cls)
        print(loss_introspector, logits)
        break


def test_attensum_memreplay():
    test_source = "./hh_data/1024_v2_shuffle/hh_test.pkl"
    device = f'cpu'

    parser = ArgumentParser(add_help=True)
    parser.add_argument("--ff_size", default=512, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--inter_layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    intro_config = parser.parse_args()

    introspector = AttenSumIntrospector.from_pretrained("../pretrain_model/chinese-bert-wwm-ext",
                                                        intro_config=intro_config).to(device).eval()
    reasoner = BertForMultipleChoice.from_pretrained("../pretrain_model/chinese-bert-wwm-ext").to(device).eval()
    qd_dataset = SimpleListDataset(test_source)

    with torch.no_grad():
        for qbuf, cbuf_list, dbuf, label in tqdm(qd_dataset):
            _id = qbuf[0]._id
            c_max_trans_length = 0
            for cbuf in cbuf_list:
                c_trans_length = cbuf.calc_size() - 2 * (len(cbuf) - 1)
                c_max_trans_length = max(c_max_trans_length, c_trans_length)
            qc_max_trans_length = qbuf.calc_size() - 2 * (len(qbuf) - 1) + c_max_trans_length
            fill_buf = Buffer()
            pbuf, relevance_score = attensum_mem_replay(
                introspector,
                fill_buf,
                qbuf,
                cbuf_list,
                dbuf,
                times="3,5",
                device=device,
                qc_max_trans_length=qc_max_trans_length
            )  # TODO times hyperparam
            buf_list, qblk_num, cblk_num, pblk_num = [], [], [], []
            for cbuf in cbuf_list:
                qcp_buf = Buffer()
                qcp_buf.blocks = qbuf.blocks + cbuf.blocks + pbuf.blocks
                buf_list.append(qcp_buf)
                qblk_num.append(len(qbuf))
                cblk_num.append(len(cbuf))
                pblk_num.append(len(pbuf))
            inputs = torch.zeros(3, 1, 4, 512, dtype=torch.long, device=device)
            for i, buf in enumerate(buf_list):
                buf.export_reason(
                    qblk_num[i],
                    cblk_num[i],
                    pblk_num[i],
                    out=(
                        inputs[0, 0, i],
                        inputs[1, 0, i],
                        inputs[2, 0, i]
                    )
                )
            output = reasoner(*inputs)
            print(output)
            break


def read_change():
    with open("./hh_data/compare_cuda_1.txt", "r", encoding="utf8") as file:
        lines = file.readlines()
        res = []
        all, up, down, relevance = 0, 0, 0, 0
        for line in lines:
            # if line == '\n':
            #     if len(res) > 0:
            #         all, up, down, relevance = 0, 0, 0, 0
            #         for item in res:
            #             data = json.loads(item)
            #             losses_delta = data["losses_delta"]
            #             relevances = data["relevances"]
            #             for idx, delta in enumerate(losses_delta):
            #                 all += 1
            #                 if delta > 0.2 and relevances[idx] < 2:
            #                     up += 1
            #                 if delta < -0.05 and relevances[idx] > -1:
            #                     down += 1
            #                 if relevances[idx] >= 1:
            #                     relevance += 1
            #         print(all, up, down, relevance)
            #         print(up / all, down / all, relevance / all)
            #     res = []
            # else:
            #     res.append(line)
            data = json.loads(line)
            losses_delta = data["losses_delta"]
            relevances = data["relevances"]
            for idx, delta in enumerate(losses_delta):
                all += 1
                if delta > 0.2 and relevances[idx] < 2:
                    up += 1
                if delta < -0.05 and relevances[idx] > -1:
                    down += 1
                if relevances[idx] >= 1:
                    relevance += 1
        print(all, up, down, relevance)
        print(up / all, down / all, relevance / all)


def test_data():
    train_source = "./hh_data/1024_v2_shuffle/hh_train.pkl"
    dataset = SimpleListDataset(train_source)
    interface = SwagBlkPosInterface(dataset)
    dataset = interface.build_swag_promising_buffer(num_samples='2,2,2,2')
    ori_ids = []
    with open("./all_dataset_rel.json", "a+", encoding="utf8") as file:
        for data in dataset:
            buf_list, s_buf, qblk_num, cblk_num, pblk_num, label = data
            res = {
                "pos": [],
                "rel": []
            }
            for blk in s_buf:
                res["pos"].append(blk.pos)
                res["rel"].append(blk.relevance)
            file.write(json.dumps(res, ensure_ascii=False) + '\n')
        file.close()
    # for qbuf, cbuf_list, dbuf, swag_label in dataset:
    #     for blk in dbuf:
    #         ori_ids.append(blk.pos)
    # interface = SwagBlkPosInterface(dataset)
    # dataset = interface.build_swag_random_buffer(num_samples='3,3,3,3')
    # select_ids = []
    # for data in dataset:
    #     for blk in data:
    #         if blk.pos in ori_ids:
    #             select_ids.append(blk.pos)
    # print("select_ids:", len(select_ids))
    # print("select_ids set:", len(set(select_ids)))
    # print("ori_ids:", len(ori_ids))
    # print("ori_ids set:", len(set(ori_ids)))
    # count = 0
    # for qbuf, cbuf_list, dbuf, swag_label in dataset:
    #     if len(dbuf) > 32:
    #         count += 1
    # print(count)
    # print(len(dataset))


def test_rel():
    train_source = "./hh_data/1024_v2_shuffle/hh_train.pkl"
    dataset = SimpleListDataset(train_source)
    ori_ids = []
    for qbuf, cbuf_list, dbuf, swag_label in dataset:
        for blk in dbuf:
            ori_ids.append(blk.pos)
    select_ids = []
    relevance_ids = []
    for i in range(4):
        with open("./hh_data/reason_dataset_rel_{}.json".format(str(i)), "r", encoding="utf8") as file:
            lines = file.readlines()
            all, relevance, un_relevance = 0, 0, 0
            for line in lines:
                data = json.loads(line)
                poses = data["pos"]
                rels = data["rel"]
                for idx, rel in enumerate(rels):
                    all += 1
                    if rel >= 1:
                        relevance += 1
                        relevance_ids.append(poses[idx])
                for pos in poses:
                    select_ids.append(pos)
            print(all, relevance, relevance / all)
    print(len(ori_ids), len(set(select_ids)), len(set(relevance_ids)))


if __name__ == '__main__':
    test_rel()
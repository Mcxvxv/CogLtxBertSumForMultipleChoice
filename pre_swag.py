import json
import logging
import os
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from buffer import Buffer
from data_helper import SimpleListDataset, find_lastest_checkpoint
from memreplay import mem_replay, attensum_mem_replay
from swag_bertsum_introspector_module import BertSumIntrospectorModule
from swag_introspector_module import SwagIntrospectorModule
from swag_reasoner_module import SwagReasonerModule

logging.basicConfig(level=logging.INFO)


def attensum_prediction(intro_version, reason_version):
    log_dir = os.path.join(os.getcwd(), 'log_dir')
    test_source = "./data/all/hh_test.pkl"
    device = f'cuda:2'

    intro_model = BertSumIntrospectorModule.load_from_checkpoint(
        os.path.join(log_dir, 'introspector', f'version_0', 'checkpoints',
                     'epoch=1-step=315-v{}.ckpt'.format(intro_version))).to(device).eval()
    reason_model = SwagReasonerModule.load_from_checkpoint(
        os.path.join(log_dir, 'reasoner', f'version_0', 'checkpoints',
                     'epoch=1-step=315-v{}.ckpt'.format(reason_version))).to(device).eval()
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
                intro_model.introspector,
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
            output = reason_model.reasoner(*inputs)
            yield _id, dbuf, buf, relevance_score, inputs[0][0], output, label


def prediction(intro_version, reason_version):
    log_dir = os.path.join(os.getcwd(), 'log_dir')
    test_source = "./data/all/hh_test.pkl"
    device = f'cuda:2'

    intro_model = BertSumIntrospectorModule.load_from_checkpoint(
        os.path.join(log_dir, 'introspector', f'version_0', 'checkpoints',
                     'epoch=1-step=315-v{}.ckpt'.format(intro_version))).to(device).eval()
    reason_model = SwagReasonerModule.load_from_checkpoint(
        os.path.join(log_dir, 'reasoner', f'version_0', 'checkpoints',
                     'epoch=1-step=315-v{}.ckpt'.format(reason_version))).to(device).eval()
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
                intro_model.introspector,
                fill_buf,
                dbuf,
                times="3,5",
                device=device,
                qc_max_trans_length=qc_max_trans_length
            )
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
            output = reason_model.reasoner(*inputs)
            yield _id, dbuf, buf, relevance_score, inputs[0][0], output, label


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--intro_start_version", type=int, default=0)
    parser.add_argument("--intro_end_version", type=int, default=8)
    parser.add_argument("--reason_start_version", type=int, default=0)
    parser.add_argument("--reason_end_version", type=int, default=8)
    config = parser.parse_args()

    acc_dir = os.path.join(os.getcwd(), 'acc_dir')

    intro_start_version = config.intro_start_version
    intro_end_version = config.intro_end_version
    reason_start_version = config.reason_start_version
    reason_end_version = config.reason_end_version
    for intro_version in range(intro_start_version, intro_end_version):
        for reason_version in range(reason_start_version, reason_end_version):
            logging.info(
                "start prediction with intro_version={intro_version}, reason_version={reason_version} ...".format(
                    intro_version=intro_version, reason_version=reason_version))
            ans, acc, total, acc_long, total_long = {}, 0., 0, 0., 0
            for _id, dbuf, buf, relevance_score, ids, output, label in prediction(intro_version, reason_version):
                # bs为1
                pred, gold = output.logits.view(-1).argmax().item(), int(label)
                ans[_id] = (pred, gold)
                total += 1.
                acc += pred == gold
                if dbuf.calc_size() + 2 > 512:
                    acc_long += pred == gold
                    total_long += 1
                    # if pred != gold:
                    #     import pdb; pdb.set_trace()
            acc /= total
            acc_long /= total_long
            w_data = {
                "intro_version": intro_version,
                "reason_version": reason_version,
                "total_long": total_long,
                "acc_long": acc_long,
                "acc": acc
            }
            with open(os.path.join(acc_dir, 'pre.json'), 'a+') as fout:
                fout.write(json.dumps(w_data, ensure_ascii=False) + '\n')
                fout.close()

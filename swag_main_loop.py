import os
import json
import logging
import pickle
from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import pdb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.loggers import TensorBoardLogger

from data_helper import SimpleListDataset, find_lastest_checkpoint, SwagBlkPosInterface
from swag_bertsum_introspector_module import BertSumIntrospectorModule
from swag_introspector_module import SwagIntrospectorModule
from memreplay import mem_replay
from initialize_relevance import init_relevance, init_relevance_for_swag
from swag_reasoner_module import SwagReasonerModule

logging.basicConfig(level=logging.INFO)


def main_loop(config):
    if len(config.gpus) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in config.gpus])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus[0])
    os.makedirs(config.tmp_dir, exist_ok=True)
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    train_dataset = SimpleListDataset(config.train_source)
    # val_dataset = SimpleListDataset(config.val_source)
    train_interface = SwagBlkPosInterface(train_dataset)
    # val_interface = SwagBlkPosInterface(val_dataset)
    logger_intro = TensorBoardLogger(config.log_dir, name='introspector', version=config.version)
    logger_reason = TensorBoardLogger(config.log_dir, name='reasoner', version=config.version)
    if config.init_relevance != '':
        if hasattr(config, 'conditional_transforms'):
            ct = config.conditional_transforms
            del config.conditional_transforms
        else:
            ct = []
        init_relevance_for_swag(train_dataset, method=config.init_relevance, conditional_transforms=ct)

    introspector = BertSumIntrospectorModule(config)
    # introspector = SwagIntrospectorModule(config)
    reasoner = SwagReasonerModule(config)
    pickle.dumps(introspector)
    pickle.dumps(reasoner)

    # introspector = SwagIntrospectorModule.load_from_checkpoint(
    #     os.path.join(config.log_dir, 'introspector', f'version_0', 'checkpoints', 'epoch=0-step=513-v7.ckpt'))
    # reasoner = SwagReasonerModule.load_from_checkpoint(
    #     os.path.join(config.log_dir, 'reasoner', f'version_0', 'checkpoints', 'epoch=0-step=513-v7.ckpt'))

    def _create_new_trainer(epoch, logger, accumulate_grad_batches, callbacks=None):
        return Trainer(
            max_epochs=epoch,
            gpus=len(config.gpus),
            accelerator='ddp_s',
            default_root_dir=config.log_dir,
            logger=logger,
            weights_summary=None,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
            replace_sampler_ddp=False
        )

    # min_epoch = min(find_lastest_checkpoint(
    #     os.path.join(config.log_dir, 'introspector', f'version_{config.version}', 'checkpoints'), epoch=True),
    #                 find_lastest_checkpoint(
    #                     os.path.join(config.save_dir, 'reasoner', f'version_{config.version}', 'checkpoints'),
    #                     epoch=True)) + 1
    # logging.info(f'Continue training at epoch {min_epoch}...')
    for wepoch in range(0, config.num_epochs):
        # 随机初始化rel
        train_intro_dataset = train_interface.build_swag_random_buffer(num_samples=config.num_samples)
        logging.info("set train swag intro_dataset...")
        introspector.set_dataset(train_intro_dataset, 'train')
        # val_intro_dataset = val_interface.build_swag_random_buffer(num_samples=config.num_samples)
        # logging.info("set val swag intro_dataset...")
        # introspector.set_dataset(val_intro_dataset, 'val')

        # checkpoint_callback = ModelCheckpoint(
        #     monitor='val_loss',
        #     save_top_k=1,
        #     mode='min',
        #     save_weights_only=False,
        #     filename='wepoch={wepoch}-'.format(wepoch=wepoch) + '{epoch}-{step}'
        # )
        trainer = _create_new_trainer(
            config.intro_trainer_num_epochs,
            logger_intro,
            config.intro_accumulate_grad_batches,
            # [checkpoint_callback]
        )
        trainer.fit(introspector)

        train_interface.collect_estimations_from_dir(config.tmp_dir)
        train_reason_dataset = train_interface.build_swag_promising_buffer(num_samples=config.num_samples)
        with open("./reason_dataset_rel_{}.json".format(str(wepoch)), "a+", encoding="utf8") as file:
            for data in train_reason_dataset:
                res = {
                    "pos": [],
                    "rel": []
                }
                buf_list, s_buf, qblk_num, cblk_num, pblk_num, label = data
                for blk in s_buf:
                    res["pos"].append(blk.pos)
                    res["rel"].append(blk.relevance)
                file.write(json.dumps(res, ensure_ascii=False) + '\n')
            file.close()
        logging.info("set train swag reason_dataset...")
        reasoner.set_dataset(train_reason_dataset, 'train')
        # val_reason_dataset = val_interface.build_swag_promising_buffer(num_samples=config.num_samples)
        # logging.info("set val swag reason_dataset...")
        # reasoner.set_dataset(val_reason_dataset, 'val')

        # checkpoint_callback = ModelCheckpoint(
        #     monitor='val_acc',
        #     save_top_k=1,
        #     mode='max',
        #     save_weights_only=False,
        #     filename='wepoch={wepoch}-'.format(wepoch=wepoch) + '{epoch}-{step}'
        # )
        trainer = _create_new_trainer(
            config.reason_trainer_num_epochs,
            logger_reason,
            config.reason_accumulate_grad_batches,
            # [checkpoint_callback]
        )
        trainer.fit(reasoner)
        if config.latent:
            logging.info("apply_changes_from_dir...")
            train_interface.apply_changes_from_dir(config.tmp_dir)

        with open("./all_dataset_rel_{}.json".format(str(wepoch)), "a+", encoding="utf8") as file:
            for data in train_dataset:
                qbuf, cbuf_list, dbuf, label = data
                res = {
                    "pos": [],
                    "rel": []
                }
                for blk in dbuf:
                    res["pos"].append(blk.pos)
                    res["rel"].append(blk.relevance)
                file.write(json.dumps(res, ensure_ascii=False) + '\n')
            file.close()


def main_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()

    parser.add_argument("--save_dir", type=str, default=os.path.join(os.getcwd(), 'save_dir'), help="saving models")
    parser.add_argument("--tmp_dir", type=str, default=os.path.join(os.getcwd(), 'tmp_dir'),
                        help="saving ddp tmp files")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), 'log_dir'), help="saving logs")
    parser.add_argument("--num_epochs", type=int, default=2, help="num epoch")
    parser.add_argument("--intro_trainer_num_epochs", type=int, default=2, help="intro num epoch")
    parser.add_argument("--reason_trainer_num_epochs", type=int, default=2, help="reasoner num epoch")
    parser.add_argument('--model_name', type=str, default='./ft_model/chinese-roberta-wwm-ext',
                        help='name of pretrained models')
    parser.add_argument('--version', type=int, default=0, help='the version to save or restore')
    parser.add_argument('--num_workers', type=int, default=16, help='num of num_workers')

    parser.add_argument('--num_samples', type=str, default='1,1,1,1',
                        help='num of continous, discrete random samples and promising samples')
    parser.add_argument('--times', type=str, default='3,5', help='memreplay times')

    parser.add_argument('--batch_size_inference', type=int, default=8, help='batch_size in memreplay')

    parser.add_argument('--latent', action='store_true', help='without relevance labels')
    parser.add_argument('--init_relevance', type=str, default='', help='bm25 or glove')

    parser.add_argument("--gpus", type=int, nargs='+', default=0, help="available gpus")
    parser.add_argument('--train_source', type=str, help='training dataset')
    parser.add_argument('--val_source', type=str, help='validation dataset')
    parser.add_argument('--test_source', type=str, help='test dataset')
    BertSumIntrospectorModule.add_specific_args(parser)
    SwagReasonerModule.add_specific_args(parser)
    return parser


if __name__ == '__main__':
    # print(find_lastest_checkpoint(os.path.join("log_dir", 'introspector', f'version_0', 'checkpoints')))
    intro_model = SwagIntrospectorModule.load_from_checkpoint(
        find_lastest_checkpoint(os.path.join("log_dir", 'introspector', f'version_0', 'checkpoints')))

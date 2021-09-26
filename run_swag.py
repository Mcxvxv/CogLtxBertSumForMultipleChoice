import logging
from argparse import ArgumentParser
import os
import torch
import pdb
import json
from copy import copy
from transformers import AutoTokenizer, BertTokenizer

from swag_main_loop import main_loop, main_parser
from buffer import Buffer
from utils import CAPACITY

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info('=====================================')
    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = ArgumentParser(add_help=False)
    # ------------ add dataset-specific argument ----------
    parser.add_argument('--reasoner_config_num_labels', type=int, default=4)
    parser.add_argument('--only_predict', action='store_true')
    # ---------------------------------------------
    parser = main_parser(parser)
    parser.set_defaults(
        train_source="./data/hh_train.pkl",
        val_source="./data/hh_val.pkl",
        test_source="./data/hh_test.pkl"
    )
    config = parser.parse_args()
    for k in list(vars(config).keys()):
        logging.info('%s: %s' % (k, vars(config)[k]))

    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    main_loop(config)

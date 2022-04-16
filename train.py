"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           train.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

from utils.module import ModelManager
from utils.loader import DatasetManager
from utils.process import Processor

import torch

import os
import json
import random
import argparse
import numpy as np
import logging
#TODO 
# 1. Add bert embedding
# 2. BIO BMESO
#

parser = argparse.ArgumentParser()

# Training parameters.
parser.add_argument('--data_dir', '-dd', type=str, default='data/crosswoz')
parser.add_argument('--save_dir', '-sd', type=str, default='save')
parser.add_argument("--random_state", '-rs', type=int, default=0)
parser.add_argument('--num_epoch', '-ne', type=int, default=100)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--l2_penalty', '-lp', type=float, default=1e-6)
parser.add_argument("--learning_rate", '-lr', type=float, default=0.001)
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.4)
parser.add_argument('--intent_forcing_rate', '-ifr', type=float, default=0.9)
parser.add_argument("--differentiable", "-d",
                    action="store_true", default=False)
parser.add_argument('--slot_forcing_rate', '-sfr', type=float, default=0.9)
parser.add_argument('--bio_forcing_rate', '-bfr', type=float, default=0.9)
# model parameters.
parser.add_argument('--word_embedding_dim', '-wed', type=int, default=64)
parser.add_argument('--encoder_hidden_dim', '-ehd', type=int, default=256)
parser.add_argument('--intent_embedding_dim', '-ied', type=int, default=8)
parser.add_argument('--slot_embedding_dim', '-sed', type=int, default=32)
parser.add_argument('--bio_embedding_dim', '-bed', type=int, default=8)

parser.add_argument('--slot_decoder_hidden_dim', '-sdhd', type=int, default=64)
parser.add_argument('--intent_decoder_hidden_dim',
                    '-idhd', type=int, default=64)
parser.add_argument('--bio_decoder_hidden_dim', '-bdhd', type=int, default=64)

parser.add_argument('--attention_hidden_dim', '-ahd', type=int, default=1024)
parser.add_argument('--attention_output_dim', '-aod', type=int, default=128)
parser.add_argument('--index', type=int, default=0)
parser.add_argument("--template_encoder", "-tr",
                    action="store_true", default=False)
parser.add_argument("--template_share_encoder", "-tsr",
                    action="store_true", default=False)
parser.add_argument("--bio_schame", "-bse", type=str, default='bio')
parser.add_argument('--use_bert', '-ub', action="store_true", default=False)
parser.add_argument('--bert_name', '--bn',  type=str, default='bert-base-chinese')

if __name__ == "__main__":
    args = parser.parse_args()
    # 创建一个logger
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    if not os.path.exists(args.save_dir):
        os.system("mkdir -p " + args.save_dir)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(args.save_dir + '/' + 'test.log', mode='w')
    fh.setLevel(logging.DEBUG)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter(
        '[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    # Save training and model parameters.

    log_path = os.path.join(args.save_dir, "param.json")
    with open(log_path, "w") as fw:
        fw.write(json.dumps(args.__dict__, indent=True))

    # Fix the random seed of package random.
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # Fix the random seed of Pytorch when using GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
        torch.cuda.manual_seed(args.random_state)

    # Fix the random seed of Pytorch when using CPU.
    torch.manual_seed(args.random_state)
    torch.random.manual_seed(args.random_state)

    # Instantiate a dataset object.
    dataset = DatasetManager(args)
    dataset.quick_build()
    dataset.show_summary()

    print()

    # Instantiate a network model object.
    model = ModelManager(
        args, len(dataset.word_alphabet),
        len(dataset.slot_alphabet),
        len(dataset.intent_alphabet),
        len(dataset.bio_alphabet),
        dataset.bio_alphabet
    )
    model.show_summary()
    logger.info(args)
    logger.info(model)

    # To train and evaluate the models.
    process = Processor(dataset, model, args.batch_size, logger)
    process.train()

    logger.info('\nAccepted performance: ' + str(Processor.validate(
        os.path.join(args.save_dir, "model/model.pkl"),
        os.path.join(args.save_dir, "model/dataset.pkl"),
        args.batch_size, index=args.index)) + " at test dataset;\n")

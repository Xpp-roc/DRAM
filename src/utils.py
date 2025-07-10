import json
import logging
import datetime
import os
import sys

import torch
def print_msg(msg):
    msg = "## {} ##".format(msg)
    length = len(msg)
    msg = "\n{}\n".format(msg)
    print(length*"#" + msg + length * "#")

import torch 
def cross_entropy(y,y_pre,weight):
    res = weight * (y*torch.log(y_pre))
    loss=-torch.sum(res)
    return loss/y_pre.shape[0]

def load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_config(args, cfg, cli_overwrite=True):
    for k, v in cfg.items():
        if not hasattr(args, k):
            setattr(args, k, v)
        else:
            if cli_overwrite:
                default_val = args.__parser__.get_default(k)
                if getattr(args, k) == default_val:
                    setattr(args, k, v)
            else:
                setattr(args, k, v)
def build_exp_name(args):
    return (
        f"[epochs{args.n_epoch}]"
        f"[alpha{args.alpha}]"
        f"[anchor{args.anchor}]"
        f"[max_sample{args.max_path_len}]"
        f"[emb{args.emb_size}]"
        f"[lr{args.lr}]"
        f"[batchsize{args.batch_size}]"
    )

def prepare_exp_dir(dataset: str, exp_name: str) -> str:
    path = os.path.join("..", "record", dataset, exp_name)
    os.makedirs(path, exist_ok=True)
    return path

def set_logger(save_path):
    log_file = os.path.join(save_path, 'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def set_logger_with_stdout_redirect(save_path):
    log_file = os.path.join(save_path, 'run.log')
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    class StreamToLogger:
        def __init__(self, level_fn):
            self.level_fn = level_fn

        def write(self, message):
            if message.strip():
                self.level_fn(message.strip())

        def flush(self):
            pass
    sys.stdout = StreamToLogger(logging.info)
    sys.stderr = StreamToLogger(logging.error)

import time, torch
import logging, os, sys
import numpy as np

def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def init_logging(log_root, models_root, rank=0):
    if rank == 0:
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s-%(message)s", "%Y-%m-%d,%H:%M:%S")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)

def load_syn_dataset(exp_root="../snapshots", exp_dir="exp1", file_name="syn_data.pt", ):
    syn_data_path = os.path.join(exp_root, exp_dir, file_name)
    saved_data = torch.load(syn_data_path)
    data_syn, label_syn = saved_data["syn_dataset"][0]
    return data_syn, label_syn



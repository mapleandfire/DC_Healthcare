from dataset.data_loaders import get_loaders_mimic3, get_loader_physio, get_loaders_covid
from easydict import EasyDict as edict
import torch
import numpy as np

# MMIC-III IHM dataset
mimic3 = edict(
    {
        "data_root": "../datasets/mimic3",
        "fea_dim":60,
        "time_dim": 48,
        "num_classes":2,
        "train_sample": 14698,
        "val_sample": 3222,
        "test_sample": 3236,
        "data_loader_fn": get_loaders_mimic3,
        "label_dtype": torch.long,
    }
)

# the feature dimensions without duplications for mimic3
mimic3_fea_dim_no_dup = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 16, 17, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39,
     44, 40, 41] + list(range(49, 76)))

physio = edict(
    {
        "data_root": "../datasets/PhysioNet",
        "fea_dim": 47,
        "time_dim": 48,
        "num_classes": 2,
        "train_sample": 5120,
        "val_sample": 1280,
        "test_sample": 1600,
        "data_loader_fn": get_loader_physio,
        "label_dtype": torch.float32,
    }
)

### Coswara
covid_b = edict(
    {
        "data_root": "../datasets/covid/breath",
        "fea_dim": 64,
        "time_dim": 96,
        "num_classes": 2,
        "train_sample": 987,
        "val_sample": 175,
        "test_sample": 206,
        "data_loader_fn": get_loaders_covid,
        "label_dtype": torch.float32,
    }
)


dataset_info = edict(
    {
        "mimic3":mimic3,
        "physio":physio,
        "covid_b":covid_b,
    }
)

ds_name_mapping = {
    "mimic3": "MIMIC-III",
    "physio": "PhysioNet-2012",
    "covid_b": "Coswara",
}

net_name_mapping = {
    "tcn": "TCN-α",
    "tcn2": "TCN-β",
    "tcn3": "TCN-γ",
    "lstm": "LSTM-α",
    "lstm2": "LSTM-β",
    "vit": "ViT-α",
    "vit2": "ViT-β",
    "trsf": "TRSF-α",
    "trsf2": "TRSF-β",
    "rnn": "RNN-α",
    "rnn2": "RNN-β",
}
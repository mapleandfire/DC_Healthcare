from easydict import EasyDict as edict
import argparse, os
from dataset.meta import dataset_info

def get_args(strict=True):
    parser = argparse.ArgumentParser(description='Training configs')
    parser.add_argument('--dataset', "--ds", '-d', type=str, default="physio", help='the dataset to use, mimic3/physio/covid_b')
    parser.add_argument('--net', '-n', type=str, default="tcn", help='the network to use, tcn/trsf/lstm')
    parser.add_argument('--pre_process', '--prpr', type=str, default="none",
                        help='the pre-processing method, std/minmax/none')
    parser.add_argument('--save_root', type=str, default="../snapshots", help='save root path')
    parser.add_argument('--save_dir_name', '-s', type=str, default="exp1",
                        help='name of the saving directory under save_root')
    parser.add_argument('--device', type=str, default="cuda:0", help='the torch device')
    parser.add_argument('--syn_data_path', "--syn", type=str, default=None,
                        help='The path to saved condensed data. If specified, will use it as training data ')

    parser.add_argument('--val_metric', type=str, default="auc", help='validation metric, auc or loss')

    parser.add_argument('--early_stop', '--es', type=int, default=None,
                        help='The early stop tolerance (epochs), default: None (no early stop)')
    parser.add_argument('--early_stop_metric', '--es_metric', type=str, default="loss",
                        help='early stop metric, auc or loss')

    parser.add_argument('--train_batch', type=int, default=None, help='the training batch')
    parser.add_argument('--lr', type=float, default=None, help='the learning rate')
    parser.add_argument('--epochs', type=int, default=None, help="the training epochs")
    parser.add_argument('--weight_decay', "--wd", type=float, default=0.0, help='the weight decay')

    parser.add_argument('--enable_tensorboard', type=int, default=True,
                        help="whether to enable tensorboard for training logs")

    ##### Distribution Matching parameters
    parser.add_argument('--dm_ipc', type=int, default=40, help='instances per class for DC')
    parser.add_argument('--dm_train_net', "--trnet", type=str, default="tcn,lstm,vit",
                        help='the networks to train condensed data (seperated by commas), e.g. \"tcn,lstm\"')
    parser.add_argument('--dm_eval_net', "--evnet", type=str, default="tcn,lstm,vit,vit2,trsf,trsf2,tcn2,tcn3,lstm2,rnn,rnn2",
                        help='the networks to evaluate condensed data (seperated by commas), e.g. \"tcn,lstm\"')
    parser.add_argument('--dm_num_eval', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--dm_iteration', "--sp", type=int, default=24000, help='training iterations')
    parser.add_argument('--dm_eval_iteration', "--eval_sp", type=int, default=24000, help='iteration for evaluating condensed dataset')
    parser.add_argument('--dm_lr_data', type=float, default=0.001, help='learning rate for updating condensed data')
    parser.add_argument('--dm_batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--dm_syn_time_dim',"--stime_dim", type=int, default=None,
                        help='the temporal dimension of the condensed dataset')
    parser.add_argument('--dm_init', type=str, default='noise', help='noise/real: initialize condensed images from random noise or randomly sampled real images.')
    parser.add_argument('--dm_logging_iter', type=int, default=1000, help='the iterations to display training loss')

    parser.add_argument('--dm_combine_val', type=int, default=0,
                        help='whether to combine validation to train set to learn condensed dataset')
    parser.add_argument('--dm_ignore_init_eval', type=int, default=1,
                        help='whether to ignore the evaluation at the beginning')

    parser.add_argument('--ds_ori', type=str, default="mimic3",
                        help='the original dataset in transfer learning')
    parser.add_argument('--prpr_ori', type=str, default="none",
                        help='the pre-processing method for original dataset in transfer learning')

    parser.add_argument('--lr_ori', type=float, default=None, help='the learning rate on the original dataset')
    parser.add_argument('--epochs_ori', type=int, default=None, help="the training epochs on the original dataset")

    parser.add_argument('--val_every_step', type=int, default=100,
                        help="do validation for every steps, only used for plotting training curve")

    if strict:
        args = parser.parse_args()
    else:
        args = parser.parse_known_args()[0]
    return args


def get_config(args):

    cf = edict()
    cf.net=args.net.lower().strip()

    cf.save_root=args.save_root
    cf.save_dir_name=args.save_dir_name
    cf.save_dir = os.path.join(cf.save_root, cf.save_dir_name)
    cf.device=args.device
    cf.pre_process = args.pre_process.strip().lower()

    ds_name = args.dataset.strip().lower()
    cf.ds_name = ds_name
    ds_info = dataset_info[cf.ds_name]

    cf.data_root=ds_info.data_root
    cf.fea_dim=ds_info.fea_dim
    true_cls = ds_info.num_classes
    cf.num_class=1 if true_cls==2 else true_cls
    cf.data_loader_fn = ds_info.data_loader_fn
    cf.label_dtype = ds_info.label_dtype

    cf.syn_data_path = args.syn_data_path.strip() if args.syn_data_path else None

    cf.val_metric=args.val_metric
    cf.enable_tensorboard = args.enable_tensorboard

    ### Distribution Matching configs
    dm = edict()
    dm.ipc = args.dm_ipc

    dm.train_net = tuple(n for n in args.dm_train_net.strip().split(","))  # the network(s) to learn the condensed dataset
    dm.eval_net = tuple(n for n in args.dm_eval_net.strip().split(","))  # the network(s) to eval the condensed dataset

    # dm.num_exp = args.dm_num_exp
    dm.num_eval = args.dm_num_eval
    dm.iteration = args.dm_iteration
    dm.lr_data = args.dm_lr_data
    dm.batch_real = args.dm_batch_real
    dm.init = args.dm_init
    dm.eval_iteration = args.dm_eval_iteration
    # dm.optmizer = args.dm_opt

    dm.combine_val = args.dm_combine_val
    dm.ignore_init_eval = args.dm_ignore_init_eval

    if args.dm_syn_time_dim is not None:
        dm.syn_time_dim = args.dm_syn_time_dim
    elif ds_info.time_dim is not None:
        dm.syn_time_dim = ds_info.time_dim
    else:
        print("Condensed dataset's temporal dimension is not provided. Setting it to default value (48)")
        dm.syn_time_dim = 48
    dm.logging_iter = args.dm_logging_iter
    # dm.bp_steps = args.dm_bp_steps

    cf.dm = dm

    cf.train_batch = args.train_batch or 64
    cf.weight_decay = args.weight_decay

    if cf.ds_name == "mimic3":
        cf.lr = args.lr or 0.00005
        cf.epochs = args.epochs or 20
    elif cf.ds_name == "physio":
        cf.lr = args.lr or 0.0005
        cf.epochs = args.epochs or 30
    elif cf.ds_name == "pheno":
        cf.lr = args.lr or 0.001
        cf.epochs = args.epochs or 40
    elif cf.ds_name == "covid_b" or cf.ds_name=="covid_c":
        cf.lr = args.lr or 0.0001
        cf.epochs = args.epochs or 30
    else:
        raise NotImplementedError(
                "Training parameters for dataset: {} are not pre-defined ".format(cf.ds_name))

    cf.early_stop = args.early_stop
    cf.early_stop_metric = args.early_stop_metric

    # transfer learning parameters
    tl = edict()
    tl.ds_ori = args.ds_ori.strip().lower()
    tl.prpr_ori = args.prpr_ori.strip().lower()

    tl.lr_ori = args.lr_ori or 0.00005
    tl.epochs_ori = args.epochs_ori or 20


    cf.tl = tl
    ######### TCN parameters  ######
    tcn_options = {
        "relu_type": "prelu",
        "dropout": 0.5,
        "dwpw": False,
        "kernel_size": [3,5,7],  #ms-tcn
        "num_layers": 2,
        "width_mult": 1,
        "hidden_dim": 64,
    }
    num_channels = [tcn_options['hidden_dim'] * len(tcn_options['kernel_size']) * tcn_options['width_mult']] * tcn_options['num_layers']
    cf.tcn_args = {
        "input_size": cf.fea_dim,
        "num_channels": num_channels,
        "num_classes": cf.num_class,
        "tcn_options": tcn_options,
        "dropout": tcn_options['dropout'],
        "relu_type": tcn_options['relu_type'],
        "dwpw": tcn_options['dwpw'],
    }

    tcn2_options = {
        "relu_type": "prelu",
        "dropout": 0.5,
        "dwpw": False,
        "kernel_size": [3],   # single TCN
        "num_layers": 2,
        "width_mult": 1,
        "hidden_dim": 64,
    }
    num_channels_2 = [tcn2_options['hidden_dim'] * len(tcn2_options['kernel_size']) * tcn2_options['width_mult']] * tcn2_options['num_layers']
    cf.tcn2_args = {
        "input_size": cf.fea_dim,
        "num_channels": num_channels_2,
        "num_classes": cf.num_class,
        "tcn_options": tcn2_options,
        "dropout": tcn2_options['dropout'],
        "relu_type": tcn2_options['relu_type'],
        "dwpw": tcn2_options['dwpw'],
    }

    tcn3_options = {
        "relu_type": "prelu",
        "dropout": 0.5,
        "dwpw": False,
        "kernel_size": [3,5],   # ms-tcn
        "num_layers": 2,
        "width_mult": 1,
        "hidden_dim": 128,
    }
    num_channels_3 = [tcn3_options['hidden_dim'] * len(tcn3_options['kernel_size']) * tcn3_options['width_mult']] * tcn3_options['num_layers']
    cf.tcn3_args = {
        "input_size": cf.fea_dim,
        "num_channels": num_channels_3,
        "num_classes": cf.num_class,
        "tcn_options": tcn3_options,
        "dropout": tcn3_options['dropout'],
        "relu_type": tcn3_options['relu_type'],
        "dwpw": tcn3_options['dwpw'],
    }

    ######### ViT parameters  ######
    cf.vit_args = {
        "num_classes": cf.num_class,
        "dim": cf.fea_dim,
        "depth": 4,
        "use_class_token": True,   # ViT are just transformers that use class token as video representations
        "heads":16,  # head number of transformer
        "ff_dim":16,  # MLP dimension of transformer's feedforward layer
        "mlp_head_hidden_dim":[128],   # the hidden layer dimensions of the MLP head
        "dim_head":64,  # head dimension of transformer's attention module
        "pool":"cls",  # "cls" or "mean"
        "dropout":0.5,  # dropout rate of transformer
        "mlp_head_dropout":0.5,   # dropout rate of the MLP head
        "pe_method":None,  # the Positional Embedding method
        "pe_max_len":48,  # the maximum sequence length for Positional Embedding
        "activation":"gelu",   # the activation method, can be "gelu" "prelu" or "relu"
    }

    cf.vit2_args = {
        "num_classes": cf.num_class,
        "dim": cf.fea_dim,
        "depth": 4,
        "use_class_token": True,   # ViT are just transformers that use class token as video representations
        "heads":4,  # head number of transformer
        "ff_dim":16,  # MLP dimension of transformer's feedforward layer
        "mlp_head_hidden_dim":[128],   # the hidden layer dimensions of the MLP head
        "dim_head":256,  # head dimension of transformer's attention module
        "pool":"cls",  # "cls" or "mean"
        "dropout":0.5,  # dropout rate of transformer
        "mlp_head_dropout":0.5,   # dropout rate of the MLP head
        "pe_method":None,  # the Positional Embedding method
        "pe_max_len":48,  # the maximum sequence length for Positional Embedding
        "activation":"gelu",   # the activation method, can be "gelu" "prelu" or "relu"
    }

    ######### Transformer parameters  ######
    cf.trsf_args = {
        "num_classes": cf.num_class,
        "dim": cf.fea_dim,
        "depth": 2,
        "use_class_token": False,   # set "use_class_token" to get usual transformer encoder
        "heads":16,  # head number of transformer
        "ff_dim":16,  # MLP dimension of transformer's feedforward layer
        "mlp_head_hidden_dim":[64],   # the hidden layer dimensions of the MLP head
        "dim_head":64,  # head dimension of transformer's attention module
        "pool":"cls",  # "cls" or "mean"
        "dropout":0.5,  # dropout rate of transformer
        "mlp_head_dropout":0.5,   # dropout rate of the MLP head
        "pe_method":None,  # the Positional Embedding method
        "pe_max_len":48,  # the maximum sequence length for Positional Embedding
        "activation":"gelu",   # the activation method, can be "gelu" "prelu" or "relu"
    }

    cf.trsf2_args = {
        "num_classes": cf.num_class,
        "dim": cf.fea_dim,
        "depth": 2,
        "use_class_token": False,   # set "use_class_token" to get usual transformer encoder
        "heads":4,  # head number of transformer
        "ff_dim":16,  # MLP dimension of transformer's feedforward layer
        "mlp_head_hidden_dim":[128],   # the hidden layer dimensions of the MLP head
        "dim_head":256,  # head dimension of transformer's attention module
        "pool":"cls",  # "cls" or "mean"
        "dropout":0.5,  # dropout rate of transformer
        "mlp_head_dropout":0.5,   # dropout rate of the MLP head
        "pe_method":None,  # the Positional Embedding method
        "pe_max_len":48,  # the maximum sequence length for Positional Embedding
        "activation":"gelu",   # the activation method, can be "gelu" "prelu" or "relu"
    }

    ######### LSTM parameters  ######
    cf.lstm_args={
        "input_dim":cf.fea_dim,
        "hidden_dim":256,
        "num_classes":cf.num_class,
        "device":cf.device,
    }

    cf.lstm2_args={
        "input_dim":cf.fea_dim,
        "hidden_dim":128,
        "num_classes":cf.num_class,
        "device":cf.device,
    }

    ### RNN parameters
    cf.rnn_args={
        "input_dim":cf.fea_dim,
        "hidden_dim":256,
        "num_classes":cf.num_class,
        "device":cf.device,
    }

    cf.rnn2_args={
        "input_dim":cf.fea_dim,
        "hidden_dim":128,
        "num_classes":cf.num_class,
        "device":cf.device,
    }

    return cf


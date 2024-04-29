##### Dataset condensation for healthcare dataset
import numpy as np
from dataset.data_loaders import create_data_loader, build_data_getter
import torch
import os
from configs.config_DM import get_args, get_config
from utils.train_utils import get_net, eval_net
from utils.misc import init_logging
import copy
from utils.metric_tracker import MetricTracker, TensorboardWriter
import logging


def main():

    this_args = get_args()

    cf=get_config(this_args)

    os.makedirs(cf.save_dir, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root, cf.save_dir)

    logging.info("Training data will be saved to: {}".format(cf.save_dir))
    logging.info("Args: {}".format(this_args))
    logging.info("Configs:")
    for k, v in cf.items():
        if not "_args" in k:
            logging.info("  {}: {}".format(k, v))

    # the real validation/test data loader
    logging.info("Creating real validation/test data loader for dataset: {}".format(cf.ds_name))
    _, val_loader, test_loader, tr_data, tr_lb, prpr = cf.data_loader_fn(
        path=cf.data_root, train_batch=cf.train_batch, val_batch=128, test_batch=128, pre_process=cf.pre_process)

    # which iterations to evaluate
    eval_it_pool = list(range(0, cf.dm.iteration+1, cf.dm.eval_iteration))
    if cf.dm.iteration not in eval_it_pool:
        eval_it_pool.append(cf.dm.iteration)  # evaluate at the end of training
    if cf.dm.ignore_init_eval:
        _=eval_it_pool.pop(0)   # ignore evaluation at initialization

    num_classes = 2 if cf.num_class==1 else cf.num_class  # find class number
    # build the real train data getter: get random n data from class c
    get_data = build_data_getter(cf.ds_name, tr_data, tr_lb, cf.device)

    for c in range(num_classes):
        logging.info('Class {}: {} real samples'.format(c, len(get_data.indices_class[c])))

    syn_shape=(num_classes * cf.dm.ipc, cf.dm.syn_time_dim, cf.fea_dim,)

    logging.info("Real train dataset shape: {} ".format(tr_data.shape))
    logging.info("Condensed dataset shape: {} ".format(syn_shape))
    # initialise the condensed dataset
    data_syn = torch.randn(
        size=syn_shape, dtype=torch.float, requires_grad=True, device=cf.device)

    if cf.ds_name == "mimic3" or cf.ds_name=="physio" or cf.ds_name == "covid_b":
        label_syn = np.asarray([np.ones(cf.dm.ipc) * i for i in range(num_classes)])  # [0,0,0, ..., 1,1,1, ]
        label_syn = torch.tensor(label_syn, dtype=cf.label_dtype, requires_grad=False, device=cf.device).view(-1)
    else:
        raise NotImplementedError("Dataset {} not implemented".format(cf.ds_name))

    if cf.dm.init == 'real':
        logging.info('Initialize condensed data from random real data')
        for c in range(num_classes):
            data_syn.data[c * cf.dm.ipc:(c + 1) * cf.dm.ipc] = get_data(c, cf.dm.ipc).detach().data
    else:
        logging.info('Initialize condensed data from random noise')

    logging.info("Using Adam optimizer for DC ...")
    optimizer_data = torch.optim.Adam([data_syn, ], lr=cf.dm.lr_data)

    logging.info("Learning condensed dataset with network(s): {}".format(cf.dm.train_net))
    logging.info("Evaluating condensed dataset on network(s): {}".format(cf.dm.eval_net))

    # tensorboard writer
    writer = TensorboardWriter(cf.save_dir, cf.enable_tensorboard)
    train_metrics = MetricTracker("mmd_loss",writer=writer)
    eval_keys = tuple("syn_auc ({})".format(n) for n in cf.dm.eval_net)
    eval_metric = MetricTracker(*eval_keys, writer=writer)

    final_test_auc = dict()
    for this_net in cf.dm.eval_net:
        final_test_auc[this_net]=[]

    all_test_auc = dict()

    logging.info('Training starts ...')
    optimizer_data.zero_grad()

    for it in range(cf.dm.iteration + 1):

        #### Evaluate condensed data ####
        if it in eval_it_pool:
            aucs = dict()
            for this_net in cf.dm.eval_net:  # iterates through all networks to evaluate condensed dataset
                logging.info("Iter {}, evaluating condensed dataset on network: {} ...".format(it, this_net))
                aucs[this_net] = []

                for it_eval in range(cf.dm.num_eval):
                    net_eval = get_net(this_net, **cf[this_net+"_args"]).to(cf.device)  # get a random model
                    # avoid any unaware modification
                    data_syn_eval, label_syn_eval = \
                        copy.deepcopy(data_syn.detach()), copy.deepcopy(label_syn.detach())

                    # create a data loader from condensed data; also disable the sampler
                    syn_train_loader = create_data_loader(
                        data_syn_eval, label_syn_eval, batch_size=cf.train_batch, sampler=False)

                    _, test_auc = eval_net(
                        net_eval, syn_train_loader, val_loader, test_loader,
                        lr=cf.lr, epochs=cf.epochs, weight_decay=cf.weight_decay,
                        save_dir=cf.save_dir, val_metric=cf.val_metric, device=cf.device,
                        early_stop=cf.early_stop, early_stop_metric=cf.early_stop_metric,
                    )

                    logging.info("eval {:02d}/{:02d}, auc: {:.4f}".format(it_eval+1, cf.dm.num_eval, test_auc), )
                    aucs[this_net].append(test_auc)

                logging.info("Iter {}, network: {}, condensed AUC: {:.4f}, std: {:.4f}".format(
                    it, this_net, np.mean(aucs[this_net]), np.std(aucs[this_net])))

                if eval_metric.writer is not None:
                    eval_metric.writer.set_step(it, mode="eval")
                eval_metric.update("syn_auc ({})".format(this_net), np.mean(aucs[this_net]))

            all_test_auc["iter_" + str(it)] = aucs
            if it == cf.dm.iteration:  # record the final results
                for this_net in cf.dm.eval_net:
                    final_test_auc[this_net]+=aucs[this_net]

        #### Train condensed data ####
        tr_net = np.random.choice(cf.dm.train_net)   # randomly pick a network from train network candidates
        net = get_net(tr_net, **cf[tr_net+"_args"]).to(cf.device)   # get a random model
        net.train()
        for param in list(net.parameters()):
            param.requires_grad = False
        loss_avg = 0

        # compute MMD loss
        loss = torch.tensor(0.0).to(cf.device)
        for _, c in enumerate(range(num_classes)):
            # the batch size should not exceed total samples of this class
            this_batch_real = min(len(get_data.indices_class[c]), cf.dm.batch_real)
            batch_data_real = get_data(c, this_batch_real)
            batch_data_syn = data_syn[c * cf.dm.ipc : (c + 1) * cf.dm.ipc]

            output_real = net(batch_data_real).detach()
            output_syn = net(batch_data_syn)

            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

        # update condensed data
        optimizer_data.zero_grad()
        loss.backward()
        optimizer_data.step()
        loss_avg += loss.item()
        loss_avg /= (num_classes)
        if train_metrics.writer is not None:
            train_metrics.writer.set_step(it)
        train_metrics.update("mmd_loss", loss_avg)

        if it % cf.dm.logging_iter == 0:
            logging.info('iter = {:04d}, MMD loss = {:.7f}'.format(it, loss_avg))

    for this_net in cf.dm.eval_net:
        logging.info("Final condensed AUC ({}): {:.4f}, std: {:.4f}".format(
            this_net, np.mean(final_test_auc[this_net]), np.std(final_test_auc[this_net])))

    # save condensed dataset
    syn_data_save_path = os.path.join(cf.save_dir, "syn_data.pt")
    data_save=[]
    data_save.append([copy.deepcopy(data_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
    torch.save(
        {'syn_dataset': data_save,
         'final_test_auc': final_test_auc,
         "all_test_auc":all_test_auc},
        syn_data_save_path)
    logging.info("Condensed dataset saved to : {}".format(syn_data_save_path))

if __name__ == '__main__':
    main()

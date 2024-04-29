import torch
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import math


def train_one_epoch(net, train_loader, optimizer, criterion, device="cuda:0"):

    losses = []
    net.train()

    for batch_idx, (X, y) in enumerate(train_loader):

        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        score = net(X).squeeze()
        loss = criterion(score, y.float())

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def evaluate(net, data_loader, criterion=None, device="cuda:0"):
    # evaluate net with a data loader
    net.eval()
    eval_preds, eval_losses, eval_true_y = [], [], []

    with torch.no_grad():
        net.eval()
        for batch_idx, (X, y) in enumerate(data_loader):

            X, y = X.to(device), y.to(device)
            score = net(X).squeeze()

            if criterion is not None:
                this_loss = criterion(score, y.float())
                eval_losses.append(this_loss.item())

            eval_preds.extend(score.cpu().detach().tolist())
            eval_true_y.extend(y.cpu().detach().tolist())

    loss = np.mean(eval_losses) if eval_losses else -1
    auc = roc_auc_score(y_true=eval_true_y, y_score=eval_preds)

    return loss, auc


def get_net(net_type, **net_args):

    if net_type.startswith("tcn"):
        ### TCN network
        from model.tcn import TCN, MultiscaleMultibranchTCN
        if len(net_args["tcn_options"]["kernel_size"])==1:
            net_fn = TCN
        else:
            net_fn = MultiscaleMultibranchTCN
    elif net_type.startswith("lstm"):
        ### LSTM network
        from model.lstm import LSTMClassifier
        net_fn = LSTMClassifier
    elif net_type.startswith("trsf") or  net_type.startswith("vit"):
        ### transformer or vit networks
        from model.vit import TransformerEncoder
        net_fn = TransformerEncoder
    elif net_type.startswith("rnn"):
        ## RNN network
        from model.lstm import RNNClassifier
        net_fn = RNNClassifier
    else:
        raise NotImplementedError("Network: {} not implemented".format(net_type))
    net = net_fn(**net_args)
    return net


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(
            self,
            patience=7,
            delta=0,
            trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7, if None, will not do early stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        # self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func


    def __call__(self, val_score, mode, ):

        if self.patience is not None:  # only do early stopping when patience is not None

            assert mode in ("min", "max",)
            if mode == "min":
                score = -val_score   # for validation loss, aim to find minimum
            else:
                score = val_score   # for validation accuracy metric like auc, find maximum

            if self.best_score is None:
                self.best_score = score
                # self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                # self.save_checkpoint(val_loss, model)
                self.counter = 0

class StepCtr:
    def __init__(self):
        self.ctr = 0

    def advance_ctr(self):
        self.ctr+=1

    def reset_ctr(self):
        self.ctr = 0

    def get_ctr(self):
        return self.ctr


def update_tracker(tracker, step, key, val, mode="train"):
    if tracker is not None:
        if tracker.writer is not None:
            tracker.writer.set_step(step, mode=mode)
        if type(key)==str:
            tracker.update(key, val)
        else:
            for k,v in zip(key, val):
                tracker.update(k, v)

def eval_net(
        net, train_loader, val_loader, test_loader,
        lr, epochs, save_dir,
        save_name="ck_best.pt", val_metric="auc",
        device="cuda:0", verbose=False,
        weight_decay=0.0,
        tr_tracker=None, eval_tracker=None,
        early_stop=None, early_stop_metric="loss",
):

    verboseprint = print if verbose else lambda *a, **k: None

    net.to(device)

    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCELoss()

    early_stopping = EarlyStopping(patience=early_stop, trace_func=verboseprint)

    best_val_loss, best_val_auc, best_val_ep = math.inf, -1.0, 0
    _, test_auc = evaluate(net, test_loader, device=device, criterion=None)

    update_tracker(eval_tracker, -1, "test_auc", test_auc, mode="test")

    ck_save_path = os.path.join(save_dir, save_name)

    # tr_step_ctr = 0
    for ep in range(epochs):

        tr_loss = train_one_epoch(net, train_loader, optimizer, criterion, device=device)
        update_tracker(tr_tracker, ep, "tr_loss", tr_loss)

        val_loss, val_auc = evaluate(net, val_loader, device=device, criterion=criterion)
        update_tracker(eval_tracker, ep, ("val_loss", "val_auc",), (val_loss, val_auc,), mode="validate")

        verboseprint("Epoch {:02d}/{:02d}, loss: {:.4f}, val loss: {:.4f}, val AUC: {:.4f}".format(
            ep + 1, epochs, tr_loss, val_loss, val_auc))

        if val_metric=="loss":
            if val_loss < best_val_loss:
                best_val_loss, best_val_ep = val_loss, ep
                _, test_auc = evaluate(net, test_loader, device=device, criterion=None)
                update_tracker(eval_tracker, ep, "test_auc", test_auc, mode="test")
                torch.save(net.state_dict(), ck_save_path)

                verboseprint("Best val loss: {:.4f}, epoch: {}, saving checkpoint ...".format(best_val_loss, best_val_ep))
                verboseprint("Test AUC: {:.4f}".format(test_auc))
        else:
            if val_auc>best_val_auc:
                best_val_auc, best_val_ep = val_auc, ep
                _, test_auc = evaluate(net, test_loader, device=device, criterion=None)
                update_tracker(eval_tracker, ep, "test_auc", test_auc, mode="test")
                torch.save(net.state_dict(), ck_save_path)
                verboseprint("Best val AUC: {:.4f}, epoch: {}, saving checkpoint ...".format(best_val_auc, best_val_ep))
                verboseprint("Test AUC: {:.4f}".format(test_auc))

        if early_stop_metric=="loss":
            val_score, mode = val_loss, "min"
        else:
            val_score, mode = val_auc, "max"

        early_stopping(val_score, mode)
        if early_stopping.early_stop:
            verboseprint("Early stopping at epoch: {}".format(ep))
            break

    verboseprint("Final test AUC: {:.4f}".format(test_auc))

    return net, test_auc
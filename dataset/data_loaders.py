import torch
from torch.utils.data import TensorDataset, DataLoader
import _pickle as cPickle
import numpy as np
from sklearn.model_selection import train_test_split
import os

def get_loaders_mimic3(
        path="../datasets/mimic3", train_batch=64, val_batch=64, test_batch=64, sampler=True,
        combine_val=False, pre_process="minmax",
        ds_half=0,):

    from dataset.meta import mimic3_fea_dim_no_dup   # the feature dimensions without duplications for mimic3

    with open(os.path.join(path,'train_raw.p'), 'rb') as f:
        tr_x = cPickle.load(f)

    tr_data, tr_lb = tr_x[0], tr_x[1]
    tr_data = tr_data[..., mimic3_fea_dim_no_dup]   # remove duplicated feature dimensions

    ### half train set exp
    assert ds_half in (0,1,2,)
    if ds_half:
        mid=len(tr_data)//2
        if ds_half==1:
            # use first half as the training data
            print("Using first half as the training set")
            tr_data, tr_lb = tr_data[0:mid], tr_lb[0:mid]
        else:
            print("Using second half as the training set")
            tr_data, tr_lb = tr_data[mid:], tr_lb[mid:]

    prpr = DataNormalization(tr_data, pre_process=pre_process)
    tr_data = prpr(tr_data)
    
    train_dataset= TensorDataset(torch.FloatTensor(tr_data),torch.LongTensor(tr_lb))
    
    if sampler:
        class_sample_count = np.array([len(np.where(tr_lb == t)[0]) for t in np.unique(tr_lb)])
        weight = 1. / class_sample_count
        # weight[1]=weight[1]*2
        samples_weight = np.array([weight[t] for t in tr_lb])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight)) 
        train_loader = DataLoader(train_dataset, batch_size=train_batch, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
       
    with open(os.path.join(path,'val_raw.p'), 'rb') as f:
        x= cPickle.load(f)

    val_data, val_lb = x[0], x[1]
    val_data = val_data[..., mimic3_fea_dim_no_dup]  # remove duplicated feature dimensions
    val_data = prpr(val_data)

    val_dataset= TensorDataset(torch.FloatTensor(val_data),torch.LongTensor(val_lb))
    val_loader = DataLoader(val_dataset, batch_size=val_batch)

    with open(os.path.join(path,'test_raw.p'), 'rb') as f:
        x= cPickle.load(f)

    test = x["data"][0]
    test = test[..., mimic3_fea_dim_no_dup]  # remove duplicated feature dimensions

    test = prpr(test)
    test_labels = np.array(x["data"][1])

    test_dataset= TensorDataset(torch.FloatTensor(test),torch.LongTensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=test_batch)

    if combine_val:
        # combine validation with train set
        res_data=np.concatenate([tr_data, val_data], axis=0)
        res_lb=np.concatenate([tr_lb, val_lb], axis=0)
        return train_loader, val_loader, test_loader, res_data, res_lb, prpr
    else:
        return train_loader, val_loader, test_loader, tr_data, tr_lb, prpr


def create_data_loader(T, L, batch_size=64, sampler=False):

    if not torch.is_tensor(T):
        T = torch.FloatTensor(T)

    if not torch.is_tensor(L):
        L = torch.LongTensor(L)

    train_dataset = TensorDataset(T, L)

    if sampler:
        L_np = L.cpu().detach().clone().numpy()
        class_sample_count = np.array([len(np.where(L_np == t)[0]) for t in np.unique(L_np)])
        weight = 1. / class_sample_count
        # weight[1]=weight[1]*2
        samples_weight = np.array([weight[t] for t in L_np])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def collate_batch(batch):
  
    label_list, data_list, = [], []
  
    for (_data,_label) in batch:
        label_list.append(_label)
        data = torch.tensor(_data, dtype=torch.float32)
        data_list.append(data)
  
    label_list = torch.tensor(label_list, dtype=torch.float32)
  
    data_list = torch.nn.utils.rnn.pad_sequence(data_list, batch_first=True, padding_value=0)
  
    return data_list,label_list


class DataNormalization(object):
    # do normalization along each feature dimension
    def __init__(self, X_train, pre_process="std", eps=1e-9, prpr_ax=(0,1,)):
        # X_train is a N*T*F ndarray or a list of len N where X_train[i] has a shape T_i*F
        # pre_process: "minmax": min-max normalization, "std": z-normalization, "none": do nothing

        self.eps=eps # be added to denominator to avoid divided by zero

        assert pre_process in ("none", "std", "minmax"), \
            "Invalid pre_process: {}, needs to be one of: none/std/minmax".format(pre_process)

        self.pre_process = pre_process  # pre-processing method

        if self.pre_process!="none":

            # only compute statistical information when necessary
            if type(X_train) is list and len(X_train[0].shape)==2:
                # X_train is a list with len N, X_train[i] has a shape T_i*F
                X_source = np.concatenate(X_train)  # concatenate list into T_{ALL}*F
                ax = (0,)  # the axis to compute statistical information
            elif type(X_train) is np.ndarray and len(X_train.shape)==3:
                # X_train is a N*T*F matrix
                X_source = X_train
                ax = prpr_ax   # the axis to compute statistical information
            else:
                raise NotImplementedError("Normalization method not implemented!")

            self.mean = np.mean(X_source, axis=ax, keepdims=True)
            self.std = np.std(X_source, axis=ax, keepdims=True)+self.eps
            self.max = np.amax(X_source, axis=ax, keepdims=True)
            self.min = np.amin(X_source, axis=ax, keepdims=True)
            self.dif = (self.max-self.min)+self.eps

    def norm_std(self, data):
        if type(data) is list:
            norm_data = [(item - self.mean) / self.std for item in data]
        else:
            norm_data = (data - self.mean) / self.std
        return norm_data

    def norm_minmax(self, data):
        if type(data) is list:
            norm_data = [(item-self.min)/self.dif for item in data]
        else:
            norm_data = (data-self.min)/self.dif
        return norm_data

    def recover_minmax(self, data):
        if type(data) is list:
            rec_data = [(item*self.dif)+self.min for item in data]
        else:
            rec_data = (data * self.dif) + self.min
        return rec_data

    def recover_std(self, data):
        if type(data) is list:
            rec_data = [(item*self.std)+self.mean for item in data]
        else:
            rec_data = (data*self.std)+self.mean
        return rec_data

    def __call__(self, data):
        # data is a N*T*F ndarray or a list of len N where X_train[i] has a shape T_i*F
        if self.pre_process == "none":
            return data
        else:
            if self.pre_process == "minmax":
                # min-max normalization
                norm_data = self.norm_minmax(data)
            elif self.pre_process == "std":
                # standard normalization
                norm_data = self.norm_std(data)
            else:
                raise NotImplementedError("Pre-processing method: {} not implemented".format(self.pre_process))
            return norm_data

    def recover(self, data):
        # recover the data before normalization
        if self.pre_process == "none":
            return data
        else:
            if self.pre_process == "minmax":
                rec_data = self.recover_minmax(data)
            elif self.pre_process == "std":
                rec_data = self.recover_std(data)
            else:
                raise NotImplementedError("Pre-processing method: {} not implemented".format(self.pre_process))
            return rec_data


def get_loader_physio(path="../datasets/PhysioNet",train_batch=64, val_batch=64, test_batch=64,
                      sampler=True, random_seed=42,
                      combine_val=False, pre_process="std"):

    T=np.load(os.path.join(path, 'Data.npy'))
    L=np.load(os.path.join(path, 'Labels.npy'))

    X_train_val, X_test, y_train_val, y_test = train_test_split(T, L, test_size=0.2, random_state=random_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=random_seed)

    prpr = DataNormalization(X_train, pre_process=pre_process)
    X_train, X_val, X_test, X_train_val = \
        prpr(X_train), prpr(X_val), prpr(X_test), prpr(X_train_val)

    train_dataset= TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)

    val_dataset= TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=val_batch,shuffle=False)

    test_dataset= TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=test_batch,shuffle=False)

    if combine_val:
        return train_loader, val_loader, test_loader, X_train_val, y_train_val, prpr
    else:
        return train_loader, val_loader, test_loader, X_train, y_train, prpr


class GetData(object):
    # load random n samples from class c

    def __init__(self, tr_data, indices_class, device="cuda:0"):
        self.tr_data = tr_data
        self.indices_class = indices_class
        self.device = device

    def __call__(self, c, n):
        idx_shuffle = np.random.permutation(self.indices_class[c])[:n]
        return self.tr_data[idx_shuffle].to(self.device)


# build a real train data getter to load random n samples from class c
def build_data_getter(ds_name, tr_data, tr_lb, device):

    if ds_name=="mimic3" or ds_name=="physio" or ds_name=="covid_b" or ds_name=="covid_c":
        num_classes = 2
        indices_class = [[] for _ in range(num_classes)]  # class indices

        for i, lab in enumerate(tr_lb):
            indices_class[int(lab)].append(i)

        tr_data = torch.FloatTensor(tr_data).to("cpu")
        data_getter = GetData(tr_data, indices_class, device=device)
        return data_getter
    else:
        raise NotImplementedError("Not implemented data getter for dataset: {}")


def get_loaders_covid(
        path="../datasets/covid/breath",
        train_batch=64, val_batch=64, test_batch=64,
        pre_process="none", combine_val=False,
):
    X_train = np.squeeze(np.load(os.path.join(path,'X_train.npy')))
    X_val = np.squeeze(np.load(os.path.join(path, 'X_val.npy')))
    X_test = np.squeeze(np.load(os.path.join(path, 'X_test.npy')))
    y_train = np.load(os.path.join(path, 'y_train.npy'))
    y_val = np.load(os.path.join(path,'y_val.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy'))

    prpr = DataNormalization(X_train, pre_process=pre_process,)

    X_train = prpr(X_train)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)

    X_val = prpr(X_val)
    val_dataset= TensorDataset(torch.FloatTensor(X_val),torch.LongTensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=val_batch)

    X_test = prpr(X_test)
    test_dataset= TensorDataset(torch.FloatTensor(X_test),torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=test_batch)

    return train_loader, val_loader, test_loader, X_train, y_train, prpr
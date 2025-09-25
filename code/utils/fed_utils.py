from collections import defaultdict
from torch.optim import Optimizer
import torch
import copy
from prettytable import PrettyTable
from torch import nn
import math
from utils.finch import FINCH
import numpy as np

class AdamSVD(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, svd=True, thres=600.,
                 weight_decay=0, amsgrad=False, ratio=0.8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, svd=svd, thres=thres)
        super(AdamSVD, self).__init__(params, defaults)

        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        self.ratio = ratio

    def __setstate__(self, state):
        super(AdamSVD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('svd', False)

    def step(self, closure=None):
       
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamSVD does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)
                if svd and len(self.transforms) > 0:
                  
                    if len(update.shape) == 3:
                      
                        update_ = torch.bmm(update, self.transforms[p]).view_as(update)

                    else:
                        update_ = torch.mm(update, self.transforms[p][0])

                else:
                    update_ = update
                p.data.add_(update_)
        return loss

    def get_transforms(self):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                thres = group['thres']
                temp = []
                for s in range(self.eigens[p]['eigen_value'].shape[0]):
                    ind = self.eigens[p]['eigen_value'][s] <= self.eigens[p]['eigen_value'][s][-1] * thres
                    ind = torch.ones_like(ind)
                    ind[: int(ind.shape[0] * (1.0 - self.ratio))] = False
                 
                    basis = self.eigens[p]['eigen_vector'][s][:, ind]
                    transform = torch.mm(basis, basis.transpose(1, 0))
                    temp.append(transform / torch.norm(transform))
                self.transforms[p] = torch.stack(temp, dim=0)
                self.transforms[p].detach_()

    def get_eigens(self, fea_in):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                eigen = self.eigens[p]
                eigen_values, eigen_vectors = [], []

               
                _, eigen_value, eigen_vector = torch.svd(fea_in[idx], some=False)
                eigen_values.append(eigen_value)
                eigen_vectors.append(eigen_vector)
                eigen['eigen_value'] = torch.stack(eigen_values, dim=0)
                eigen['eigen_vector'] = torch.stack(eigen_vectors, dim=0)

    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]

       
        if len(state) == 0:
            state['step'] = 0
           
            state['exp_avg'] = torch.zeros_like(p.data)

            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
             
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

       
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
       
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
          
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * \
                    math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update


def show_results(cfg, results, epoch,global_test_acc_dict):

    global_test_acc = []
    global_test_error = []
    global_test_f1 = []
    for k in results.keys():
        global_test_acc.append(results[k]['accuracy'])
        global_test_error.append(results[k]['error_rate'])
        global_test_f1.append(results[k]['macro_f1'])

        if k in global_test_acc_dict:
            global_test_acc_dict[k].append(results[k]['accuracy'])
        else:
            global_test_acc_dict[k] = [results[k]['accuracy']]

    print("--Global test acc:", sum(global_test_acc) / len(global_test_acc))
    print(f"Epoch:{epoch}")
    return global_test_acc,global_test_acc_dict

def fin_proto_agg(idxs_users, local_prototype):
    global_cluster_prototypes = dict()
    for idx in idxs_users:
        local_protos = local_prototype[idx]
        for label in local_protos.keys():
            if label in global_cluster_prototypes:
                global_cluster_prototypes[label].append(local_protos[label])
            else:
                global_cluster_prototypes[label] = [local_protos[label]]
    agg_proto = {}
    for label in global_cluster_prototypes.keys():
        proto_list = global_cluster_prototypes[label]
        if len(proto_list) > 1:
            proto_list = [item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list]
            proto_list = np.array(proto_list)
            c, num_clust, req_c = FINCH(proto_list, initial_rank=None, req_clust=None, distance='cosine',
                                        ensure_early_exit=False, verbose=True)
            m, n = c.shape
            class_cluster_list = []
            for index in range(m):
                class_cluster_list.append(c[index, -1])
            class_cluster_array = np.array(class_cluster_list)
            uniqure_cluster = np.unique(class_cluster_array).tolist()
            agg_selected_proto = []
            for _, cluster_index in enumerate(uniqure_cluster):
                selected_array = np.where(class_cluster_array == cluster_index)
                selected_proto_list = proto_list[selected_array]
                proto = np.mean(selected_proto_list, axis=0, keepdims=True)
                agg_selected_proto.append(torch.tensor(proto))
            agg_proto[label] = torch.cat(agg_selected_proto)
        else:
            agg_proto[label] = torch.cat([proto_list[0].data]).view(1, -1)
    return agg_proto

def mean_proto_agg(idxs_users, local_prototype):
    global_prototypes = dict()
    for idx in idxs_users:
        local_protos = local_prototype[idx]
        for label in local_protos.keys():
            if label in global_prototypes:
                global_prototypes[label].append(local_protos[label].view(1, -1))
            else:
                global_prototypes[label] = [local_protos[label].view(1, -1)]
    agg_proto = {}
    for label in global_prototypes.keys():
        proto_list = global_prototypes[label]
        proto_list = torch.cat(proto_list, dim=0)
        proto_list = torch.mean(proto_list, dim=0).view(1, -1)
        agg_proto[label] = proto_list


    sorted_keys = sorted(agg_proto.keys())

    sorted_tensors = [agg_proto[key] for key in sorted_keys]
    concatenated_tensor = torch.cat(sorted_tensors, dim=0)

    norms = concatenated_tensor.norm(p=2, dim=1, keepdim=True)

    norm_global_prototypes = concatenated_tensor / norms

    return norm_global_prototypes


def calculate_global_class_gaussian(idxs_users, local_prototype):
    global_prototypes = dict()
    for idx in idxs_users:
        local_protos = local_prototype[idx]
        for label in local_protos.keys():
            if label in global_prototypes:
                global_prototypes[label].append(local_protos[label].view(1, -1))
            else:
                global_prototypes[label] = [local_protos[label].view(1, -1)]
    sorted_keys = sorted(global_prototypes.keys())

    global_class_gaussian = dict()
    global_class_gaussian['mean'] = dict()
    global_class_gaussian['var'] = dict()
    global_class_gaussian['cov_matrix'] = dict()

    for label in sorted_keys:
        proto_list = global_prototypes[label]
        proto_list = torch.cat(proto_list, dim=0)

        means = proto_list.mean(dim=0)
        global_class_gaussian['mean'][label] = means

        variances = proto_list.var(dim=0, unbiased=True)  
        global_class_gaussian['var'][label] = variances

        cov_matrix = np.corrcoef(proto_list.T.cpu().numpy())
        global_class_gaussian['cov_matrix'][label] = cov_matrix

    return global_class_gaussian


class FiLM(nn.Module):
    def __init__(self,
                 dim,
                 bias=True,
                 use_sigmoid=False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.has_bias = bias
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        scale = self.scale.unsqueeze(0).type(x.dtype)
        bias = self.bias.unsqueeze(0).type(x.dtype) if self.has_bias else None

        x = scale * x
        if bias is not None:
            x = x + bias

        if self.use_sigmoid:
            return x.sigmoid()

        return x

def average_weights(w, idxs_users, datanumber_client, islist=False):

    total_data_points = sum([datanumber_client[r] for r in idxs_users])

    w_avg = copy.deepcopy(w[idxs_users[0]])
    for idx in range(len(idxs_users)):
        fed_avg_freqs = datanumber_client[idxs_users[idx]] / total_data_points

        if islist:
            if idx == 0:
                w_avg = w_avg * fed_avg_freqs
            else:
                w_avg += w[idxs_users[idx]] * fed_avg_freqs
        else:
            if idx == 0:
                for key in w_avg:
                    w_avg[key] = w_avg[key] * fed_avg_freqs
            else:
                for key in w_avg:
                    w_avg[key] += w[idxs_users[idx]][key] * fed_avg_freqs

    return w_avg


def count_parameters(model, model_name):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if model_name in name:

            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params



import os

def save_acc_csv(para_dir,global_test_acc_dict,cfg):
    acc_path = os.path.join(para_dir, 'acc.csv')
    if os.path.exists(acc_path):
        with open(acc_path, 'a') as result_file:
            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(key + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
    else:
        with open(acc_path, 'w') as result_file:
            result_file.write('domain,')
            for epoch in range(cfg.OPTIM.ROUND):
                result_file.write('epoch_' + str(epoch))
                if epoch != cfg.OPTIM.ROUND - 1:
                    result_file.write(',')
                else:
                    result_file.write('\n')

            for key in global_test_acc_dict:
                method_result = global_test_acc_dict[key]
                result_file.write(key + ',')
                for epoch in range(len(method_result)):
                    result_file.write(str(method_result[epoch]))
                    if epoch != len(method_result) - 1:
                        result_file.write(',')
                    else:
                        result_file.write('\n')
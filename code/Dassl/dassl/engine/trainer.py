import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
from utils.fed_utils import AdamSVD
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from Dassl.dassl.data import DataManager
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from Dassl.dassl.modeling import build_head, build_backbone
from Dassl.dassl.evaluation import build_evaluator
from torch.nn import functional as F

def WireframeSphere(centre=[0.,0.,0.], radius=1.,
                    n_meridians=30, n_circles_latitude=None):
    if n_circles_latitude is None:
        n_circles_latitude = max(n_meridians/2, 4)
    u, v = np.mgrid[0:2*np.pi:n_meridians*1j, 0:np.pi:n_circles_latitude*1j]
    sphere_x = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere_y = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere_z = centre[2] + radius * np.cos(v)
    return sphere_x, sphere_y, sphere_z

class SimpleNet(nn.Module):


    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features
    
        self.head = None
       
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            print("num_classes", num_classes)
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:


    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            print("save model name", name)
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print(f"Load {model_path} to {name} (epoch={epoch})")
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
          
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch, idx=-1, global_epoch=-1, is_fed=False, **kwargs):

        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        self.before_train(is_fed)
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch(idx, global_epoch, **kwargs)
            self.after_epoch()
        self.after_train(idx, global_epoch, is_fed)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self,idx, batch_idx, batch, **kwargs):
        raise NotImplementedError


    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)

    def vis_prograd_backward_and_update(self, loss_ce, loss_kl, loss_extra=None, gama=1, names=None):
        self.model_zero_grad(names)

        names = self.get_model_names(names)
       
        self.detect_anomaly(loss_kl)
        loss_kl.backward(retain_graph=True, create_graph=True)

        kl_grads = []
        for name in names:
            for query_name, query_param in self._models[name].named_parameters():
                if query_param.requires_grad and 'VPT' in query_name:
                    kl_grads = (query_param.grad.clone().detach())
        self.model_zero_grad(names)
        self.detect_anomaly(loss_ce)
        loss_ce.backward(retain_graph=True, create_graph=True)
        for name in names:
            for query_name, query_param in self._models[name].named_parameters():
                if query_param.requires_grad and 'VPT' in query_name:
                    ce_grads = (query_param.grad.clone().requires_grad_(True))
        self.model_zero_grad(names)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        gradient_divergence = 0
        cos_para = - cos(kl_grads.view(1, -1), ce_grads.view(1, -1))
        if cos_para < 0:  
            cos_para = 0
      
        gradient_divergence += cos_para
        loss = loss_ce + gama * gradient_divergence

        if loss_extra is not None:
            loss = loss + loss_extra
        self.model_backward(loss)
        self.model_update(names)

    def prograd_backward_and_update(self, loss_ce, loss_kl, loss_extra=None, gama=1, names=None):
        self.model_zero_grad(names)
       
        names = self.get_model_names(names)

        self.detect_anomaly(loss_kl)
        loss_kl.backward(retain_graph=True, create_graph=True)
  
        kl_grads = []
        for name in names:
            for query_name, query_param in self._models[name].named_parameters():
                if query_param.requires_grad:
                    kl_grads.append(query_param.grad.clone().detach())
        self.model_zero_grad(names)
        self.detect_anomaly(loss_ce)
        loss_ce.backward(retain_graph=True, create_graph=True)
        ce_grads = []
        for name in names:
            for query_name, query_param in self._models[name].named_parameters():
                if query_param.requires_grad:
                    ce_grads.append(query_param.grad.clone().requires_grad_(True))
        self.model_zero_grad(names)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        gradient_divergence = 0
        num_pairs = len(kl_grads)
        for kl_grad, ce_grad in zip(kl_grads, ce_grads):
            cos_para = - cos(kl_grad.view(1, -1), ce_grad.view(1, -1))
            if cos_para < 0: 
                cos_para = 0
            else:
                print(cos_para)
            gradient_divergence += cos_para
        gradient_divergence = gradient_divergence / num_pairs

        loss = loss_ce + gama * gradient_divergence

        if loss_extra is not None:
            loss = loss + loss_extra
        self.model_backward(loss)
        self.model_update(names)

    def text_prograd_backward_and_update(self, loss_ce, loss_kl, loss_extra=None, gama=1, names=None):
        self.model_zero_grad(names)

        names = self.get_model_names(names)

        self.detect_anomaly(loss_kl)
        loss_kl.backward(retain_graph=True, create_graph=True)

        kl_grads = []
        for name in names:
            for query_name, query_param in self._models[name].named_parameters():
                if query_param.requires_grad and 'ctx' in query_name:
                    kl_grads = (query_param.grad.clone().detach())
        self.model_zero_grad(names)
        self.detect_anomaly(loss_ce)
        loss_ce.backward(retain_graph=True, create_graph=True)
        for name in names:
            for query_name, query_param in self._models[name].named_parameters():
                if query_param.requires_grad and 'ctx' in query_name:
                    ce_grads = (query_param.grad.clone().requires_grad_(True))
        self.model_zero_grad(names)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        gradient_divergence = 0
        cos_para = - cos(kl_grads.view(1, -1), ce_grads.view(1, -1))
        if cos_para < 0: 
            cos_para = 0
        else:
            print(cos_para)
        gradient_divergence += cos_para
        loss = loss_ce + gama * gradient_divergence

        if loss_extra is not None:
            loss = loss + loss_extra
        self.model_backward(loss)
        self.model_update(names)


def cal_confidence_diff(output, label, device):
    prob = torch.softmax(output, dim=-1)
    n_class = prob.shape[1]
    all_non_label_index = []

    for i in range(len(label)):
        this_label = label[i]
        non_label_index_list = []
        for j in range(n_class):
            if j != this_label:
                non_label_index_list.append(j)

        all_non_label_index.append(non_label_index_list)

    all_non_label_index = torch.tensor(all_non_label_index, device=device)

    not_label_prob = prob.gather(-1, all_non_label_index)
    not_label_max_prob, not_label_max_prob_label = not_label_prob.max(1)
    label_prob = prob.gather(-1, label.view(-1, 1)).view(-1)

    prob_diff = label_prob - not_label_max_prob

    right_pred_place = output.max(1)[1] == label

    right_prob_diff = prob_diff[right_pred_place]
    wrong_prob_diff = prob_diff[~right_pred_place]
    return prob_diff, right_prob_diff, wrong_prob_diff


class SimpleTrainer(TrainerBase):
  

    def __init__(self, args, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda:" + str(args.device_id))
        else:
            self.device = torch.device("cpu")

    
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.args = args
        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)  
        self.best_result = -np.inf

    def check_cfg(self, cfg):
      
        pass

    def build_data_loader(self):
      
        dm = DataManager(self.cfg)

      
        self.test_loader = dm.test_loader
        self.fed_train_loader_x_dict = dm.fed_train_loader_x_dict  
        self.fed_test_loader_x_dict = dm.fed_test_loader_x_dict 
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  
        self.classnames = dm.classnames

        self.dm = dm

    def build_model(self):
      
        cfg = self.cfg

        print("Building model")
        print("self.num_classes", self.num_classes)
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
           
    def train(self, idx=-1, global_epoch=0, is_fed=False, **kwargs):
        super().train(self.start_epoch, self.max_epoch, idx, global_epoch, is_fed, **kwargs)

    def fed_before_train(self, is_global=False):
        self.directory_path = self.cfg.OUTPUT_DIR
       
        self.start_epoch = 0

      

        self.total_time_start = time.time()

    def before_train(self, is_fed=False):
        if not is_fed:
            directory = self.cfg.OUTPUT_DIR
            if self.cfg.RESUME:
                directory = self.cfg.RESUME
        
        self.start_epoch = 0

        # Initialize summary writer
        if not is_fed:
            writer_dir = osp.join(self.output_dir, "tensorboard")
            mkdir_if_missing(writer_dir)
            self.init_writer(writer_dir)

        if self.args.trainer =='VPTPR':
            self.optim = AdamSVD([self.model.image_encoder.VPT], lr=self.cfg.OPTIM.LR,
                                 weight_decay=self.cfg.OPTIM.WEIGHT_DECAY, ratio=self.cfg.TRAINER.VPTPR.RATIO)
            if self.fea_in != None:
                self.optim.get_eigens(fea_in=self.fea_in)
                self.optim.get_transforms()
        else:
          
            param_groups = self.optim.param_groups
            self.optim = optim.SGD(param_groups)

        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
     
        self.time_start = time.time()

    def after_train(self, idx=-1, epoch=0, is_fed=False):
        print("Finish training:", idx, "user")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")

                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
      

        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        if not is_fed:
            print(f"Total time Elapsed: {elapsed}")
        else:
            print(f"{idx} User, Elapsed: {elapsed}")


        if not is_fed:
            self.close_writer()

    def fed_after_train(self):
        elapsed = round(time.time() - self.total_time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Total time Elapsed: {elapsed}")

        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result

    @torch.no_grad()
    def test(self, split=None, is_global=False, current_epoch=0, idx=-1, global_test=True):
     
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  
            if global_test == True:
                data_loader = self.test_loader
                print(f"Evaluate on the Global_{split} set")
            elif global_test == False:
                data_loader = self.fed_test_loader_x_dict[idx]
                print(f"Evaluate on the client{idx}_{split} set")

        if isinstance(data_loader, dict):  
            results = dict()
            domain_visual_feature_list_dict  = {}
            domain_domain_label_list_dict = {}
            for query_domain, query_data_loader in data_loader.items():
                query_domain_visual_feature_list  = []
                query_domain_textual_features_list = []
                query_domain_label_list = []
                for batch_idx, batch in enumerate(tqdm(query_data_loader)):
                    input, label = self.parse_batch_test(batch)
                    output,image_features, text_features = self.model_inference(input)

                    self.evaluator.process(output, label)
                    if image_features is not None:
                        query_domain_visual_feature_list.append(image_features.detach())
                        query_domain_label_list.append(label.detach())
                        if batch_idx ==0:
                            query_domain_textual_features_list.append(text_features.detach())
                domain_visual_feature_list_dict[query_domain] = query_domain_visual_feature_list
                domain_domain_label_list_dict[query_domain] = query_domain_label_list
                print(query_domain)
                results[query_domain] = self.evaluator.evaluate()

                self.evaluator.reset()

             

        else:
            results = dict()
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                input, label = self.parse_batch_test(batch)
                output,_,_ = self.model_inference(input)
                self.evaluator.process(output, label)
            results['global'] = self.evaluator.evaluate()

      
        return results

    def model_inference(self, input):
        output = self.model(input)
        if isinstance(output, tuple):
            image_features = output[1]
            text_features = output[2]
            output = output[0]
            return output,image_features,text_features
        else:
            return output,None,None

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):


    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

      
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):


    def run_epoch(self, idx=-1, global_epoch=-1, **kwargs):

        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        if idx >= 0:
            loader = self.fed_train_loader_x_dict[idx]
        else:
            loader = self.train_loader_x
        self.num_batches = len(loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(idx ,self.batch_idx, batch, **kwargs)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                                     self.max_epoch - self.epoch - 1
                             ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                info += [f"user {idx}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            if global_epoch >= 0:
                max_per_epoch = self.max_epoch * self.num_batches
             
                n_iter = global_epoch * max_per_epoch + n_iter
            
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name + "/" + str(idx), meter.avg, n_iter)
            
            self.write_scalar("train/lr/" + str(idx), self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

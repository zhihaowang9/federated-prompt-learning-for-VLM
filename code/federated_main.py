from collections import defaultdict
from utils.fed_utils import (
    average_weights,
    count_parameters,
    show_results,
    save_acc_csv,
)
from Dassl.dassl.utils import setup_logger, set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import setproctitle
import numpy as np
import argparse
import torch
import time
import copy
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def calculate_similarity(vec1, vec2):
    similarity = 0
    for i in range(vec1.shape[1]):
        similarity += torch.abs(torch.dot(vec1[:, i].t(), vec2[:, i]))
    return similarity


def cal_importance_per_channel(feats, channel_inds):
    labels = (feats["label"]).long()
    img_feats = (feats["img"]).float()
    text_feats = (feats["text"]).float()

    img_feats = img_feats[:, channel_inds].view(-1, 1)
    text_feats = text_feats[:, channel_inds].view(-1, 1)
 
    similarities = img_feats @ text_feats.t()
    similarities = similarities.clamp_min_(0.0)
   
    similarities_gt = torch.gather(similarities, 1, labels.unsqueeze(-1)).squeeze(-1)
   
    importance = similarities_gt / (similarities.mean(dim=1) + 1e-12)
    return importance.mean().item()


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    from yacs.config import CfgNode as CN

    cfg.TRAINER.CLIP = CN()
    cfg.TRAINER.CLIP.PREC = "fp16" 
    cfg.TRAINER.CLIP.CLASS_TOKEN_POSITION = "end" 

    cfg.TRAINER.FedCLIP = CN()
    cfg.TRAINER.FedCLIP.PREC = "fp16"  
    cfg.TRAINER.FedCLIP.CLASS_TOKEN_POSITION = "end"  

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = 16  
    cfg.TRAINER.PROMPTFL.CSC = False  
    cfg.TRAINER.PROMPTFL.CTX_INIT = False  
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  

    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 16 
    cfg.TRAINER.IVLP.N_CTX_TEXT = 16  
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  
    cfg.TRAINER.IVLP.CSC = False  
    cfg.TRAINER.IVLP.PREC = "fp16"  
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = (
        1  
    )
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = (
        1  
    )

    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 16 
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  
    cfg.TRAINER.VPT.PREC = "fp16"  
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1 
    
    cfg.TRAINER.VPTPR = CN()
    cfg.TRAINER.VPTPR.PREC = "fp16" 
    cfg.TRAINER.VPTPR.N_CTX_VISION = 20
    cfg.TRAINER.VPTPR.CTX_INIT = "a photo of a"  
    cfg.TRAINER.VPTPR.PROMPT_DEPTH_VISION = 1
    cfg.TRAINER.VPTPR.RATIO = 0.8

    cfg.TRAINER.LPT = CN()
    cfg.TRAINER.LPT.N_CTX_TEXT = 16  
    cfg.TRAINER.LPT.CTX_INIT = "a photo of a"  
    cfg.TRAINER.LPT.CSC = False  
    cfg.TRAINER.LPT.PREC = "fp16"  
    cfg.TRAINER.LPT.PROMPT_DEPTH_TEXT = 1

    cfg.TRAINER.TPG = CN()
    cfg.TRAINER.TPG.N_CTX_TEXT = 4  
    cfg.TRAINER.TPG.CTX_INIT = "a photo of a"  
    cfg.TRAINER.TPG.PREC = "fp16"  
    cfg.TRAINER.TPG.PROMPT_DEPTH_TEXT = 1
    cfg.TRAINER.TPG.D_CTX = 1

    current_trainer_cfg = cfg["TRAINER"][args.trainer]
    cfg["TRAINER"] = CN()
    cfg["TRAINER"][args.trainer] = current_trainer_cfg
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  
    cfg.DATASET.Max_Class = 0  
    cfg.DATASET.USERS = args.num_users  
    cfg.DATASET.PARTITION = args.partition
    cfg.DATASET.BETA = args.beta
    cfg.DATASET.AGGREGATION = args.aggregation
    cfg.DATASET.REPEATRATE = 0.0  
    cfg.OPTIM.ROUND = 1 
    cfg.OPTIM.GAMMA = args.gamma  
    cfg.MODEL.BACKBONE.PRETRAINED = True


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)
    if args.dataset:
        cfg.merge_from_file(f"configs/datasets/{args.dataset}.yaml")
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    reset_cfg(cfg, args)
    cfg.OUTPUT_DIR = f"output/{args.dataset}/beta:{args.beta}/{args.trainer}"
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    args.para_dir = setup_logger(cfg)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    print_args(args, cfg)

    ckpt_path = args.para_dir + "/model.pth"
    local_weights = [[] for i in range(cfg.DATASET.USERS)]
    group_text_prompt_weight_list = None

    local_trainer = build_trainer(args, cfg)
    local_trainer.fed_before_train()
    count_parameters(local_trainer.model, "prompt_learner")
    count_parameters(local_trainer.model, "image_encoder")
    count_parameters(local_trainer.model, "text_encoder")
    datanumber_client = []
    if args.trainer == "CLIP":
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        if args.aggregation == "Weight":
            for net_i in range(cfg.DATASET.USERS):
                datanumber_client.append(
                    len(local_trainer.fed_train_loader_x_dict[net_i].dataset)
                )
        elif args.aggregation == "Equal":
            for net_i in range(cfg.DATASET.USERS):
                datanumber_client.append(1 / cfg.DATASET.USERS)
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    start_epoch = 0
    end_epoch = cfg.OPTIM.ROUND
    global_test_acc_dict = {}
    global_time_list = []
    global_similarity_list = []
    local_similarity_list = []
    start = time.time()
    if not args.eval_only:
        for epoch in range(start_epoch, end_epoch):
            if args.trainer == "CLIP":
                print("------------Global test start -------------")
                m = max(
                    int(args.frac * cfg.DATASET.USERS), 1
                )
                idxs_users = list(range(m))
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(
                        global_weights, strict=False
                    )
                results = local_trainer.test()
                global_test_acc, global_test_acc_dict = show_results(
                    cfg, results, epoch, global_test_acc_dict
                )
                global_time_list.append(time.time() - start)
                print("------------Global test finish-------------")
                break
            elif args.trainer == "VPTPR":
                m = max(int(args.frac * cfg.DATASET.USERS), 1)
                idxs_users = list(range(m))
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")

                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                print(f"------------{args.trainer}:Global test start-------------")
                all_users = list(range(0, cfg.DATASET.USERS))
                for idx in all_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)  
                results = local_trainer.test()
                global_test_acc, global_test_acc_dict = show_results(cfg, results, epoch, global_test_acc_dict)
                global_time_list.append(time.time() - start)
                print("------------Global test finish-------------")

                fea_in = defaultdict(dict)
                # 抓取聚合后的权重系数
                fea_in[0] = torch.mm(local_trainer.model.image_encoder.VPT.T, local_trainer.model.image_encoder.VPT)
                local_trainer.fea_in = fea_in
            elif args.trainer in ["VPT", "LPT", "IVLP", "PROMPTFL"]:
                m = max(
                    int(args.frac * cfg.DATASET.USERS), 1
                )
                idxs_users = list(range(m))
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(
                        idx=idx,
                        global_epoch=epoch,
                        is_fed=True,
                        global_weights=global_weights,
                    )
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")

                last_global_weights = copy.deepcopy(global_weights)
                global_weights = average_weights(
                    local_weights, idxs_users, datanumber_client
                )
                if args.trainer == "VPT":
                    global_direction=global_weights["image_encoder.VPT"]-last_global_weights["image_encoder.VPT"]
                    local_directions=[local_weights[idx]["image_encoder.VPT"]-last_global_weights["image_encoder.VPT"] for idx in idxs_users]
                    global_similarity=[]
                    local_similarity=[]
                    for idx in idxs_users:
                        sim=[cosine_similarity(global_direction.cpu()[i].reshape(1, -1), local_directions[idx].cpu()[i].reshape(1, -1))[0, 0] for i in range(16)]
                        mean_sim=np.mean(sim)
                        global_similarity.append(mean_sim)

                    similarity_matrix = np.zeros((len(local_directions), len(local_directions)))
                    local_directions= [local_directions[i].cpu() for i in range(len(local_directions))]
                    for i in range(len(local_directions)):
                        for j in range(len(local_directions)):
                            if i != j:
                                sim = [cosine_similarity(local_directions[i][k].reshape(1, -1), local_directions[j][k].reshape(1, -1))[0, 0] for k in range(16)]
                                mean_sim = np.mean(sim)
                                similarity_matrix[i, j] = mean_sim
                    local_similarity = similarity_matrix
                      
                    print("global_similarity",global_similarity)
                    print("local_similarity",local_similarity)
                    global_similarity_list.append(global_similarity)
                    local_similarity_list.append(local_similarity)
                if args.trainer == "LPT":
                    global_direction=global_weights["prompt_learner.ctx"]-last_global_weights["prompt_learner.ctx"]
                    local_directions=[local_weights[idx]["prompt_learner.ctx"]-last_global_weights["prompt_learner.ctx"] for idx in idxs_users]
                    global_similarity=[]
                    local_similarity=[]
                    for idx in idxs_users:
                        sim=[cosine_similarity(global_direction.cpu()[i].reshape(1, -1), local_directions[idx].cpu()[i].reshape(1, -1))[0, 0] for i in range(16)]
                        mean_sim=np.mean(sim)
                        global_similarity.append(mean_sim)
                    similarity_matrix = np.zeros((len(local_directions), len(local_directions)))
                    local_directions= [local_directions[i].cpu() for i in range(len(local_directions))]
                    for i in range(len(local_directions)):
                        for j in range(len(local_directions)):
                            if i != j:
                                sim = [cosine_similarity(local_directions[i][k].reshape(1, -1), local_directions[j][k].reshape(1, -1))[0, 0] for k in range(16)]
                                mean_sim = np.mean(sim)
                                similarity_matrix[i, j] = mean_sim
                    local_similarity = similarity_matrix
                    print("global_similarity",global_similarity)
                    print("local_similarity",local_similarity)
                    global_similarity_list.append(global_similarity)
                    local_similarity_list.append(local_similarity)
                    
                    

                print(f"------------{args.trainer}:Global test start-------------")
                local_trainer.model.load_state_dict(
                    global_weights, strict=False
                )
                results = local_trainer.test()

                global_test_acc, global_test_acc_dict = show_results(
                    cfg, results, epoch, global_test_acc_dict
                )
                global_time_list.append(time.time() - start)
                print("------------Global test finish-------------")
            if args.resume:
                torch.save(local_trainer.model.state_dict(), ckpt_path)
    else:
        global_weights = torch.load(ckpt_path)
        epoch = 0
        print("------------Global test start -------------")
        m = max(
            int(args.frac * cfg.DATASET.USERS), 1
        )
        idxs_users = list(range(m))
        for idx in idxs_users:
            local_trainer.model.load_state_dict(
                global_weights, strict=False
            )
        results = local_trainer.test()
        global_test_acc, global_test_acc_dict = show_results(
            cfg, results, epoch, global_test_acc_dict
        )
        global_time_list.append(time.time() - start)
        print("------------Global test finish-------------")
    for idx in idxs_users:
        local_trainer.fed_after_train()
    for key, global_test_acc_list in global_test_acc_dict.items():
        print(key, "global_test_acc_list:", global_test_acc_list)
        print(key, "maximum test acc:", max(global_test_acc_list))
        print(key, "mean of acc:", np.mean(global_test_acc_list[-5:]))
        print(key, "std of acc:", np.std(global_test_acc_list[-5:]))
    print('global_similarity_list',global_similarity_list)
    print('local_similarity_list',local_similarity_list)
    save_acc_csv(local_trainer.args.para_dir, global_test_acc_dict, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer",
        type=str,
        default="VPT",
        help="name of trainer, choose from: "
        "Baseline, CLIP, IVLP,  VPT, LPT,  ",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="name of dataset, choose from: tyimagenet"
        " cifar100 domainnet  OfficeHome  Office31 ",
    )
    parser.add_argument(
        "--backbone", type=str, default="ViT-B/16", help="name of CNN backbone"
    )
    parser.add_argument(
        "--aggregation", type=str, default="Weight", help="name of Aggregation strategy"
    )
    parser.add_argument(
        "--device_id", type=int, default=0, help="The Device Id for Experiment"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="The parameter for the dirichlet distribution",
    )
    parser.add_argument("--num_users", type=int, default=4, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=1, help="the fraction of clients: C"
    )
    parser.add_argument("--gamma", type=float, default=1, help="gamma of single_step")
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="number of trainer batch size"
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=128, help="number of test batch size"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )


    parser.add_argument(
        "--partition",
        type=str,
        default="noniid-labeldir",
        help="the data partitioning strategy of  cifar100,"
        ' select from "noniid-labeluni, noniid-labeldir,noniid-labeldir100"',
    )
  
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default="./logs/",
        help="Log directory path",
    )
    parser.add_argument(
        "--root", type=str, default="data0/Domain/", help="path to dataset"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/..", help="output directory"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument(
        "--eval-only", action="store_true", default=False, help="evaluation only"
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    setproctitle.setproctitle(
        "{}_{}_{}".format(args.trainer, args.backbone, args.dataset)
    )
    main(args)

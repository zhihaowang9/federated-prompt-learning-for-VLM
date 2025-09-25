import os.path as osp
import os
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.utils import Registry
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from Dassl.dassl.utils import (
    count_num_param, load_checkpoint,load_pretrained_weights
)


_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
      
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CLIP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"


        ctx_init = "a photo of a"
        ctx_init = ctx_init.replace("_", " ")
        
        prompt_prefix = ctx_init

        print(f'Initial context: "{prompt_prefix}"')


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

   

        self.n_cls = n_cls
        self.embedding = nn.Parameter(embedding).requires_grad_(False)

        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.CLIP.CLASS_TOKEN_POSITION

    def forward(self):
        prompts = self.embedding
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2) 
        x = self.transformer(x)
        x = x.permute(1, 0, 2) 
        x = self.ln_final(x).type(self.dtype)

  
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def forward(self, image):
        image_features,_ = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits,image_features,text_features



class CLIP(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.CLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(self.dm.dataset)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CLIP.PREC == "fp32" or cfg.TRAINER.CLIP.PREC == "amp":

            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
         
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        print(f"# params: {count_num_param(self.model):,}")
        print(f"# prompt learner params: {count_num_param(self.model.prompt_learner):,}")


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
       

        self.scaler = GradScaler() if cfg.TRAINER.CLIP.PREC == "amp" else None

      
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,3,2,1"
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
          
    def forward_backward(self,idx, batch, global_weight=None, fedprox=False, mu=0.5):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.CLIP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            if fedprox:
                model_weight = self.model.state_dict()
                fed_prox_reg = ((mu / 2) * torch.norm((model_weight['prompt_learner.ctx'] - global_weight['prompt_learner.ctx'])) ** 2)
                loss += fed_prox_reg
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }



        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()


        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model bash main.sh caltech101 rn50_ep50 end 16 1 Falsenot found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]


            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            self._models[name].load_state_dict(state_dict, strict=False)





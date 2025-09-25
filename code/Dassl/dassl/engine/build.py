from Dassl.dassl.utils import Registry, check_availability

from trainers.CLIP import CLIP

from trainers.LPT import LPT
from trainers.IVLP import IVLP
from trainers.VPT import VPT



TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(CLIP)

TRAINER_REGISTRY.register(LPT)
TRAINER_REGISTRY.register(IVLP)
TRAINER_REGISTRY.register(VPT)

def build_trainer(args,cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(args,cfg)
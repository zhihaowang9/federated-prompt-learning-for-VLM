from .defaults import _C as cfg_default


def get_cfg_default():
    return cfg_default.clone()


def clean_cfg(cfg, trainer):
 
    keys = list(cfg.TRAINER.keys())
    for key in keys:
        if key == "NAME" or key == trainer.upper():
            continue
        cfg.TRAINER.pop(key, None)

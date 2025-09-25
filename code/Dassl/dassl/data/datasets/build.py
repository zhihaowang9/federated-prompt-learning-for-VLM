from Dassl.dassl.utils import Registry, check_availability

from datasets.cifar100 import Cifar100

from datasets.office31 import Office31
from datasets.officehome import OfficeHome

from datasets.domainnet import DomainNet



DATASET_REGISTRY = Registry("DATASET")

DATASET_REGISTRY.register(Cifar100)

DATASET_REGISTRY.register(DomainNet)

DATASET_REGISTRY.register(Office31)
DATASET_REGISTRY.register(OfficeHome)
def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)

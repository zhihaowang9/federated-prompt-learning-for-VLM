import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from Dassl.dassl.utils import read_image
from Dassl.dassl.data.datasets import DATASET_REGISTRY

from collections import defaultdict

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform


def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    n_domain=0,
    n_ins=2,
    tfm=None,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
        batch_size=batch_size,
        n_domain=n_domain,
        n_ins=n_ins
    )

    if cfg.DATASET.NAME == "Cifar100" or cfg.DATASET.NAME == "Cifar10":
        dataset_wrapper = DatasetCifar100
    elif dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper


    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
   
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
     
        dataset = build_dataset(cfg)


        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        fed_train_loader_x_dict = defaultdict()
        if dataset.federated_train_x:
            for idx in range(cfg.DATASET.USERS):
                fed_train_loader_x = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                    data_source=dataset.federated_train_x[idx],
                    batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                    n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                    n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
                    tfm=tfm_train,
                    is_train=True,
                    dataset_wrapper=dataset_wrapper
                )
                fed_train_loader_x_dict[idx] = fed_train_loader_x

       
        fed_test_loader_x_dict = defaultdict()
        if dataset.federated_test_x:
            for idx in range(cfg.DATASET.USERS):
                fed_test_loader_x = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.federated_test_x[idx],
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    dataset_wrapper=dataset_wrapper
                )
                fed_test_loader_x_dict[idx] = fed_test_loader_x

      
        if any(isinstance(i, list) for i in dataset.data_test):
            test_loader = defaultdict()
            for idx in range(len(dataset.data_test)):
                fed_test_loader = build_data_loader(
                    cfg,
                    sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                    data_source=dataset.data_test[idx],
                    batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                    tfm=tfm_test,
                    is_train=False,
                    dataset_wrapper=dataset_wrapper
                )
                test_loader[cfg.DATASET.SOURCE_DOMAINS[idx]] = fed_test_loader
        else:
            test_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.data_test,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

    
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname
        self.classnames = dataset.classnames


        self.dataset = dataset
        self.test_loader = test_loader
        self.fed_train_loader_x_dict = fed_train_loader_x_dict
        self.fed_test_loader_x_dict = fed_test_loader_x_dict

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def num_source_domains(self):
        return self._num_source_domains

    @property
    def lab2cname(self):
        return self._lab2cname

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME
        source_domains = cfg.DATASET.SOURCE_DOMAINS
        target_domains = cfg.DATASET.TARGET_DOMAINS

        table = []
        table.append(["Dataset", dataset_name])
        if source_domains:
            table.append(["Source", source_domains])
        if target_domains:
            table.append(["Target", target_domains])
        table.append(["# classes", f"{self.num_classes:,}"])
   
        if self.dataset.federated_train_x:
            table.append(["# federated_train_x", f"{len(self.dataset.federated_train_x):,}"])
       
        table.append(["# federated_test_x", f"{len(self.dataset.federated_test_x):,}"])

        table.append(["# test", f"{len(self.dataset.data_test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform 
        self.is_train = is_train
      
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

     
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            # "domain": item.domain,
            "impath": item.impath
        }

        img0 = read_image(item.impath)


        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output["img"] = img
        else:
            output["img"] = img0

        if self.return_img0:
            output["img0"] = self.to_tensor(img0) 
        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

class DatasetCifar100(TorchDataset):

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        self.cfg = cfg
        self.data_source = data_source
        self.transform = transform  
        self.is_train = is_train
     
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                "Cannot augment the image {} times "
                "because transform is None".format(self.k_tfm)
            )

   
        interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
        to_tensor = []
        to_tensor += [T.Resize(cfg.INPUT.SIZE, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        if "normalize" in cfg.INPUT.TRANSFORMS:
            normalize = T.Normalize(
                mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
            )
            to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            "label": item.label,
            "domain": item.domain
        }
        img0 = item.impath

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = "img"
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                if self.cfg.DATASET.NAME == "Cifar100":
                    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                        T.Resize([224, 224])
                    ])
                    img = transform(img0)
                elif self.cfg.DATASET.NAME == "Cifar10":
                    transform = T.Compose([
                        T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        T.Resize([224, 224])
                    ])
                    img = transform(img0)

                else:
                    img = self._transform_image(self.transform, img0)

                output["img"] = img
        else:
            output["img"] = img0

        return output

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

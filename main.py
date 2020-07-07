import copy
import numbers
from pathlib import Path

import numpy as np
from deepclustering.augment import SequentialWrapper, pil_augment
from deepclustering.dataset import PatientSampler
from deepclustering.manager import ConfigManger
from distributed.protocol.tests.test_torch import torch
from torch.utils.data import DataLoader

from contrastyou import CONFIG_PATH, DATA_PATH
from contrastyou.augment import SequentialWrapperTwice
from contrastyou.dataloader._seg_datset import ContrastBatchSampler
from contrastyou.dataloader.acdc_dataset import ACDCDataset
from contrastyou.modules.model import DPModule as Model

config = ConfigManger(Path(CONFIG_PATH) / "config.yaml", integrality_check=False, verbose=False).config

config2 = copy.deepcopy(config)
config2["Scheduler"]["gamma"]=0

model = Model(arch=config["Arch"], optimizer=config["Optim"], scheduler=config["Scheduler"])
model2 = Model(arch=config2["Arch"], optimizer=config2["Optim"], scheduler=config2["Scheduler"])

train_transform = SequentialWrapper(
    pil_augment.Compose([
        pil_augment.RandomRotation(40),
        pil_augment.RandomHorizontalFlip(),
        pil_augment.RandomCrop(224),
        pil_augment.ToTensor()
    ]),
    pil_augment.Compose([
        pil_augment.RandomRotation(40),
        pil_augment.RandomHorizontalFlip(),
        pil_augment.RandomCrop(224),
        pil_augment.ToLabel()
    ]),
    (False, True)
)
val_transform = SequentialWrapper(
    pil_augment.Compose([
        pil_augment.CenterCrop(224),
        pil_augment.ToTensor()
    ]),
    pil_augment.Compose([
        pil_augment.CenterCrop(224),
        pil_augment.ToLabel()
    ]),
    (False, True)
)
dataset = ACDCDataset(root_dir=DATA_PATH, mode="train", transforms=SequentialWrapperTwice(train_transform))
batch_sampler = ContrastBatchSampler(dataset, group_sample_num=8, partition_sample_num=1)
train_loader = DataLoader(dataset, batch_sampler=batch_sampler)
val_dataset = ACDCDataset(root_dir=DATA_PATH, mode="val", transforms=val_transform)
val_batch_sampler = PatientSampler(val_dataset, grp_regex=val_dataset.dataset_pattern, shuffle=False, )
val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler)

# class Epoch:




class Trainer:
    def __init__(self, model, train_loader, val_loader, max_epoch, device, config) -> None:
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._max_epoch = max_epoch
        self._device = device
        self._config = config
        self._begin_epoch = 0
        self._best_score = -1

    def state_dict(self):
        """
        return trainer's state dict. The dict is built by considering all the submodules having `state_dict` method.
        """
        state_dictionary = {}
        for module_name, module in self.__dict__.items():
            if hasattr(module, "state_dict"):
                state_dictionary[module_name] = module.state_dict()
            elif isinstance(module, (numbers.Number, str, torch.Tensor, np.ndarray)):
                state_dictionary[module_name] = module
        return state_dictionary

    def state_dict2(self):
        return self.__dict__

    def load_state_dict(self, state_dict) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :return:
        """
        for module_name, module in self.__dict__.items():
            if hasattr(module, "load_state_dict"):
                try:
                    module.load_state_dict(state_dict[module_name])
                except KeyError as e:
                    print(f"Loading checkpoint error for {module_name}, {e}.")
                except RuntimeError as e:
                    print(f"Interface changed error for {module_name}, {e}")
            elif isinstance(module, (numbers.Number, str, torch.Tensor, np.ndarray)):
                self.__dict__[module_name] = state_dict[module_name]
    def load_state_dict2(self, state_dict):
        self.__dict__.update(state_dict)

trainer = Trainer(model, train_loader, val_loader, 100, "cuda", config)
trainer._best_score=torch.Tensor([10000000])
trainer._best_epoch=1232
trainer._big=123
trainer2 = Trainer(model2, train_loader, val_loader, 200, "cpu", config2)
state_dict = trainer.state_dict()
from torchvision.models import vgg11_bn
model1 = vgg11_bn()
model2 = vgg11_bn()

model2.load_state_dict(model1.state_dict())

trainer2.load_state_dict(state_dict)
## with this method, the id of the two items are the same..
print(trainer2.__dict__)
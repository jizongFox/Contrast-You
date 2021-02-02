from .base import *
from .pretrain import *
from .trainer import *

base_trainer_zoos = {
    "partial": SemiTrainer,
    "finetune": FineTuneTrainer,
    "uda": UDATrainer,
    "iic": IICTrainer,
    "mine": MineTrainer,
    "udaiic": UDAIICTrainer,
    "entropy": EntropyMinTrainer,
    "meanteacher": MeanTeacherTrainer,
    "ucmeanteacher": UCMeanTeacherTrainer,
    "iicmeanteacher": IICMeanTeacherTrainer,
    "midl": MIDLTrainer,
    "infonce": InfoNCETrainer,
    "prototype": PrototypeTrainer,
    "experiment": ExperimentalTrainer,
    "experiment2": ExperimentalTrainer2

}
pre_trainer_zoos = {
    "udaiicpretrain": PretrainUDAIICTrainer,
    "iicpretrain": PretrainIICTrainer,
    "infoncepretrain": PretrainInfoNCETrainer,
    "experimentpretrain": PretrainExperimentalTrainer,
    "experimentpretrain2": PretrainExperimentalTrainer2
}

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}

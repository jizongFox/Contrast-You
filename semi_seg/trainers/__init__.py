from .base import *
from .pretrain import *
from .proposedtrainer import *
from .trainer import *

base_trainer_zoos = {
    "directtrain": DirectTrainer,
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
    "experiment2": ExperimentalTrainer2,
    "experiment3": ExperimentalTrainer3,

}
pre_trainer_zoos = {
    "udaiicpretrain": PretrainUDAIICTrainer,
    "iicpretrain": PretrainIICTrainer,
    "infoncepretrain": PretrainInfoNCETrainer,
    "experimentpretrain": PretrainExperimentalTrainer,
    "experimentpretrain2": PretrainExperimentalTrainer2,
    "experimentpretrain3": PretrainExperimentalTrainer3,
}

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}

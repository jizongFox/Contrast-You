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
    "mixup": ExperimentalTrainerwithMixUp,
    "multitask": ExperimentalTrainerwithMultiTask,

}
pre_trainer_zoos = {
    "udaiicpretrain": PretrainUDAIICTrainer,
    "iicpretrain": PretrainIICTrainer,
    "infoncepretrain": PretrainInfoNCETrainer,
    "experimentpretrain": PretrainExperimentalTrainer,
    "experimentmixuppretrain": PretrainExperimentalMixupTrainer,
    "experimentmultitaskpretrain": PretrainExperimentalMultiTaskTrainer
}

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}

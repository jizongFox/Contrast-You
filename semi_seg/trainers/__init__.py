from .base import *
from .pretrain import *
from .trainer import *

base_trainer_zoos = {
    "partial": SemiTrainer,
    "uda": UDATrainer,
    "iic": IICTrainer,
    "mine":MineTrainer,
    "udaiic": UDAIICTrainer,
    "entropy": EntropyMinTrainer,
    "meanteacher": MeanTeacherTrainer,
    "ucmeanteacher": UCMeanTeacherTrainer,
    "iicmeanteacher": IICMeanTeacherTrainer,
    "midl": MIDLTrainer,
    # "featureoutputiic": IICFeatureOutputTrainer,
    # "featureoutputudaiic": UDAIICFeatureOutputTrainer,
    # "pica": PICATrainer,
    "infonce": InfoNCETrainer,
    # "infonce_demo": InfoNCETrainerDemo,
    "prototype": PrototypeTrainer,
    "dp": DifferentiablePrototypeTrainer,

}
pre_trainer_zoos = {
    "udaiicpretrain": PretrainUDAIICTrainer,
    "iicpretrain": PretrainIICTrainer,
    "infoncepretrain": PretrainInfoNCETrainer
}

trainer_zoos = {**base_trainer_zoos, **pre_trainer_zoos}

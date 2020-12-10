from .trainer import *
from .trainer_pre import *

trainer_zoos = {
    "partial": SemiTrainer,
    "uda": UDATrainer,
    "iic": IICTrainer,
    "udaiic": UDAIICTrainer,
    "entropy": EntropyMinTrainer,
    "meanteacher": MeanTeacherTrainer,
    "ucmeanteacher": UCMeanTeacherTrainer,
    "iicmeanteacher": IICMeanTeacherTrainer,
    "midl": MIDLTrainer,
    "featureoutputiic": IICFeatureOutputTrainer,
    "featureoutputudaiic": UDAIICFeatureOutputTrainer,
    "pica": PICATrainer,
    "infonce": InfoNCETrainer,
    "infonce_demo": InfoNCETrainerDemo,
    "infoncepretrain": InfoNCEPretrainTrainer,
    "prototype": PrototypeTrainer,
    "dp": DifferentiablePrototypeTrainer,

    "understandps": UnderstandPSTrainer
}

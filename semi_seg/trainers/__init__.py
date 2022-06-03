from semi_seg.trainers.pretrain import PretrainEncoderTrainer, PretrainDecoderTrainer
from semi_seg.trainers.trainer import SemiTrainer, FineTuneTrainer, MixUpTrainer, MTTrainer, DMTTrainer

trainer_zoo = {"semi": SemiTrainer,
               "ft": FineTuneTrainer,
               "pretrain": PretrainEncoderTrainer,
               "pretrain_decoder": PretrainDecoderTrainer,
               "mt": MTTrainer,
               "dmt": DMTTrainer,
               "mixup": MixUpTrainer}

import typing as t

from contrastyou.trainer.base import Trainer as _Trainer
from semi_seg.trainers.pretrain import PretrainEncoderTrainer, PretrainDecoderTrainer
from semi_seg.trainers.trainer import SemiTrainer, FineTuneTrainer, MixUpTrainer, MTTrainer, DMTTrainer

trainer_zoo: t.Dict[str, t.Type[_Trainer]] = {
    "semi": SemiTrainer,
    "ft": FineTuneTrainer,
    "pretrain": PretrainEncoderTrainer,
    "pretrain_decoder": PretrainDecoderTrainer,
    "mt": MTTrainer,
    "dmt": DMTTrainer,
    "mixup": MixUpTrainer
}

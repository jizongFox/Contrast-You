from .base_trainer import FSTrainer, SemiTrainer
from .contrast_trainer import ContrastTrainer

trainer_zoos = {"fs": FSTrainer, "semi": SemiTrainer, "contrast": ContrastTrainer}

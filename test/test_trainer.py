import torch

from contrastyou.modules.model import Model
from contrastyou.trainer._trainer import _Trainer

model = Model({"name": "enet"})
trainer = _Trainer(model, None, None, 100, save_dir="tmp", checkpoint="23", device="cuda", config={"1": 2})
state_dict = trainer.state_dict()
print("1")
model2 = Model({"name": "enet"})
trainer2 = _Trainer(model2, None, None, 200, save_dir="tmp2", checkpoint="234", device="cpu", config={"1": 4})
trainer2.load_state_dict(state_dict)
assert id(trainer._model._torchnet.parameters().__next__())!= id(trainer2._model._torchnet.parameters().__next__())
assert torch.allclose(trainer._model._torchnet.parameters().__next__(),trainer2._model._torchnet.parameters().__next__())



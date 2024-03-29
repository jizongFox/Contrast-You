import contextlib
import typing as t
import weakref
from abc import abstractmethod
from contextlib import nullcontext
from functools import wraps

from torch import nn

from contrastyou.nn import ModuleBase
from contrastyou.utils import class_name

if t.TYPE_CHECKING:
    from contrastyou.epochers.base import EpocherBase
    from contrastyou.meters import MeterInterface
    from torch.nn import Parameter
    from contrastyou.trainer import Trainer


class _ClassNameMeta(type):
    """
    This meta class is to make sure that the training hook cannot have the same name in a single experiment.
    if two hooks with the same name is given, a `HookNameExistError` would be raised to stop the algorithm.
    """
    names: t.Set[str] = set()

    def __call__(cls, *args, **kwargs):
        # sourcery skip: instance-method-first-arg-name
        if "hook_name" in kwargs:
            hook_name = kwargs["hook_name"]
            if hook_name in cls.names:
                raise HookNameExistError(hook_name)
            cls.names.add(hook_name)
        return super(_ClassNameMeta, cls).__call__(*args, **kwargs)


class TrainerHook(ModuleBase, metaclass=_ClassNameMeta):

    def __init__(self, *, hook_name: str):
        super().__init__()
        self._hook_name = hook_name
        self._initialized = False

    @t.final
    def parameters(self, recurse: bool = True) -> t.Iterator['Parameter']:
        for m in self.learnable_modules:
            yield from m.parameters(recurse=recurse)

    @property
    def learnable_modules(self) -> t.List[nn.Module]:
        return []

    def __call__(self, **kwargs):
        raise NotImplementedError(f"subclass {class_name(self)} must implement __call__ function.")

    def close(self):
        pass

    def after_initialize(self):
        pass

    """
    @t.final
    @property
    def trainer(self):
        if self._initialized:
            return self._trainer
        raise RuntimeError(f"{class_name(self)} not initialized yet.")

    @t.final
    @trainer.setter
    def trainer(self, trainer: 'Trainer'):
        self._initialized = True
        self.register_trainer(trainer)
"""

    def register_trainer(self, trainer: 'Trainer'):
        self._initialized = True
        self.register_non_trackable_buffer("trainer", trainer)
        self.register_non_trackable_buffer("_trainer", trainer)


class CombineTrainerHook(TrainerHook):

    def __init__(self, *trainer_hook: 'TrainerHook'):
        super().__init__(hook_name="")
        self._hooks = nn.ModuleList(trainer_hook)

    def __call__(self):
        return CombineEpochHook(*[h() for h in self._hooks])

    @property
    def learnable_modules(self):
        return self._hooks

    def close(self):
        for h in self._hooks:
            h.close()

    """
    @t.final
    @property
    def trainer(self):
        for h in self._hooks:
            if h._initialized:  # noqa
                return h.trainer
        raise RuntimeError(f"{class_name(self)} not initialized yet.")

    @t.final
    @trainer.setter
    def trainer(self, trainer: 'Trainer'):
        for h in self._hooks:
            h.trainer = trainer
    """

    @t.final
    @property
    def trainer(self):
        for h in self._hooks:
            if h._initialized:  # noqa
                return h.trainer
        raise RuntimeError(f"{class_name(self)} not initialized yet.")

    def register_trainer(self, trainer: 'Trainer'):
        for h in self._hooks:
            h.register_non_trackable_buffer("trainer", trainer)
            h.register_non_trackable_buffer("_trainer", trainer)

    def after_initialize(self):
        for h in self._hooks:
            h.after_initialize()


class EpocherHook:

    def __init__(self, *, name: str, ) -> None:
        self._name = name
        self._epocher = None
        self.meters: t.Optional['MeterInterface'] = None
        self._epocher_init = False

    @t.final
    @property
    def epocher(self):
        if self._epocher_init:
            return self._epocher
        raise HookNotInitializedError(f"{self._name} not initialized yet.")

    @t.final
    @epocher.setter
    def epocher(self, epocher: "EpocherBase"):
        self._epocher = weakref.proxy(epocher)
        self.meters = weakref.proxy(epocher.meters)
        self._epocher_init = True
        with self.meters.focus_on(self.name):
            self.configure_meters_given_epocher(self.meters)

    def configure_meters_given_epocher(self, meters: 'MeterInterface') -> 'MeterInterface':
        return meters

    # calling interface for epocher.
    @t.final
    def call_before_batch_update(self, **kwargs):
        assert self._epocher_init
        with self.context:
            return self.before_batch_update(**kwargs)

    @t.final
    def call_before_forward_pass(self, **kwargs):
        assert self._epocher_init
        with self.context:
            return self.before_forward_pass(**kwargs)

    @t.final
    def call_after_forward_pass(self, **kwargs):
        assert self._epocher_init
        with self.context:
            return self.after_forward_pass(**kwargs)

    @t.final
    def call_before_regularization(self, **kwargs):
        assert self._epocher_init
        with self.context:
            return self.before_regularization(**kwargs)

    @t.final
    def call_after_regularization(self, **kwargs):
        assert self._epocher_init
        with self.context:
            return self.after_regularization(**kwargs)

    @t.final
    def call_after_batch_update(self, **kwargs):
        assert self._epocher_init
        with self.context:
            return self.after_batch_update(**kwargs)

    # real implementations
    def before_batch_update(self, **kwargs):
        pass

    def before_forward_pass(self, **kwargs):
        pass

    def after_forward_pass(self, **kwargs):
        pass

    def before_regularization(self, **kwargs):
        pass

    def after_regularization(self, **kwargs):
        pass

    def after_batch_update(self, **kwargs):
        pass

    @t.final
    def __call__(self, **kwargs):
        assert self._epocher_init
        with self.context:
            return self._call_implementation(**kwargs)

    @abstractmethod
    def _call_implementation(self, **kwargs):
        ...

    def close(self):
        pass

    @property
    def name(self):
        return self._name

    @property
    def context(self) -> contextlib.AbstractContextManager:
        context = nullcontext()
        if self.meters:
            context = self.meters.focus_on(self._name)
        return context


class CombineEpochHook(EpocherHook):
    def __init__(self, *epocher_hook: EpocherHook) -> None:  # noqa
        self._epocher_hook = epocher_hook

    @t.final
    def call_before_forward_pass(self, **kwargs):  # noqa
        for h in self._epocher_hook:
            h.call_before_forward_pass(**kwargs)

    @t.final
    def call_after_forward_pass(self, **kwargs):  # noqa
        for h in self._epocher_hook:
            h.call_after_forward_pass(**kwargs)

    @t.final
    def call_before_regularization(self, **kwargs):  # noqa
        for h in self._epocher_hook:
            h.call_before_regularization(**kwargs)

    @t.final
    def call_after_regularization(self, **kwargs):  # noqa
        for h in self._epocher_hook:
            h.call_after_regularization(**kwargs)

    @t.final
    def call_before_batch_update(self, **kwargs):  # noqa
        for h in self._epocher_hook:
            h.call_before_batch_update(**kwargs)

    @t.final
    def call_after_batch_update(self, **kwargs):  # noqa
        for h in self._epocher_hook:
            h.call_after_batch_update(**kwargs)

    @t.final
    def __call__(self, **kwargs):  # noqa just to modify it once.
        return sum(h(**kwargs) for h in self._epocher_hook)

    def _call_implementation(self, **kwargs):
        raise NotImplementedError()

    def close(self):
        for h in self._epocher_hook:
            h.close()

    @t.final
    @property
    def epocher(self):
        for h in self._epocher_hook:
            return h._epocher

    @t.final
    @epocher.setter
    def epocher(self, epocher: "EpocherBase"):
        for h in self._epocher_hook:
            h.epocher = epocher


def meter_focus(_func=None, *, attribute="_name"):
    def decorator_focus(func):
        @wraps(func)
        def wrapper_focus(self, *args, **kwargs):
            with self.meters.focus_on(getattr(self, attribute)):
                return func(self, *args, **kwargs)

        return wrapper_focus

    if _func is None:
        return decorator_focus
    else:
        return decorator_focus(_func)


class HookNameExistError(Exception):
    pass


class HookNotInitializedError(Exception):
    pass

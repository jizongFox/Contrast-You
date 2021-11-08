# this hook collaborates with the Epocher to provide a scalable thing.
import contextlib
import typing as t
import weakref
from abc import abstractmethod
from contextlib import nullcontext
from functools import wraps

from torch import nn
from torch.nn import Parameter

from contrastyou.meters import MeterInterface
from contrastyou.utils import class_name


class HookNameExistError(Exception):
    pass


class _ClassNameMeta(type):
    """
    This meta class is to make sure that the training hook cannot have the same name in a single experiment.
    if two hooks with the same name is given, a RuntimeError would be raise to stop the algorithm.
    """
    names: t.List[str] = []

    def __call__(cls, *args, **kwargs):
        if "hook_name" in kwargs:
            hook_name = kwargs["hook_name"]
            if hook_name in cls.names:
                raise HookNameExistError(hook_name)
            cls.names.append(hook_name)
        return super(_ClassNameMeta, cls).__call__(*args, **kwargs)


class TrainerHook(nn.Module, metaclass=_ClassNameMeta):

    def __init__(self, *, hook_name: str):
        super().__init__()
        self._hook_name = hook_name

    def parameters(self, recurse: bool = True) -> t.Iterator[Parameter]:
        for m in self.learnable_modules:
            yield from m.parameters(recurse=recurse)

    @property
    def learnable_modules(self) -> t.List[nn.Module]:
        return []

    def __call__(self, **kwargs):
        raise NotImplementedError(f"subclass {class_name(self)} must implement __call__ function.")

    def close(self):
        pass


class CombineTrainerHook(TrainerHook):

    def __init__(self, *trainer_hook: TrainerHook):
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


class EpocherHook:

    def __init__(self, *, name: str, ) -> None:
        self._name = name
        self._epocher = None
        self.meters = None
        self.__epocher_set__ = False

    def set_epocher(self, epocher):
        """not necessary to be called, but need to be called with training with the epocher."""
        self._epocher = weakref.proxy(epocher)
        self.meters = weakref.proxy(epocher.meters)
        self.__epocher_set__ = True

    @property
    def epocher(self):
        return self._epocher

    def set_meters_given_epocher(self):
        assert self.__epocher_set__, f"epocher not set to {class_name(self)}."
        with self.meters.focus_on(self.name):
            self.configure_meters_given_epocher(self.meters)

    def configure_meters_given_epocher(self, meters: MeterInterface):
        return meters

    # calling interface for epocher.
    @t.final
    def call_before_batch_update(self, **kwargs):
        with self.context:
            return self.before_batch_update(**kwargs)

    @t.final
    def call_before_forward_pass(self, **kwargs):
        with self.context:
            return self.before_forward_pass(**kwargs)

    @t.final
    def call_after_forward_pass(self, **kwargs):
        with self.context:
            return self.after_forward_pass(**kwargs)

    @t.final
    def call_before_regularization(self, **kwargs):
        with self.context:
            return self.before_regularization(**kwargs)

    @t.final
    def call_after_regularization(self, **kwargs):
        with self.context:
            return self.after_regularization(**kwargs)

    @t.final
    def call_after_batch_update(self, **kwargs):
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

    def set_meters_given_epocher(self):
        for h in self._epocher_hook:
            h.set_meters_given_epocher()

    def set_epocher(self, epocher):
        for h in self._epocher_hook:
            h.set_epocher(epocher)

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
        return sum([h(**kwargs) for h in self._epocher_hook])

    def _call_implementation(self, **kwargs):
        raise NotImplementedError()

    def close(self):
        for h in self._epocher_hook:
            h.close()


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

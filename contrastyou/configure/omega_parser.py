import argparse
import itertools
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import List, Union

from omegaconf import OmegaConf, DictConfig
from prettytable import PrettyTable
from prettytable.colortable import ColorTable, Themes  # noqa

__config_dictionary__: OrderedDict = OrderedDict()


def dict_config_has_key(config: DictConfig, name: str) -> bool:
    impossible_value = str(uuid.uuid4())
    tried_value = OmegaConf.select(config, name, default=impossible_value)
    return tried_value != impossible_value


class OmegaParser:
    def __init__(self, *path: str, check_missing: bool = True) -> None:
        self._check_missing = check_missing

        args = self._setup()
        config_path = args.path or [list(path)]
        self._config_paths: List[str] = list(itertools.chain(*config_path))
        cmd_str_list = [x.strip() for x in itertools.chain(*args.optional_variables)] if args.optional_variables else []
        self._cmd_str_list: List[str] = cmd_str_list

    @property
    def base_config(self) -> DictConfig:
        """
        return base config provided by --path
        """
        if self._config_paths:
            config_list = [self.load_yaml(p) for p in self._config_paths]
            config = OmegaConf.merge(*config_list)
        # no configure loaded
        else:
            config = OmegaConf.create()
        OmegaConf.set_struct(config, True)
        return config

    @staticmethod
    def _cli_merge(config: DictConfig, cmd_str: str) -> DictConfig:
        if cmd_str.startswith("~"):
            name = cmd_str[1:]
            OmegaConf.set_struct(config, False)

            def del_path(*path, _config):
                if len(path) == 1:
                    del _config[path[0]]
                    return
                return del_path(*path[1:], _config=_config[path[0]], )

            del_path(*name.split("."), _config=config, )

            OmegaConf.set_struct(config, True)
            return config
        assert len(cmd_str.split("=")) == 2, "cmd_str should be key=value"
        if cmd_str.startswith("+"):
            name, value = cmd_str[1:].split("=")
            if dict_config_has_key(config, name):
                raise ValueError(f"{name} already exists, remove '+' ")
            OmegaConf.set_struct(config, False)
            config = OmegaConf.merge(config, OmegaConf.from_cli([cmd_str[1:]]))
            OmegaConf.set_struct(config, True)
        else:
            name, value = cmd_str.split("=")
            if not dict_config_has_key(config, name):
                raise ValueError(f"{name} does not exist, please add `+` before {name}={value}")
            config = OmegaConf.merge(config, OmegaConf.from_cli([cmd_str]))

        return config

    @property
    def cmd_config(self):
        """
        merge cmd variables to base config
        """
        if self._cmd_str_list:
            return OmegaConf.from_cli(self._cmd_str_list)
        return OmegaConf.create()

    @staticmethod
    def load_yaml(path: Union[str, Path]) -> DictConfig:
        """
        load yaml file and return as dict
        """
        with open(path, "r") as fp:
            loaded = OmegaConf.load(fp.name)
        return loaded

    @staticmethod
    def save_yaml(config: DictConfig, path: str):
        with open(path, "w") as fp:
            OmegaConf.save(config, fp)

    @staticmethod
    def _setup() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            "Augment parser for yaml config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        path_parser = parser.add_argument_group("config path parser")
        path_parser.add_argument(
            "-p", "--path", type=str, required=False, default=None, nargs=argparse.ZERO_OR_MORE, action='append',
            help="base config path location",
        )
        cmd_parser = parser.add_argument_group("cmd parser")
        cmd_parser.add_argument("-o", "--optional_variables", nargs=argparse.ZERO_OR_MORE, type=str, default=None,
                                action='append',
                                help="commandline based variables to override config file")
        return parser.parse_args()

    def parse_args(self) -> DictConfig:
        """
        merge base config and cmd config
        """
        config = self.base_config
        for cmd_str in self._cmd_str_list:
            config = self._cli_merge(config, cmd_str)
        OmegaConf.set_readonly(config, True)
        if self._check_missing:
            if missing_key := OmegaConf.missing_keys(config):
                raise ValueError(f"missing keys: {','.join(missing_key)}")
        OmegaConf.resolve(config)
        return config

    @contextmanager
    def __call__(self, scope="base"):
        assert scope not in __config_dictionary__, scope
        config = self.parse_args()
        __config_dictionary__[scope] = config
        try:
            yield config
        finally:
            del __config_dictionary__[scope]

    @property
    def parsed_config(self):
        return self.parse_args()

    config = parsed_config

    @staticmethod
    @contextmanager
    def struct_cxm(config: DictConfig, enable: bool):
        prev = OmegaConf.is_struct(config)
        OmegaConf.set_struct(config, enable)
        try:
            yield config
        finally:
            OmegaConf.set_struct(config, prev)

    @staticmethod
    @contextmanager
    def writable_cxm(config: DictConfig, enable: bool):
        prev = OmegaConf.is_readonly(config)
        OmegaConf.set_readonly(config, not enable)
        try:
            yield config
        finally:
            OmegaConf.set_readonly(config, prev)

    @staticmethod
    @contextmanager
    def modifiable_cxm(config: DictConfig, enable: bool):
        prev_read = OmegaConf.is_readonly(config)
        prev_str = OmegaConf.is_struct(config)

        OmegaConf.set_readonly(config, not enable)
        OmegaConf.set_struct(config, not enable)
        try:
            yield config
        finally:
            OmegaConf.set_readonly(config, prev_read)
            OmegaConf.set_struct(config, prev_str)

    @staticmethod
    def set_modifiable(config: DictConfig, enable: bool) -> None:
        OmegaConf.set_readonly(config, not enable)
        OmegaConf.set_struct(config, not enable)

    def summary(self, colored=True) -> PrettyTable:
        table = ColorTable(theme=Themes.OCEAN) if colored else PrettyTable()
        table._max_width = {"Base Params": 50, "CMD-parsed params:": 55, "Merged params:": 50}
        table.add_column("Base params:",
                         [OmegaConf.to_yaml(self.base_config)], align="l")
        table.add_column("CMD-parsed params:", [OmegaConf.to_yaml(self.cmd_config), ], align="l", valign="t")
        table.add_column("Merged params:",
                         [OmegaConf.to_yaml(self.config), ], align="l")
        return table

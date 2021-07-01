# this is to compare experimental data cross different folders
import argparse
from pathlib import Path
from pprint import pprint
from typing import List, Dict

import numpy as np
import pandas as pd

from contrastyou.configure import dictionary_merge_by_hierachy
from contrastyou.configure.yaml_parser import str2bool


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report results from different folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    filepath_group = parser.add_mutually_exclusive_group(required=True)
    filepath_group.add_argument(
        "--specific_folders",
        "-s",
        type=str,
        nargs="+",
        help="list specific folders.",
        metavar="PATH",
    )
    filepath_group.add_argument(
        "--top_folder", "-t", type=str, help="top folder.", metavar="PATH"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        help="Targeted class in the .csv file.",
        required=True,
    )
    parser.add_argument(
        "--anchor", type=str, help="anchor class to rank and select rows.", default=None
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        nargs="+",
        default=["*/*.csv"],
        metavar="FILENAME or REGEX",
        help=".csv file name or regex rules.",
    )
    parser.add_argument(
        "--high_better",
        type=str2bool,
        nargs="+",
        help="is the class value is high is better. default True,"
             "if given, high_better must have the same size as classes.",
        default=True,
    )
    parser.add_argument("--save_dir", type=str, help="save summary dir.", required=True)
    parser.add_argument(
        "--save_filename", type=str, help="save summary name", default="summary.csv"
    )
    args = parser.parse_args()
    if isinstance(args.high_better, list):
        assert len(args.high_better) == len(args.classes), (
            f"high_better must correspond to classes, "
            f"given classes: {args.classes} and high_better: {args.high_better}."
        )
    # anchor class
    if args.anchor is None:
        if len(args.classes) == 1:
            args.anchor = args.classes[0]
        else:
            raise ValueError(
                f"archor should be provided given {len(args.classes)} classes."
            )
    assert args.anchor, args.anchor
    print(vars(args))
    return args


def _search_files(args):
    if args.top_folder is not None:
        # a top folder is provided.
        csvfile_paths: List[Path] = []
        for filename in args.file:
            csvfile_paths.extend(list(Path(args.top_folder).rglob(f"{filename}")))
    else:
        # several top folders are provided:
        csvfile_paths = []
        for path in args.specific_folders:
            for filename in args.file:
                csvfile_paths.extend(list(Path(path).rglob(f"{filename}")))
    return csvfile_paths


def main(args: argparse.Namespace):
    csvfile_paths = _search_files(args)
    assert len(csvfile_paths) > 0, f"Found 0 {args.file} file."
    print(f"Found {len(csvfile_paths)} {args.file} files, e.g.,")
    pprint(csvfile_paths[:5])
    path_features = extract_path_info(csvfile_paths)

    values: Dict[str, Dict[str, float]] = {
        str(p): dict(
            zip(
                args.classes,
                extract_value_with_anchor(
                    p, args.classes, args.anchor, args.high_better
                ),
            )
        )
        for p in csvfile_paths
    }

    table = pd.DataFrame(dictionary_merge_by_hierachy(path_features, values)).T
    print(table)
    table.to_csv(Path(args.save_dir, args.save_filename))


def extract_value(file_path, class_name, is_high=True):
    try:
        if is_high:
            return pd.read_csv(file_path)[class_name].max()
        else:
            return pd.read_csv(file_path)[class_name].min()
    except KeyError:
        return np.nan


def extract_value_with_anchor(file_path, classes, anchor, is_high=True):
    assert anchor in classes, (anchor, classes)
    try:
        file = pd.read_csv(file_path)
        right_idx = file[anchor].idxmax() if is_high else file[anchor].idxmin()
        right_row = file.iloc[right_idx, :]
        return right_row[classes].tolist()
    except KeyError:
        return np.nan


def extract_path_info(file_paths: List[Path]) -> List[List[str]]:
    # return the list of path features for all the file_paths
    def split_path(file_path: str, sep="/") -> List[str]:
        parents: List[str] = file_path.split(sep)[:-1]
        return parents

    assert (
               set([len(split_path(str(p))) for p in file_paths])
           ).__len__() == 1, f"File paths must have located in a structured way."
    parents_path = []
    for i, p in enumerate(file_paths):
        parents_path.append(split_path(str(p)))

    path_begin: int = (pd.DataFrame(parents_path).nunique(axis=0) > 1).values.argmax()
    return {
        str(p): {
            f"feature_{i}": _p for i, _p in enumerate(split_path(str(p))[path_begin:])
        }
        for p in file_paths
    }


def call_from_cmd():
    import sys
    import subprocess

    calling_folder = str(subprocess.check_output("pwd", shell=True))
    sys.path.insert(0, calling_folder)
    args = arg_parser()
    main(args)


if __name__ == "__main__":
    args = arg_parser()
    main(args)

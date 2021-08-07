import numpy as np
import os
import re
from pathlib import Path
from typing import Union, List


def extract_suject_num(path: Union[str, Path]) -> str:
    path = str(path)
    # Spleen
    subject_num = re.compile(r"\d+").findall(path)[0].split("/")[0]

    return subject_num


def glob_folder(current_folder: Path, postfix: str) -> List[str]:
    return sorted(
        set([extract_suject_num(x) for x in current_folder.rglob(f"*.{postfix}")])
    )


if __name__ == "__main__":
    image_root = Path('.data/Spleen/train/img')
    dataset_postfix = "png"
    content_list = glob_folder(image_root, dataset_postfix)
    imgs_list = os.listdir(image_root)
    info_all =[]

    info_all = {f'Patient_{scan_num}': sum(img_q[0:10] == f'Patient_{scan_num}' for img_q in imgs_list) for scan_num in content_list}

    np.save('spleen_info', info_all)
    np.load(("spleen_info.npy"), allow_pickle=True).item()


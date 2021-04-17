import os
from pathlib import Path

root_path = Path("./")


def path2Path(path):
    return Path(path) if isinstance(path, str) else path


def find_checkpoint(root, category="*/tra/ratio_0.01"):
    sorted_checkpoint = sorted([x for x in root.rglob(f"{category}/best.pth")])
    return sorted_checkpoint


def find_folder(checkpoint_path):
    return checkpoint_path.parent.absolute()


def main(root_path):
    root_path = path2Path(root_path)
    checkpoint_list = find_checkpoint(root_path, "*/baseline/tra/ratio_0.13")
    for c in checkpoint_list:
        save_dir = find_folder(c)
        print(c)
        os.system(
            f"/opt/anaconda3/bin/python  ../../inference.py Trainer.save_dir={str(save_dir)} "
            f"Trainer.name=finetune  Arch.checkpoint={str(c.absolute())} trainer_checkpoint=null Trainer.device=cpu "
            f"--config_path={str(save_dir)}/config.yaml ", )


if __name__ == '__main__':
    main("./")

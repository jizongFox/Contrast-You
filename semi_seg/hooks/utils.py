import shutil
import typing as t
from functools import lru_cache
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
from torch import Tensor

from contrastyou.utils import switch_plt_backend
from contrastyou.utils.colors import label2colored_image
from contrastyou.writer import get_tb_writer
from semi_seg.epochers.helper import PartitionLabelGenerator, PatientLabelGenerator, ACDCCycleGenerator, \
    SIMCLRGenerator


@lru_cache()
def global_label_generator(dataset_name: str, contrast_on: str):
    logger.debug("initialize {} label generator for encoder training", contrast_on)
    if dataset_name == "acdc":
        if contrast_on == "partition":
            return PartitionLabelGenerator()
        elif contrast_on == "patient":
            return PatientLabelGenerator()
        elif contrast_on == "cycle":
            return ACDCCycleGenerator()
        elif contrast_on == "self":
            return SIMCLRGenerator()
        else:
            raise NotImplementedError(contrast_on)
    elif dataset_name in ("prostate", "prostate_md"):
        if contrast_on == "partition":
            return PartitionLabelGenerator()
        elif contrast_on == "patient":
            return PatientLabelGenerator()
        elif contrast_on == "self":
            return SIMCLRGenerator()
        else:
            raise NotImplementedError(contrast_on)
    elif dataset_name == "mmwhs":
        if contrast_on == "partition":
            return PartitionLabelGenerator()
        elif contrast_on == "patient":
            return PatientLabelGenerator()
        elif contrast_on == "self":
            return SIMCLRGenerator()
        else:
            raise NotImplementedError(contrast_on)
    elif dataset_name == "spleen":
        if contrast_on == "partition":
            return PartitionLabelGenerator()
        elif contrast_on == "patient":
            return PatientLabelGenerator()
        elif contrast_on == "self":
            return SIMCLRGenerator()
        else:
            raise NotImplementedError(contrast_on)
    elif dataset_name == "hippocampus":
        if contrast_on == "partition":
            return PartitionLabelGenerator()
        elif contrast_on == "patient":
            return PatientLabelGenerator()
        elif contrast_on == "self":
            return SIMCLRGenerator()
        else:
            raise NotImplementedError(contrast_on)
    else:
        NotImplementedError(dataset_name)


def get_label(contrast_on, data_name, partition_group, label_group):
    if data_name == "acdc":
        labels = global_label_generator(dataset_name="acdc", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=[p.split("_")[0] for p in label_group],
             experiment_list=[p.split("_")[1] for p in label_group])
    elif data_name in ("prostate", "prostate_md"):
        labels = global_label_generator(dataset_name="prostate", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=[p.split("_")[0] for p in label_group])
    elif data_name in ("mmwhsct", "mmwhsmr"):
        labels = global_label_generator(dataset_name="mmwhs", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=label_group)
    elif data_name == "prostate_md":
        labels = global_label_generator(dataset_name="prostate", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=label_group)
    elif data_name == "spleen":
        labels = global_label_generator(dataset_name="spleen", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=label_group)
    elif data_name == "hippocampus":
        labels = global_label_generator(dataset_name="hippocampus", contrast_on=contrast_on) \
            (partition_list=partition_group,
             patient_list=label_group)
    else:
        raise NotImplementedError(data_name)
    return labels


class FeatureMapSaver:

    def __init__(self, save_dir: t.Union[str, Path], folder_name="vis", use_tensorboard: bool = True) -> None:
        assert Path(save_dir).exists() and Path(save_dir).is_dir(), save_dir
        self.save_dir: Path = Path(save_dir)
        self.folder_name = folder_name
        (self.save_dir / self.folder_name).mkdir(exist_ok=True, parents=True)
        self.use_tensorboard = use_tensorboard

    @switch_plt_backend(env="agg")
    def save_map(self, *, image: Tensor, feature_map1: Tensor, feature_map2: Tensor, feature_type="feature",
                 cur_epoch: int,
                 cur_batch_num: int, save_name: str) -> None:
        """
        Args:
            image: image tensor with bchw dimension, where c should be 1.
            feature_map1: tensor with bchw dimension. It would transform to bhw with argmax on c dimension.
            feature_map2: tensor with bchw dimension. It would transform to bhw with argmax on c dimension.
            feature_type: image or feature. image is going to treat as image, feature would take the argmax on c.
            cur_epoch: current epoch
            cur_batch_num: cur_batch_num
            save_name: the png that would be saved under "save_name_cur_epoch_cur_batch_num.png" in to self.folder_name
                    folder.
        """
        assert feature_type in ("image", "feature")
        assert image.dim() == 4, f"image should have bchw dimensions, given {image.shape}."
        batch_size = feature_map1.shape[0]
        image = image.detach()[:, 0].float().cpu()
        assert feature_map1.dim() == 4, f"feature_map should have bchw dimensions, given {feature_map1.shape}."
        if feature_type == "image":
            feature_map1 = feature_map1.detach()[:, 0].float().cpu()
        else:
            feature_map1 = feature_map1.max(1)[1].cpu().float()
            feature_map1 = label2colored_image(feature_map1)
        assert feature_map2.dim() == 4, f"feature_map should have bchw dimensions, given {feature_map2.shape}."
        if feature_type == "image":
            feature_map2 = feature_map2.detach()[:, 0].float().cpu()
        else:
            feature_map2 = feature_map2.max(1)[1].cpu().float()
            feature_map2 = label2colored_image(feature_map2)

        for i, (img, f_map1, f_map2) in enumerate(zip(image, feature_map1, feature_map2)):
            save_path = self.save_dir / self.folder_name / f"{save_name}_{cur_epoch:03d}_{cur_batch_num:02d}_{i:03d}.png"
            fig = plt.figure(figsize=(1.5, 4.5))
            plt.subplot(311)
            plt.imshow(img, cmap="gray")
            plt.axis('off')
            plt.subplot(312)
            plt.imshow(f_map1, cmap="gray" if feature_type == "image" else None)
            plt.axis('off')
            plt.subplot(313)
            plt.imshow(f_map2, cmap="gray" if feature_type == "image" else None)
            plt.axis('off')
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_figure(
                    tag=f"{self.folder_name}/{save_name}_{cur_batch_num * batch_size + i:02d}",
                    figure=plt.gcf(), global_step=cur_epoch, close=True
                )
            plt.close(fig)

    def zip(self) -> None:
        """
        Put all image folders as a zip file, in order to avoid IO things when downloading.
        """
        try:
            shutil.make_archive(str(self.save_dir / self.folder_name.replace("/", "_")), 'zip',
                                str(self.save_dir / self.folder_name))
            shutil.rmtree(str(self.save_dir / self.folder_name))
        except (FileNotFoundError, OSError, IOError) as e:
            logger.opt(exception=True, depth=1).warning(e)

    @property
    @lru_cache()
    def tb_writer(self):
        try:
            writer = get_tb_writer()
        except RuntimeError:
            writer = None
        return writer


class DistributionTracker:

    def __init__(self, save_dir: str, folder_name="dist_track", use_tensorboard=True) -> None:
        super().__init__()

        assert Path(save_dir).exists() and Path(save_dir).is_dir(), save_dir
        self.save_dir: Path = Path(save_dir)
        self.folder_name = folder_name
        (self.save_dir / self.folder_name).mkdir(exist_ok=True, parents=True)
        self.use_tensorboard = use_tensorboard

    @switch_plt_backend(env="agg")
    def save_map(self, *, dist1: Tensor, dist2: Tensor, cur_epoch) -> None:
        """
        Args:
            dist1:Tensor, the semgentaiton distribution after softmax
            dist2:Tensor, the semgentaiton distribution after softmax

        """
        assert dist1.dim() == 4 and dist2.dim() == 4
        assert dist1.shape == dist2.shape

        def get_features(dist: Tensor):
            marginal_dist = dist.mean(dim=[0, 2, 3])
            confidents = dist.max(1)[0].ravel()[:1000]
            return marginal_dist.detach().cpu(), confidents.detach().cpu()

        fig = plt.figure()
        margin, confident = get_features(dist1)
        plt.subplot(221)
        plt.plot(margin)
        plt.subplot(222)
        plt.hist(confident, bins=20, histtype='step')
        margin, confident = get_features(dist2)
        plt.subplot(223)
        plt.plot(margin)
        plt.subplot(224)
        plt.hist(confident, bins=20, histtype='step')

        save_path = self.save_dir / self.folder_name / f"distribution_epoch_{cur_epoch:03d}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_figure(
                tag=f"{self.folder_name}/distribution",
                figure=plt.gcf(), global_step=cur_epoch, close=True
            )
        plt.close(fig)

    def zip(self) -> None:
        """
        Put all image folders as a zip file, in order to avoid IO things when downloading.
        """
        try:
            shutil.make_archive(str(self.save_dir / self.folder_name.replace("/", "_")), 'zip',
                                str(self.save_dir / self.folder_name))
            shutil.rmtree(str(self.save_dir / self.folder_name))
        except (FileNotFoundError, OSError, IOError) as e:
            logger.opt(exception=True, depth=1).warning(e)

    @property
    @lru_cache()
    def tb_writer(self):
        try:
            writer = get_tb_writer()
        except RuntimeError:
            writer = None
        return writer

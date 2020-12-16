from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

feature_name = "Conv5_Up_conv3_Up_conv2"


def get_mean_std(*dataseries: pd.Series):
    mean_serie = sum(dataseries) / len(dataseries)
    diff = sum([(d - mean_serie) ** 2 for d in dataseries]) / (len(dataseries))
    diff = diff.apply(lambda x: np.sqrt(x))

    return mean_serie, diff


def plot_mean_std(ax, mean_series: pd.Series, std_series: pd.Series, label=None, color="r"):
    x = np.array(list(range(len(mean_series))))
    ax.plot(x, mean_series.to_numpy(), color=color, label=label)
    ax.fill_between(x,
                    mean_series.to_numpy() + 0.8 * std_series.to_numpy(),
                    mean_series.to_numpy() - 0.8 * std_series.to_numpy(),
                    color=color, alpha=0.2, interpolate=True)


def load_val_scores(folder_path: str):
    f = Path(folder_path)
    assert f.exists(), f
    csvs = list(f.rglob("storage.csv"))
    assert len(csvs) == 1, f
    result_dataframe = pd.read_csv(csvs[0])
    DSC1, DSC2, DSC3, meanDSC = result_dataframe["val_dice_DSC1"], \
                                result_dataframe["val_dice_DSC2"], \
                                result_dataframe["val_dice_DSC3"], \
                                result_dataframe["val_dice_DSC_mean"]

    return DSC1, DSC2, DSC3, meanDSC


if __name__ == '__main__':
    # normal_path
    iicpaths = [f"random_seed_{r}/{feature_name}/normal/iic/0.1" for r in [8, 9, 10]]
    udaiicpaths = [f"random_seed_{r}/{feature_name}/normal/udaiic/5_0.1" for r in [8, 9, 10]]
    ps_path = [f"random_seed_{r}/{feature_name}/ps" for r in [8, 9, 10]]
    fs_path = [f"random_seed_{r}/{feature_name}/fs" for r in [8, 9, 10]]

    # pretrain_path
    iicpretrainpaths = [f"random_seed_{r}/{feature_name}/pretrain/iic/train" for r in [8, 10]]
    udaiicpretrainpaths = [f"random_seed_{r}/{feature_name}/pretrain/udaiic/10_0.1/train" for r in [8, 9, 10]]

    fig, ax = plt.subplots(1, figsize=(6, 3.9))

    mean, std = get_mean_std(*[load_val_scores(p)[-1] for p in ps_path])
    plot_mean_std(ax, mean, std, label="Partial-Supervision", color="r")

    mean, std = get_mean_std(*[load_val_scores(p)[-1] for p in fs_path])
    plot_mean_std(ax, mean, std, label="Fully-Supervision", color="c")

    mean, std = get_mean_std(*[load_val_scores(p)[-1] for p in udaiicpaths])
    plot_mean_std(ax, mean, std, label="Joint-Optimization (Ours)", color="b", )

    mean, std = get_mean_std(*[load_val_scores(p)[-1] for p in udaiicpretrainpaths])
    plot_mean_std(ax, mean, std, label="Pretrain-Finetune", color="g", )
    plt.legend()
    plt.ylim([0.4, 0.94])
    plt.xlim([3, 99])
    plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    plt.grid(which="both")
    plt.xticks(list(range(0, 100, 10)), rotation=0)
    plt.xlabel("Training Epochs")
    plt.ylabel("3D mean DSC on Validation Set")
    leg = plt.gca().get_legend()

    ltext = leg.get_texts()

    plt.setp(ltext, fontsize='small')
    plt.savefig("pretrain-finetune.pdf", bbox_inches="tight", format="pdf", dpi=500)
    plt.show()

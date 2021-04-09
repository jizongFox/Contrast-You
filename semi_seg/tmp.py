from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def plot_cycle(folder):
    label_fun = lambda name: str(name).split("_")[-1]
    root = Path(folder)
    feature_list = sorted(root.glob("*"))[:10]
    feature1 = torch.cat([torch.load(x).detach() for x in feature_list if label_fun(x) == "00"], dim=0)
    feature2 = torch.cat([torch.load(x).detach() for x in feature_list if label_fun(x) == "01"], dim=0)
    group_num = [torch.load(x).shape[0] for x in feature_list if label_fun(x) == "00"]
    features = torch.cat([feature1, feature2], dim=0)
    cor = features.mm(features.t())

    plt.figure()
    plt.imshow(cor, cmap="gray", )
    plt.colorbar()

    plt.figure()
    plt.imshow(feature1.mm(feature2.t()), cmap="gray", )
    plt.colorbar()
    # plt.vlines()

    plt.figure()
    pd.Series(cor.view(-1)).plot.hist(bins=50)
    return feature1, feature2


def plot_patient(folder):
    sorted_by_patient = lambda name_list: sorted(name_list, key=lambda n: str(n.stem).split("_")[0])
    root = Path(folder)
    feature_list = sorted(root.glob("*"))[:10]
    features = torch.cat([torch.load(x).detach() for x in sorted_by_patient(feature_list)], dim=0)
    cor = features.mm(features.t())

    plt.figure()
    plt.imshow(cor, cmap="gray", )
    plt.colorbar()
    plt.figure()
    pd.Series(cor.view(-1)).plot.hist(bins=50)


# plt.ioff()
f1, f2 = plot_cycle(
    "runs/0405_monitor/monitor/githash_ab6c96d/acdc/random_seed_10/sample_num_10/global_Conv5_1.0/contrast_on_cycle/infonce/projections/99")
# f1, f2 = plot_cycle(
#     "runs/0328/githash_3f7e6f2/acdc/random_seed_10/sample_num_10/"
#     "global_Conv5_1.0/contrast_on_partition/self-paced/method_hard/loss_params*4.0_6.0/features/119")
# # plot_cycle("cycle/self-paced")
# plt.figure()
# pd.Series(torch.cat([f1.mm(f1.t()).mean(1), f2.mm(f2.t()).mean(1), f1.mm(f2.t()).mean(1)])).plot()
plt.show()

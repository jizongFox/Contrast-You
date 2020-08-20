# adding postprocessing module to produce images.
import matplotlib.pyplot as plt
import matplotlib as mpl

labeled_ratios = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 1.0]

mts = [0.7946, 0.810737669, 0.825149854, 0.845955094, 0.857171555, 0.863374869, 0.875850737, 0.885325611, 0.8950]
uda_iic = [0.5319, 0.756039302, 0.800508598, 0.859897832, 0.862838348, 0.874597828, 0.879657408, 0.890836279, 0.8950]

ps = [0.34, 0.516110112, 0.571018706, 0.743557851, 0.820190847, 0.85607487, 0.864632746, 0.883666337, 0.8950]

iicmeanteacher_labeled_ratios = [0.02, 0.03, 0.05, 0.06, ]  # 0.07, 0.08, 0.1]
iicmeanteacher = [0.8086502, 0.817608992, 0.847206803, 0.862841765, ]  # 0.87117211, 0.874097332, 0.878502429]

linewith = 1.8

plt.figure(figsize=(5.2, 3.5))
plt.hlines(0.8950, -1, 2, linestyles="dashdot", colors="red", label="Full Supervision")
plt.plot(labeled_ratios, ps, label="Partial Supervision", marker="x", markersize=8, linewidth=linewith)
plt.plot(labeled_ratios, mts, label="Mean Teacher", marker=".", markersize=8, linewidth=linewith)

plt.vlines(0.05, -1, 1, linestyles=":")
plt.plot(labeled_ratios, uda_iic, label="Ours", marker="*", markersize=8, linewidth=linewith)
plt.plot(iicmeanteacher_labeled_ratios, iicmeanteacher, marker="^", markersize=6.5, label="Ours (Mean Teacher)",
         color="lightgreen", linewidth=linewith)
plt.xscale("log")
plt.xticks([0.02, 0.03, 0.05, 0.07, 0.1, 0.2], rotation=0)
plt.gca().get_xaxis().set_major_formatter(mpl.ticker.PercentFormatter(1.0))
plt.xlim([0.019, 0.5])
plt.ylim([0.7, 0.92])
plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
plt.legend(loc="lower right")
# plt.title("3D mean dice for ACDC dataset with different labeled data ratio")
plt.grid(which="both")
plt.xlabel("Labeled Ratio")
plt.ylabel("3D mean DSC on Validation Set")
# plt.show()
plt.savefig("different_label_ratio.pdf", bbox_inches="tight", format="pdf", dpi=500)
plt.show()

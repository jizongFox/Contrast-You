import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepclustering2.schedulers.customized_scheduler import ExpScheduler, InverseExpScheduler, LinearScheduler

scheduler1 = ExpScheduler(max_epoch=120, begin_value=0, end_value=1)
scheduler2 = InverseExpScheduler(max_epoch=120, begin_value=0, end_value=1)
scheduler3 = LinearScheduler(max_epoch=120, begin_value=0, end_value=1)

lr1 = []
lr2 = []
lr3 = []
x = list(range(120))
plt.figure(figsize=(9.5, 3.7))
plt.subplot(121)
for i in x:
    lr1.append(scheduler1.value)
    lr2.append(scheduler2.value)
    lr3.append(scheduler3.value)

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
plt.plot(x, lr1, label="Square")
plt.plot(x, lr2, label="Square root")
plt.plot(x, lr3, label="Linear")
plt.legend()

plt.xlabel("Epochs")

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ["$\gamma_{low}$", "", "", "", "", "$\gamma_{high}$"])
ax = plt.gca()
plt.grid(which="both")

ax.yaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor')


def read_true_w_percentage(csv_path):
    csv_file = pd.read_csv(csv_path)
    return csv_file["pretrain_real_percentage0_mean"][1:]


demo_path = "../../semi_seg/runs/0411/demo/soften_case2"
linear_folder = os.path.join(demo_path, "type_linear", "pre", "storage.csv")
squre_folder = os.path.join(demo_path, "type_square", "pre", "storage.csv")
inverse_squre_folder = os.path.join(demo_path, "type_inversesquare", "pre", "storage.csv")

x = np.linspace(1, 120, num=119, endpoint=True)
true_linear = read_true_w_percentage(linear_folder)
true_square = read_true_w_percentage(squre_folder)
true_inverse_square = read_true_w_percentage(inverse_squre_folder)
plt.subplot(122)

plt.plot(x, true_square, label="Square")
plt.plot(x, true_inverse_square, label="Square root")
plt.plot(x, true_linear, label="Linear")
plt.xlabel("Epochs")
ax = plt.gca()
ax.set_xscale('log')
plt.xticks([1, 2, 3, 5, 10, 20, 40, 110], [1, 2, 3, 5, 10, 20, 40, 110])
plt.ylabel("$\mathrm{E}(w_{i,j})$")
plt.grid(which="both")
plt.xlim([0.95, 115])
plt.ylim([0.01, 1.01])
plt.scatter(2, true_inverse_square[2], color=u'#ff7f0e', marker="*", linewidths=2)
plt.scatter(4, true_inverse_square[4], color=u'#ff7f0e', marker="*", linewidths=2)
plt.scatter(7, true_inverse_square[7], color=u'#ff7f0e', marker="*", linewidths=2)
plt.scatter(7, true_inverse_square[7], color=u'#ff7f0e', marker="*", linewidths=2)
plt.scatter(37, true_inverse_square[37], color=u'#ff7f0e', marker="*", linewidths=2)
plt.tight_layout()
# plt.legend()
plt.show()

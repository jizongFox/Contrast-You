import matplotlib.pyplot as plt
from deepclustering2.schedulers.customized_scheduler import ExpScheduler, InverseExpScheduler, LinearScheduler

scheduler1 = ExpScheduler(max_epoch=120, begin_value=0, end_value=1)
scheduler2 = InverseExpScheduler(max_epoch=120, begin_value=0, end_value=1)
scheduler3 = LinearScheduler(max_epoch=120, begin_value=0, end_value=1)

lr1 = []
lr2 = []
lr3 = []
x = list(range(120))
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
plt.xlabel("Epochs")
plt.ylabel("$\gamma$")
plt.legend()
plt.grid()
plt.show()

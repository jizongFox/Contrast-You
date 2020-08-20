# adding postprocessing module to produce images.
labeled_ratios = [0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 1.0]

mts = [0.810737669, 0.825149854, 0.845955094, 0.857171555, 0.863374869, 0.875850737, 0.885325611, 0.8950]
uda_iic = [0.756039302, 0.800508598, 0.859897832, 0.862838348, 0.874597828, 0.879657408, 0.890836279, 0.8950]
iic = [0.49071601, 0.645830929, 0.823460301, 0.855877717, 0.857293268, 0.875480652, 0.882863343, 0.8950]

iicmeanteacher_labeled_ratios = [0.02, ]

import matplotlib.pyplot as plt

plt.plot(labeled_ratios, mts)
plt.plot(labeled_ratios, uda_iic)
plt.hlines(0.8950, -1, 2, linestyles="dashdot")
plt.xscale("log")
plt.xlim([0, 1.001])
plt.show()

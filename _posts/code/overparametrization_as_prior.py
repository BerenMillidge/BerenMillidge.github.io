import numpy as np
import matplotlib.pyplot as plt

N = 200
ps = np.arange(200,5000,10)
print(ps)
ratios = [((p - N)/p) for p in ps]
print(ratios)
plt.plot(ps, ratios)
plt.ylabel("Volume Ratio",fontsize=17)
plt.xlabel("Number of Parameters",fontsize=17)
plt.title("Volume Ratio by Number of Parameters",fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tight_layout()
plt.savefig("volume_ratio_graph.jpg")
plt.show()
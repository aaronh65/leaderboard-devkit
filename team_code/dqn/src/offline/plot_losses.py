import matplotlib.pyplot as plt
import numpy as np
import sys

path = sys.argv[1]
with open(f'{path}/train_losses.npy', 'rb') as f:
    l = np.load(f)

exp_l = np.exp(l)
exp_l /= sum(exp_l)
exp_l.sort()

plt.plot(np.arange(len(exp_l)), exp_l)
plt.show()

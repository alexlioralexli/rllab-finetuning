import matplotlib.pyplot as plt

import numpy as np

# arr = np.load("antvelocity.npy")
# arr = np.load("snakevelocities.npy")
arr = np.load("antangles.npy")

plt.hist(arr, bins=15)
# plt.savefig("snakevelocity.png")
plt.hist(arr.flatten(), bins=15)
plt.savefig("antangles.png")
# ant: ;penalty(arr, 1.2, 0.0025, 0.001)
# snake: penalty(arr, 0.111, 0.0005, 0.001)
# ant angle: penalty(arr.flatten(), 0.96, 0.0002, 0.00015)
# snake angle: penalty(a, 2.1, 0.0001, 0.00005)
def penalty(array, threshold, coeff, bias):
    return np.sum((array[array > threshold] - threshold)*coeff + bias) / 15.0

import IPython
IPython.embed()

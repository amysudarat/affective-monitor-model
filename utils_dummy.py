# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_sample(sample):
    data = sample['data']
    label = sample['label']
    plt.plot(data)
    plt.title(label)
    plt.show()
import matplotlib.pyplot as plt

import numpy as np


def lim(x, pad=16):
    xmin, xmax = min(x), max(x)
    margin = (xmax - xmin) * pad / 100
    return xmin - margin, xmax + margin


X_axis = [2, 3, 4, 5]
num_of_conversations = ['219', '44', '15', '8']

acc_wocrf = [0.65, 0.66, 0.75, 0.60]
acc_wcrf = [0.68, 0.73, 0.76, 0.68]

x = np.arange(2, 2 + len(X_axis))  # the label locations
width = 0.3  # the width of the bars

fig, ax1 = plt.subplots()
rects1 = ax1.bar(x - width / 2, acc_wocrf, width, label='wo CRF')
rects2 = ax1.bar(x + width / 2, acc_wcrf, width, label='w CRF')

ax1.legend()
ax1.set_ylabel('Accuracy')
ax1.set_yticks((0.2, 0.4, 0.6, 0.8, 1))

ax1.set_xlabel("Number of customer turns")
ax1.set_xticks((2, 3, 4, 5), labels=['2', '3', '4', '5≥'])
# ax1.tick_params(axis='x', pad=10)
ax2 = ax1.twiny()
ax1.callbacks.connect('xlim_changed', lambda ax1: ax2.set_xlim(*lim(X_axis)))
ax1.set_xlim(*lim(X_axis))  # trigger the xlim_changed callback

ax2.set_xticks((2, 3, 4, 5), labels=['219', '44', '15', '8'])
ax2.set_xlabel("Number of conversations")

ax1.bar_label(rects1, padding=3)
ax1.bar_label(rects2, padding=3)

# ax2.tick_params(axis='y', pad=10)
# 5. Make the plot visible
# plt.show()
plt.savefig('wcrf_wocrf.png', bbox_inches='tight')
plt.savefig('wcrf_wocrf.pdf', bbox_inches='tight')

from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data, notch_filter
import torch._tensor as tt
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, lfilter
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

N = 200
# sample spacing
# T = 1.0 / 1024.0
x = np.linspace(0.0,  N, N)
# y = np.full((len(x)),1.) #
y_sin_5 = np.sin(5.0 * 2.0 * np.pi * x / N)
y_sin_20 = np.sin(20.0 * 2.0 * np.pi * x / N)
y = y_sin_5 + y_sin_20
# y = x ** 3
# y[100] = 1.

# y_0 = notch_filter(y, 256, 50, n_jobs=-1, verbose='ERROR')
# y_0 = filter_data(y, 256, 79, 81, n_jobs=-1, verbose='ERROR')


y_fft = fft(y)
# yf_0 = fft(y_0)
# xf = fftfreq(N, T)[:N//2]

y_t = tt.Tensor(y)

#############
f = 20.0 / N
BW = 2.0 / N
R = 1 - 3 * BW
K = (1 - 2 * R * np.cos (2 * np.pi  * f) + R ** 2) / (2 - 2 * np.cos(2 * np.pi * f))


#band pass
# a_0 = 1-K
# a_1 = 2 * (K - R) * np.cos(2 * np.pi * f)
# a_2 = R ** 2 - K
# b_1 = 2 * R * np.cos(2 * np.pi * f)
# b_2 = - R ** 2


#band reject
a_0 = K
a_1 = -2 * K * np.cos(2 * np.pi * f)
a_2 = K
b_0 = 0.,
b_1 = 2 * R * np.cos(2 * np.pi * f)
b_2 = - R ** 2

#############

# k = [
#     0., 0., 0., 0., 0., 0., 0., 0., 0., #0., #0.,
#     # 0.,  1., 1., 1., 1., 1., 1., 1., 1., #0., #0.,
#     0.,
#     0., 0., 0., 0., 0., 0., 0., 0., 1., #0., #0.,
#     # 1., 1., 1., 1., 1., 1., 1., 1., 1., #0., #0.,
# ]
k_a = [a_0, a_1, a_2]
# k = [
#
#
#
# ]
kernel_a = tt.Tensor(k_a)
k_b = [0.0, b_1, b_2]
kernel_b = tt.Tensor(k_b)
# print(f'x_t={y_t}')

# y_t_1 = F.conv1d(y_t.view(1, -1), kernel_a.view(1, 1, -1))
y_t_1 = np.convolve(y, k_a)
# y_t_2 = F.conv1d(y_t.view(1, -1), kernel_b.view(1, 1, -1))
y_t_2 = np.convolve(y, k_b)

y_t_0 =  y_t_1  +y_t_2 #

fig1, ax = plt.subplots(figsize=(10, 3))
# ax.plot(x[:y_t_0.size()[1]], y_t_0.view(-1).numpy(), 'g')
# ax.plot(x, y_t_0[:len(x)], 'g')
ax.plot(x, butter_bandpass_filter(y, 1, 10, 200), 'g')
ax.plot(x, y, 'r')
ax.plot(x, y_sin_5, 'b')
plt.show()

print(f'len(x)={len(x)}')
# ax.plot(x, np.pad(y_t_1.view(-1).numpy(), int(len(k)/2), 'symmetric'), 'g')
# plt.plot(2.0/N * np.abs(y_fft[0:N//2]), 'g')
# ax.plot(xf, 2.0/N * np.abs(yf_0[0:N//2]), 'g')

plt.show()
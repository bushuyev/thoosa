import torch
from torch import fft
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import stumpy

# Number of sample points

def plot_y(x, y, n=100):

    fig1, ax = plt.subplots(figsize=(50, 10))
    ax.plot(x[:n], y[:n])
    plt.show()

    # yf = fft.fft(Tensor(y))
    # xf = fft.fftfreq(N, T)[:N // 2]
    #
    # plt.plot(xf, 2.0 / N * yf[0:N // 2].abs())
    # plt.grid()
    # plt.show()


def pulse(N, T, m = 1, s = 0.025, a=1, f=80, x_scal=1):
    return np.array([a * np.sin(f * np.pi * x * x_scal ) * np.exp(-(x * x_scal-m)**2/(2*s ** 2)) / s * np.sqrt(2 * np.pi) for x in np.linspace(0.0, N * T, N)])

def gaussian(N, T, m = 0, s = 0.025, x_scal=1):
    return np.array([ np.exp(-(x * x_scal-m)**2/(2*s ** 2)) / s * np.sqrt(2 * np.pi) for x in np.linspace(0.0, N * T, N)])

def add_at(to, sub, at):
    to = to.copy()
    for i, x in enumerate(sub):
        to[i + at] += x

    return to

N = 2048 #200 * 10 # 200 samples per sec, 10 sec
# sample spacing
T = 1.0 / 200.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y_1 = np.sin(2.0*np.pi*x) #+ 0.5*np.sin(80.0 * 2.0*np.pi*x)
y_5 = np.sin(5.0 * 2.0*np.pi*x) #+ 0.5*np.sin(80.0 * 2.0*np.pi*x)
y_50 = np.sin(50.0 * 2.0*np.pi*x) #+ 0.5*np.sin(80.0 * 2.0*np.pi*x)
y_0 = np.zeros(N)

y_5_50 = 30 * y_5 + y_50


# pulse = np.exp(-1 *  np.linspace(0.0, N*T, N) ** 2),
pulse_100 = pulse(N=1000, T=T, s=0.2, f=5, x_scal=4)

# pulse = np.pad(pulse_1, (0, N - len(pulse_100)),  'constant')
pulse_1 = add_at(y_1, pulse_100, 200)
pulse_2 = add_at(pulse_1, pulse_100, 600)
pulse_3 = add_at(pulse_2, pulse_100, 800)


# guassian_1 = gaussian(N=1000, T=T, s=2, m=10, x_scal=4)
# guassian_1 = add_at(guassian_1, guassian_1[200:300], 800)
# guassian_1 = add_at(guassian_1, guassian_1[700:800], 100)

plot_y(x, pulse_3, n=N)


np.savetxt("/tmp/foo.csv", pulse_3, delimiter=",")

#
#
# # pulse_3 = (pulse_3 - pulse_3.mean())/pulse_3.std()
#
# pulse_2_ff = fft.fft(Tensor(pulse_2))
# xf = fft.fftfreq(N, T)[:N // 2]
# plot_y(xf, 2.0 / N * pulse_2_ff[0:N // 2].abs(), n=2000)
#
# # pattern = pulse_100
# pattern = pulse_2 #[240:260]
# # pattern = np.flip(pattern)
# # plot_y(x[200:300], pattern)
# # pulse_100_corrs = fft.ifft(pulse_3_ff * fft.fft(Tensor(np.pad(pattern, (0, N - len(pattern)), 'constant'))))
# pulse_100_corrs = fft.ifft(
#     pulse_2_ff
#     *
#     fft.fft(
#         Tensor(
#             np.pad(pattern, (0, N - len(pattern)), 'constant')
#             # add_at(y_0, pattern, int((N-len(pattern))/2))
#         )
#     )
# )
#
# plot_y(x, pulse_100_corrs.abs(), n=2000)
#
#
#
# m = 400
# mp = stumpy.stump(pulse_2, m)
#
# plt.plot(mp[:, 0][:500])
# plt.show()


# fig1, ax = plt.subplots(figsize=(50, 10))
# ax.plot(x, pulse_100_corrs.abs(), 'g')
# ax.plot(x, pattern, 'r')
# plt.show()
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from mne.filter import filter_data, notch_filter

N = 128
# sample spacing
T = 1.0 / 256.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + np.sin(80.0 * 2.0*np.pi*x)


y_0 = notch_filter(y, 256, 50, n_jobs=-1, verbose='ERROR')
# y_0 = filter_data(y, 256, 79, 81, n_jobs=-1, verbose='ERROR')


yf = fft(y)
yf_0 = fft(y_0)
xf = fftfreq(N, T)[:N//2]



fig1, ax = plt.subplots(figsize=(10, 3))
ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]), 'r')
ax.plot(xf, 2.0/N * np.abs(yf_0[0:N//2]), 'g')

plt.show()
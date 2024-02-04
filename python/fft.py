# import numpy as np
# import matplotlib.pyplot as plt
# from  scipy.fft import  fft
# from mne.filter import filter_data, notch_filter
#
# #
# # # Number of samplepoints
# T = 1000
# N = 100_000
# F = 20
# # sample spacing
# # T = 1.0 / 800.0
# x = np.linspace(0.0, T, N)
# y = np.sin(10 * 2 * np.pi / T * x) + np.sin(20 * 2 * np.pi / T * x) + np.sin(30 * 2 * np.pi / T * x) #+ 0.5*np.sin(80.0 * 2.0*np.pi*x)
# fig, ax = plt.subplots()
# ax.plot(x, y)
# plt.show()
#
#
# fig, ax = plt.subplots()
# yf = fft(y)
# ax.plot(2.0 / N * abs(yf[:100]), 'g')
#
# # yf = fft(notch_filter(y, 100, 20, n_jobs=-1, verbose='ERROR'))
# # yf = fft(filter_data(y, 100, 9, 11, n_jobs=-1, verbose='ERROR'))
# # ax.plot(2.0 / N * yf[:100], 'r')
#
# plt.show()
import os

from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import polars as pl

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

os.environ["POLARS_FMT_MAX_ROWS"] = "100"
os.environ["POLARS_FMT_MAX_COLS"] = "100"


df_train = pl.read_csv("../data/train.csv", try_parse_dates=True)
eegs_df = df_train.select(pl.col('eeg_id').unique())
eeg_df = pl.read_parquet(f"../data/train_eegs/{eegs_df['eeg_id'][300]}.parquet")
# print(eeg_df.columns)
# ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']


N = eeg_df.shape[0]
# sample spacing
T = 1.0 / 200.0
x = np.linspace(0.0, N*T, N, endpoint=False)
# y = np.sin(50.0 * 2.0*np.pi*x) #+ np.sin(80.0 * 2.0*np.pi*x)
y = eeg_df['F3'].to_numpy()

# y = butter_bandpass_filter(y, 0.5, 40, 200, order=6)


fig1, ax = plt.subplots(figsize=(100, 10))
plt.plot(x, y)
plt.show()
# plt.savefig('../plots/eeg_fp1_.png')


yf = fft(y)
# print(yf)
xf = fftfreq(N, T)[:N//2]

fig1, ax = plt.subplots(figsize=(50, 10))

ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# # ax.hist(np.array(yf, dtype=np.float64))
ax.grid()
plt.show()
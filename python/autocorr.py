import polars as pl
import matplotlib.pyplot as plt

df = pl.read_parquet('../data/train_eegs/582999.parquet')
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
import scipy.fftpack


print(df.columns)

# plt.plot(df['Fp1'], 'r')

fig1, ax = plt.subplots(figsize=(100, 3))
# ax.plot(  df['F3'], 'b',  linewidth=0.5) #[x / 200 for x in range(0, len(df['F3']))],
# ax.set_xscale(100)

# plot_acf(df['F3'])
# plot_pacf(df['F3'], ax=ax, lags=10)


yf = scipy.fftpack.fft(df['P3'].to_numpy())
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

fig, ax = plt.subplots()
ax.plot(2.0 / df.shape[0] * yf[:200])
plt.show()

# pyplot.show()

plt.show()
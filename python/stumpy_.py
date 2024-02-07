import polars as pl
import matplotlib.pyplot as plt
import stumpy
import numpy as np
from matplotlib.patches import Rectangle


# df = pl.read_parquet('../data/train_eegs/582999.parquet')
# values = df['Fp1'].to_numpy().astype(np.float64)
# df = pl.read_csv('/tmp/foo.csv')
df = pl.read_csv('../data_local/STUMPY_Basics_steamgen.csv')
# df.write_csv('/tmp/foo_eeq.csv')

print(df.columns)


# values = df[df.columns[0]].to_numpy().astype(np.float64)
values = df['steam flow'].to_numpy().astype(np.float64)

m = 640
mp = stumpy.stump(values, m)


motif_idx = np.argsort(mp[:, 0])[0]

print(f"The motif is located at index {motif_idx}")

nearest_neighbor_idx = mp[motif_idx, 1]

print(f"The nearest neighbor is located at index {nearest_neighbor_idx}")


fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
plt.suptitle('Motif (Pattern) Discovery', fontsize='30')

axs[0].plot(values)
axs[0].set_ylabel('Steam Flow', fontsize='20')
rect = Rectangle((motif_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].set_xlabel('Time', fontsize ='20')
axs[1].set_ylabel('Matrix Profile', fontsize='20')
axs[1].axvline(x=motif_idx, linestyle="dashed")
axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
axs[1].plot(mp[:, 0])
plt.show()


fig1, ax = plt.subplots(figsize=(50, 10))
ax.plot(mp[:, 0])
plt.show()


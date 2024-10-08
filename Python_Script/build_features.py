import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../Data/interim-Data/02-outlier-removed.pkl')

predictor_columns  = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()
    
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

for s in df['set'].unique():
    
    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]
    
    duration = stop - start
    df.loc[(df['set'] == s), 'duration'] = duration.seconds
    
    
duration_df = df.groupby(['category'])['duration'].mean()

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

lowpass = LowPassFilter()

fs = 1000/ 200

cutoff = 1.3

df_lowpass = lowpass.low_pass_filter(df_lowpass, 'acc_y', fs, cutoff_frequency=cutoff)

for col in predictor_columns:
    df_lowpass = lowpass.low_pass_filter(df_lowpass, col, fs, cutoff_frequency=cutoff)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    df_lowpass.drop(columns=col + "_lowpass", axis=1, inplace=True)

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()

PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)


## Visualizing 

subset = df_pca[df_pca['set'] == 20]

subset[['pca_1', 'pca_2', 'pca_3']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_square = df_pca.copy()

acc_r = df_square['acc_x'] ** 2 + df_square['acc_y'] ** 2 + df_square['acc_z'] ** 2 
gyro_r = df_square['gyro_x'] ** 2 + df_square['gyro_y'] ** 2 + df_square['gyro_z'] ** 2 

df_square['acc_r'] = np.sqrt(acc_r)
df_square['gyro_r'] = np.sqrt(gyro_r)

subset = df_square[df_square['set'] == 40]

subset[['acc_r', 'gyro_r']].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_square.copy()

NumAbs = NumericalAbstraction()

predictor_columns += ['acc_r', 'gyro_r']

 


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
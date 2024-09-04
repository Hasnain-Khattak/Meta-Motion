import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../Data/interim-Data/01_data_processed.pkl')

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['figure.dpi'] = 100


plt.boxplot(df['acc_x'])
plt.xlabel('Acc_x')
plt.show


df[['acc_x', 'label']].boxplot(by='label')
df[['acc_y', 'label']].boxplot(by='label')
df[['set', 'label']].boxplot(by='label')

outlier_cols  = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

df[outlier_cols[:3]+ ['label']].boxplot(by='label', layout=(1, 3))
df[outlier_cols[3:]+ ['label']].boxplot(by='label', layout=(1, 3))




# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function


# Plot a single column


# Loop over all columns


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution


# Insert Chauvenet's function


# Loop over all columns


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function


# Loop over all columns


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column


# Create a loop

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv('../Raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

single_file_gyro = pd.read_csv('../Raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv')

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob('../Raw/MetaMotion/*.csv')


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

data_path = '../Raw/MetaMotion\\'

f = files[0]
### We are fetching three things from file path
# 1. Participant (A, B, C, D, E)
participant = f.split('-')[0]
participant= participant.replace(data_path, "")

# 2. Label 
label = f.split('-')[1]

# 3. Category
category = f.split('-')[2].rstrip('1234')

## Creating a DataFrame
df = pd.read_csv(f)
df['Participant']  = participant
df['label'] = label
df['category'] = category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

### Creating an Empty DataFrame

## acc stands for Accelerometer DataFrame
acc_df = pd.DataFrame()

###gyro stands for gyroscope DataFrame
gyro_df = pd.DataFrame()

## Creating a Counter
acc_set = 1
gyro_set = 1

for file in files:
    participant = file.split('-')[0]
    participant= participant.replace(data_path, "")
    label = file.split('-')[1]
    category = file.split('-')[2].rstrip('123').rstrip('2_MetaWear_2019')
    df = pd.read_csv(file)
    df['Participant']  = participant
    df['label'] = label
    df['category'] = category 
    if 'Accelerometer' in file:
        df['set'] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
    if 'Gyroscope' in file:
        df['set'] = gyro_set
        gyro_set += 1
        gyro_df = pd.concat([gyro_df, df])

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.index  = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')

gyro_df.index  = pd.to_datetime(gyro_df['epoch (ms)'], unit='ms')

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
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

del acc_df['Unnamed: 0']
del acc_df['elapsed (s)']
del acc_df['epoch (ms)']
del acc_df['time (01:00)']

del gyro_df['Unnamed: 0']
del gyro_df['elapsed (s)']
del gyro_df['epoch (ms)']
del gyro_df['time (01:00)']

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


### The Below function is the combination of all step above we did

files = glob('../Raw/MetaMotion/*.csv')


data_path = '../Raw/MetaMotion\\'

def read_data(files):
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

    # del acc_df['Unnamed: 0']
    del acc_df['elapsed (s)']
    del acc_df['epoch (ms)']
    del acc_df['time (01:00)']

    # del gyro_df['Unnamed: 0']
    del gyro_df['elapsed (s)']
    del gyro_df['epoch (ms)']
    del gyro_df['time (01:00)']
    
    return acc_df, gyro_df
    

acc_df, gyro_df = read_data(files)
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

### Merging the DataFrames
data_merged  = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)

data_merged.columns = [
    'acc_x',
    'acc_y',
    'acc_z',
    'gyro_x',
    'gyro_y',
    'gyro_z',
    'Participant',
    'label',
    'category',
    'set'
]



# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

numerical_cols = [i for i in data_merged.columns if data_merged[i].dtype != 'object']

data_merged[numerical_cols].resample(rule='200ms').mean()

data_merged.columns
sampling = {
    'acc_x': "mean",
    'acc_y': "mean",
    'acc_z': "mean",
    'gyro_x': "mean",
    'gyro_y': "mean",
    'gyro_z': "mean",
    
    'Participant': 'last',
    'label': 'last',
    'category': 'last',
    'set': 'last'
}

days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]

data_resampled = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

data_resampled['set'] = data_resampled['set'].astype('int')

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle('../Data/interim-Data/01_data_processed.pkl')
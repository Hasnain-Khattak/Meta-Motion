import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../Data/interim-Data/01_data_processed.pkl')

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

## Plotting set column
set_df = df[df['set'] == 1] ## We are taking first set 
plt.plot(set_df['acc_y'].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

### Creating a for loop to check the individual label
df['label'].unique() ### This gives us the all the unique values in the label column
## Now writing for loop
for label in df['label'].unique():
    subset = df[df['label'] == label] ## This command will fetch all the labels in the dataframe and will use it to plot different labels
    ### Creating custom plot using matplotlib
    fig, ax = plt.subplots()
    ## We are taking only 100 samples 
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------

mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = (10, 5)
mpl.rcParams['figure.dpi'] = 100

for label in df['label'].unique():
    subset = df[df['label'] == label] ## This command will fetch all the labels in the dataframe and will use it to plot different labels
    ### Creating custom plot using matplotlib
    fig, ax = plt.subplots()
    ## We are taking only 100 samples 
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True), label = label)
    plt.legend()
    plt.show()

# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

# Compare medium vs. heavy sets
category_df = df.query("label == 'squat'").query("Participant == 'A'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(['category'])['acc_y'].plot() ## This wil groupby data on different category
ax.set_ylabel('Acc-y')
ax.set_xlabel('Samples')
ax.legend();

### Creating a for loop to check all the labels and their medium and heavy sets
for label in df['label'].unique():
    category_df_A = df.query(f"label == '{label}'").query("Participant == 'A'").reset_index()
    fig, ax = plt.subplots()
    category_df_A.groupby(['category'])['acc_y'].plot(label = label) ## This wil groupby data on different category
    ax.set_ylabel('Acc-y')
    ax.set_xlabel('Samples')
    ax.set_title('Difference Between Medium and Heavy Sets')
    ax.legend()
    plt.show()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values('Participant').reset_index()

fig, ax = plt.subplots()
participant_df.groupby(['Participant'])['acc_y'].plot() ## This wil groupby data on different category
ax.set_ylabel('Acc-y')
ax.set_xlabel('Samples')
ax.legend();

for label in df['label'].unique():
    participant_df_a = df.query(f"label == '{label}'").sort_values('Participant').reset_index()
    fig, ax = plt.subplots()
    participant_df_a.groupby(['Participant'])['acc_y'].plot() ## This wil groupby data on different Participant
    ax.set_ylabel('Acc-y')
    ax.set_xlabel(label)
    ax.legend()
    plt.show()
    

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------


label = 'squat'
participant = 'C'
all_axis_df = df.query(f"label == '{label}'").query(f"Participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
ax.set_ylabel('acc_y')
ax.set_xlabel('sample')
plt.legend()
plt.show()

# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

participant = 'C'
label = 'row'
combined_plot_df = df.query(f"label == '{label}'").query(f"Participant == '{participant}'").reset_index(drop=True)
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
combined_plot_df[['gyro_x', 'gyro_y', 'gyro_z']].plot(ax=ax[1])



# --------------------------------------------------------------
# Loop over all combinations and export for both sensors

labels = df['label'].unique()
Participants = df['Participant'].unique()

for label in labels:
    for participant in Participants:
        combined_plot_df = df.query(f"label == '{label}'").query(f"Participant == '{participant}'").reset_index(drop=True)
        
        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
            combined_plot_df[['gyro_x', 'gyro_y', 'gyro_z']].plot(ax=ax[1])
            plt.savefig(f"../Reports/figures/{label.title()} ({participant}).png")
            plt.show()
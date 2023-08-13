import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

directory = 'results/MonMiniGrid-NineRooms-v0/option_prob0.1'
# create empty dataframe to store averaged data
tot_df = pd.DataFrame()

n_seeds = 10
start = 11
# loop through each seed file
for i in range(start, start + n_seeds):
    filename = f'{directory}/seed{i}.csv'
    temp_df = pd.read_csv(filename)

    # extract relevant columns
    temp_df = temp_df[['frame', 'eval_episode_return']]

    # rename columns to include seed number
    temp_df.columns = ['frame', f'eval_return_seed{i}']

    # add data to df
    tot_df = pd.concat([tot_df, temp_df], axis=1)

# calculate mean and standard deviation over seeds
df_mean = tot_df.filter(like='eval_return').mean(axis=1)
df_std = tot_df.filter(like='eval_return').std(axis=1) / np.sqrt(n_seeds)

# smooth data with rolling window
df_smooth_mean = df_mean.rolling(window=20, min_periods=1).mean()
df_smooth_std = df_std.rolling(window=20, min_periods=1).mean()

# plot data
fig = plt.figure(figsize=(12, 8))
ax = plt.subplot()
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(28)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax.xaxis.get_offset_text().set_fontsize(30)
axis_font = {'fontname':'Arial', 'size':'38'}

ax.plot(df_smooth_mean.index * 1000, df_smooth_mean, label='DCO')
ax.fill_between(df_smooth_mean.index* 1000, 
    df_smooth_mean-df_smooth_std, df_smooth_mean+df_smooth_std, alpha=0.2)
ax.text(30_000, 0.5, 'Option discovery', rotation='vertical', fontsize=20)
ax.axvspan(0, 50_000, alpha=0.2, color='gray')
ax.axvline(x=50_000, color='k', linestyle='--')
ax.legend(prop={'size' : 24})

ax.set_xlabel('Time steps',**axis_font)
ax.set_ylabel('Episodic Return', **axis_font)
ax.set_xlim(0, 600_000)
ax.set_ylim(0., 1.0)
ax.tick_params(axis='x', which='major', labelsize=30)
ax.tick_params(axis='y', which='major', labelsize=30)
ax.set_title('Nine rooms', **axis_font)
plt.tight_layout()
# plt.show()
plt.savefig('nine_rooms_dco.png')

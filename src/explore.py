import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import july

plt.rcParams["figure.figsize"] = (25, 18)
df = pd.read_csv('../data/clean/train.csv', sep=';', parse_dates=['date'])

# Star exploration
# 1. Distribution of stars for all
# 2. Distribution of stars per assurance
# 3. Distribution of stars per assurance with same scaling
# 4. Number of review by day using a calendar
# 5. Simple evolution per assurace by time
# 6. Number of note per assurance
# 7. Mean note per assurance
# 8. Mean note per assurance with linear increase of color

# 1. Distribution of stars for all
sns.histplot(df, x='note', discrete=True)
plt.title('Stars distribution', fontsize=18)
plt.savefig('../fig/distrib.png', bbox_inches='tight')
plt.clf()

# 2. Distribution of stars per assurance without same scaling
fig, axs = plt.subplots(nrows=8, ncols=7)
assureurs = df['assureur'].unique()
for index, assureur in enumerate(assureurs):
    i, j = int(index / 7), (index - int(index / 7) * 7)
    sns.histplot(df[df['assureur'] == assureur],
                 x='note',
                 discrete=True,
                 hue='assureur',
                 legend=False,
                 binrange=(1, 5),
                 ax=axs[i, j])
    axs[i, j].set_xticks([1, 2, 3, 4, 5])
    axs[i, j].set_ylabel('')
    axs[i, j].set_xlabel('')
    axs[i, j].set_title(assureur)
fig.subplots_adjust(hspace=1)
fig.suptitle('Stars distribution per assureur without scaling', fontsize=18)
plt.savefig('../fig/distrib_split_noscale.png', bbox_inches='tight')
plt.clf()



# 3. Distribution of stars per assurance with same scaling
fig, axs = plt.subplots(nrows=8, ncols=7)
assureurs = df['assureur'].unique()
for index, assureur in enumerate(assureurs):
    i, j = int(index / 7), (index - int(index / 7) * 7)
    sns.histplot(df[df['assureur'] == assureur],
                 x='note',
                 discrete=True,
                 hue='assureur',
                 legend=False,
                 binrange=(1, 5),
                 ax=axs[i, j])
    axs[i, j].set_xticks([1, 2, 3, 4, 5])
    axs[i, j].set_ylabel('')
    axs[i, j].set_xlabel('')
    axs[i, j].set_title(assureur)
    axs[i, j].set_ylim([0, 2000])
fig.subplots_adjust(hspace=1)
fig.suptitle('Stars distribution per assureur with scaling', fontsize=18)
plt.savefig('../fig/distrib_split_scale.png', bbox_inches='tight')
plt.clf()


# 4. Number of review by day using a calendar
grouped = df.groupby('date').count()
grouped = grouped.reset_index()
fig, axs = plt.subplots(nrows=6, ncols=1)
for index, year in enumerate(range(2016, 2022)):
    temp_df = grouped[(grouped['date'].dt.year < year + 1) & (grouped['date'].dt.year >= year)]
    july.heatmap(temp_df['date'], temp_df['note'], ax=axs[index])
fig.subplots_adjust(hspace=1)
fig.suptitle('Number of review by day', fontsize=18)
plt.savefig('../fig/count_calendar.png', bbox_inches='tight')
plt.clf()


# 5. Simple evolution per assurace by time
fig, axs = plt.subplots(nrows=8, ncols=7)
assureurs = df['assureur'].unique()
df = df.sort_values(by='date')
for index, assureur in enumerate(assureurs):
    i, j = int(index / 7), (index - int(index / 7) * 7)
    filtered_df = df[df['assureur'] == assureur].groupby('date').mean().reset_index()
    sns.lineplot(x='date', y='note', data=filtered_df, ax=axs[i, j])
    axs[i, j].set_ylabel('')
    axs[i, j].set_xlabel('')
    axs[i, j].set_title(assureur)
    axs[i, j].set_ylim((1, 5))
    axs[i, j].set_xticks([])
fig.subplots_adjust(hspace=1)
fig.suptitle('Mean star evolution per assureur', fontsize=18)
plt.savefig('../fig/star_evolution.png', bbox_inches='tight')
plt.clf()

# 6. Number of note per assurance
grouped = df.groupby('assureur').size().reset_index()
sns.barplot(x='assureur', y=0, data=grouped, color='coral')
plt.tick_params(rotation=90)
plt.xlabel('')
plt.ylabel('Count')
plt.title('Number of note per assurance')
plt.savefig('../fig/nbnote_per_assureur.png', bbox_inches='tight')
plt.clf()


# 7. Mean note per assurance
grouped = df.groupby('assureur').agg({'note': 'mean', 'date': 'count'}).reset_index()
palette = sns.color_palette('crest', n_colors=grouped['date'].max())
palette = [palette[i - 1] for i in grouped['date'].values]
sns.barplot(x='assureur', y='note', data=grouped, palette=palette)
plt.tick_params(rotation=90)
plt.xlabel('')
plt.ylabel('Mean note')
plt.title('Mean note per assureur', fontsize=18)
plt.savefig('../fig/mean_note_per_assureur.png', bbox_inches='tight')
plt.clf()


# 8. Mean note per assurance with linear increase of color
grouped = df.groupby('assureur').agg({'note': 'mean', 'date': 'count'}).reset_index()
counts = np.sort(grouped['date'].unique())
grouped['date'] = grouped['date'].apply(lambda v: np.argwhere(counts == v)[0][0])
palette = sns.color_palette('crest', n_colors=grouped['date'].max())
palette = [palette[i - 1] for i in grouped['date'].values]
sns.barplot(x='assureur', y='note', data=grouped, palette=palette)
plt.tick_params(rotation=90)
plt.xlabel('')
plt.ylabel('Mean note')
plt.title('Mean note per assureur with linear coloring', fontsize=18)
plt.savefig('../fig/mean_note_per_assureur_linear.png', bbox_inches='tight')
plt.clf()

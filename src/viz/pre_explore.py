import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import july
from collections import namedtuple
from calendar import monthrange
from datetime import datetime


Date = namedtuple("Date", ["year", "month", "day"])

figwidth = 6.30045
plt.rcParams["figure.figsize"] = (figwidth, 5)
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
df = pd.read_csv('./data/clean/train.csv', sep=';', parse_dates=['date'])

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
plt.figure(figsize=(6, 5))
sns.histplot(df, x='note', discrete=True)
plt.savefig('./report/images/distrib.pgf', bbox_inches='tight')
plt.clf()


# 2. Distribution of stars per assurance without same scaling
fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(10, 6))
assureurs = df['assureur'].unique()
for index, assureur in enumerate(assureurs):
    i, j = int(index / 8), (index - int(index / 8) * 8)
    sns.histplot(df[df['assureur'] == assureur],
                 x='note',
                 discrete=True,
                 hue='assureur',
                 legend=False,
                 binrange=(1, 5),
                 ax=axs[i, j])
    axs[i, j].set_xticks([])
    axs[i, j].set_yticks([])
    axs[i, j].set_ylabel('')
    axs[i, j].set_xlabel('')
    title = assureur if len(assureur) < 10 else f"{assureur[:6]}..."
    axs[i, j].set_title(title, {'fontsize': 11})
fig.subplots_adjust(hspace=.5)
plt.savefig('./report/images/distrib_split_noscale.pgf', bbox_inches='tight')
plt.clf()


# 3. Distribution of stars per assurance with same scaling
fig, axs = plt.subplots(nrows=7, ncols=8, figsize=(10, 6))
assureurs = df['assureur'].unique()
for index, assureur in enumerate(assureurs):
    i, j = int(index / 8), (index - int(index / 8) * 8)
    sns.histplot(df[df['assureur'] == assureur],
                 x='note',
                 discrete=True,
                 hue='assureur',
                 legend=False,
                 binrange=(1, 5),
                 ax=axs[i, j])
    axs[i, j].set_xticks([])
    axs[i, j].set_yticks([])
    axs[i, j].set_ylabel('')
    axs[i, j].set_xlabel('')
    title = assureur if len(assureur) < 10 else f"{assureur[:6]}..."
    axs[i, j].set_title(title, {'fontsize': 11})
    axs[i, j].set_ylim([0, 2000])
fig.subplots_adjust(hspace=2)
plt.savefig('./report/images/distrib_split_scale.pgf', bbox_inches='tight')
plt.clf()


# 4. Number of review by day using a calendar
def all_dates_in_year(year):
    for month in range(1, 13):
        for day in range(1, monthrange(year, month)[1] + 1):
            yield pd.Timestamp(year=year, month=month, day=day)


grouped = df.groupby('date').count()
grouped = grouped.reset_index()
fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(6, 15))
for index, year in enumerate(range(2016, 2022)):
    temp_df = grouped[(grouped['date'].dt.year < year + 1) & (grouped['date'].dt.year >= year)]
    if year == 2016:
        temp_df = temp_df[['date', 'note']]
        for date in all_dates_in_year(2016):
            if date not in temp_df['date'].values:
                temp_df = temp_df.append({'date': date, 'note': 0}, ignore_index=True)
    july.heatmap(temp_df['date'], temp_df['note'], ax=axs[index])
    axs[index].tick_params(axis='both', which='major', labelsize=8)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig('./report/images/count_calendar.pgf', bbox_inches='tight')
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
plt.savefig('./report/images/star_evolution.pgf', bbox_inches='tight')
plt.clf()

# 6. Number of note per assurance
grouped = df.groupby('assureur').size().reset_index()
plt.figure(figsize=(9.5, 5))
sns.barplot(x='assureur', y=0, data=grouped, color='coral')
plt.tick_params(rotation=90)
plt.xticks(fontsize=5)
plt.xlabel('')
plt.ylabel('Count')
plt.savefig('./report/images/nbnote_per_assureur.pgf', bbox_inches='tight')
plt.clf()


# 7. Mean note per assurance
grouped = df.groupby('assureur').agg({'note': 'mean', 'date': 'count'}).reset_index()
plt.figure(figsize=(9.5, 3.5))
palette = sns.color_palette('crest', n_colors=grouped['date'].max())
palette = [palette[i - 1] for i in grouped['date'].values]
sns.barplot(x='assureur', y='note', data=grouped, palette=palette)
plt.tick_params(rotation=90)
plt.xticks(fontsize=6)
plt.xlabel('')
plt.ylabel('Mean note')
plt.savefig('./report/images/mean_note_per_assureur.pgf', bbox_inches='tight')
plt.clf()


# 8. Mean note per assurance with linear increase of color
grouped = df.groupby('assureur').agg({'note': 'mean', 'date': 'count'}).reset_index()
plt.figure(figsize=(9.5, 3.5))
counts = np.sort(grouped['date'].unique())
grouped['date'] = grouped['date'].apply(lambda v: np.argwhere(counts == v)[0][0])
palette = sns.color_palette('crest', n_colors=grouped['date'].max())
palette = [palette[i - 1] for i in grouped['date'].values]
sns.barplot(x='assureur', y='note', data=grouped, palette=palette)
plt.tick_params(rotation=90)
plt.xticks(fontsize=6)
plt.xlabel('')
plt.ylabel('Mean note')
plt.savefig('./report/images/mean_note_per_assureur_linear.pgf', bbox_inches='tight')
plt.clf()

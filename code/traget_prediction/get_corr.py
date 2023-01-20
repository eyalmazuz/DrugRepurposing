import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as ss
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

name = 'cancer_17_targets'
 
df = pd.read_csv(f'./{name}.csv')
df = df.T
corr_df = pd.DataFrame(columns=['Target 1', 'Target 2', 'Corr', 'P Value', 'Cramers Corr'])

for t1 in tqdm(df.index[1:]):
    for t2 in df.index[1:]:
        corr, pval = pearsonr(df.loc[t1], df.loc[t2])
        confusion_matrix = pd.crosstab(df.loc[t1], df.loc[t2])
        cramers_corr = cramers_v(confusion_matrix.values)
        
        series = pd.Series({'Target 1': t1, 'Target 2': t2, 'Corr': corr, 'P Value': pval, 'Cramers Corr': cramers_corr})
        corr_df = corr_df.append(series, ignore_index=True)
        
        
pt_corr = pd.pivot_table(corr_df, values='Corr', index='Target 1', columns='Target 2')

pt_val = pd.pivot_table(corr_df, values='P Value', index='Target 1', columns='Target 2')

pt_cramers = pd.pivot_table(corr_df, values='Cramers Corr', index='Target 1', columns='Target 2')

fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
sns.heatmap(pt_corr, vmin=-1, vmax=1, annot=True, fmt='.2f', ax=ax)
fig.figure.savefig(f'{name}_corr.png')

plt.clf()

fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
sns.heatmap(pt_val, vmin=0, vmax=1, annot=True, fmt='.2f', ax=ax)
fig.figure.savefig(f'{name}_pval.png')

plt.clf()

fig, ax = plt.subplots(figsize=(15,15))         # Sample figsize in inches
sns.heatmap(pt_cramers, vmin=0, vmax=1, annot=True, fmt='.2f', ax=ax)
fig.figure.savefig(f'{name}_cramers.png')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

print('loading preds')
preds = pd.read_csv('./test_preds_full.csv')
print('loading labels')
labels = pd.read_csv('./test_full.csv')

print('converting labels')
columns = labels.columns[1:]
for col in tqdm(preds.columns[1:]):
    preds[col] = preds[col].apply(eval)

df = pd.DataFrame(columns=[f'AUC_{i}' for i in range(3)] + [f'AUPR_{i}' for i in range(3)] + ['Prior'])
aucs = [[], [], []]
auprs = [[], [], []]
#ratios = []

print('predicting metrics')
for col in tqdm(columns):    
    label, pred = labels[col].tolist(), preds[col].tolist()
    for i in range(3):
        class_pred = [p[i] for p in pred]
        class_label = [1 if l == i else 0 for l in label]
        try:
            
            auc = roc_auc_score(class_label, class_pred)
            aucs[i].append(auc)
            auprs[i].append(0)
        except:
            aucs[i].append(-1)
            auprs[i].append(0)
    
        try:
            aupr = average_precision_score(class_label, class_pred)
            auprs[i].append(aupr)
        except:
            auprs[i].append(-1)
        #print(f'{auc=}, {aupr=}')
    
    #counts = labels[[col]].value_counts()
    #ratio = 1 - (counts[2.0] / labels.shape[0])
    ratio = 0
    #ratios.append(ratio)
    
    results = {f'AUC_{i}': aucs[i][-1] for i in range(3)}
    results = {**results, **{f'AUPR_{i}': auprs[i][-1] for i in range(3)}}
    results['Prior'] = ratio
    series = pd.Series(results, name=col)
    df = df.append(series)

print('saving df')
df.to_csv('./test_metrics.csv', index=True)

print(f'Average AUC: {sum(aucs) / len(aucs)} for {len(aucs)} targets')
print(f'Average AUCPR: {sum(auprs) / len(auprs)} for {len(auprs)} targets')
print('making plots')
aucplot = sns.kdeplot(aucs, fill=True)
aucplot.figure.savefig('./auc.png')

plt.clf()
auprplot = sns.kdeplot(auprs, fill=True)
auprplot.figure.savefig('./auprs.png')

       
plt.clf()
ratioplot = sns.kdeplot(ratios, fill=True)
ratioplot.figure.savefig('./pos.png')

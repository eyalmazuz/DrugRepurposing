import pandas as pd

df = pd.read_csv('./2021_12_14_12_14_18_preds.csv')

columns = df.columns[2:].tolist()

tri = [columns[i:i+3] for i in range(0, len(columns)-3, 3)]

z = [tri[i][0] for i in range(len(tri))]
o = [tri[i][1] for i in range(len(tri))]
t = [tri[i][2] for i in range(len(tri))]

names = [name[:-2] for name in z]

print(names[:5])
print(tri[:5])
print(z[:5])
print(o[:5])
print(t[:5])

zp = [df.loc[0, c] for c in z]
op = [df.loc[0,c] for c in o]
tp = [df.loc[0,c] for c in t]

print(zp[:5])
print(op[:5])
print(tp[:5])

preds = pd.DataFrame({'zero': zp, 'one': op, 'two': tp}, index=names)

print(preds.head())

print(preds['zero'].value_counts(bins=10))
print(preds['one'].value_counts(bins=10))
print(preds['two'].value_counts(bins=10))

print(preds[preds['zero'] > 0.0102].index.tolist())
print(preds[preds['one'] > 0.00361].index.tolist())


preds.to_csv('./erv.csv', index=True)

import pandas as pd
from tqdm import trange

test_smiles = pd.read_csv('./test_smiles.csv')
val_smiles = pd.read_csv('./val_smiles.csv')

for i in trange(10):
    print(f'Loading df {i}')
    df = pd.read_csv(f'./predictions/batches/df_{i}.csv')
    df = df[df.columns[1:]]

    print('Saving test')
    test = df[df['Smiles'].isin(test_smiles['Smiles'])]
    test.to_csv(f'./tests/df_{i}.csv', index=False)

    print('Saving Validation')
    val = df[df['Smiles'].isin(val_smiles['Smiles'])]
    val.to_csv(f'./val/df_{i}.csv', index=False)

    print('Saving btk')
    btk = df[['Smiles', 'BTK_class_0', 'BTK_class_1', 'BTK_class_2']]
    btk.to_csv(f'./btk/btk_{i}.csv', index=False)

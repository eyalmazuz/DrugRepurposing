import pandas as pd
from tqdm import trange

for i in trange(10):
    print('Loading df')
    df = pd.read_csv(f'./tests/test_{i}.csv')
    triplets = [df.columns.tolist()[i:i+3] for i in range(1, len(df.columns), 3)]
    di = {}
    di['Smiles'] = df['Smiles'].tolist()
    
    print('Adding classes')
    for t in triplets:
        name = t[0][:-8]
        di[name] = df[t].astype(float).idxmax(axis=1).apply(lambda x: int(x[-1])).tolist()

    print('Creating df')
    c_df = pd.DataFrame(di).set_index('Smiles')

    print('Saving df')
    c_df.to_csv(f'./tests/preds_{i}.csv', index=True)

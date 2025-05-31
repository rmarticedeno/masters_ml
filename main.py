import pandas as pd
import numpy as np

data = pd.read_csv('data/database.csv').dropna()
embeddings = np.load('data/bert_embeddings.npy')

labels = set()

for l in data['Labels']:
        for x in l.split(';'):
            lab = x.strip()
            if len(lab) > 0:
                labels.add(lab)

indexed_labels = list(labels)

vector_labels = []
for l in data['Labels']:
    row = []
    for x in l.split(';'):
        lab = x.strip()
        if len(lab) > 0:
             row.append(indexed_labels.index(lab))
    vector_labels.append(row)

# two columns for training
data['vector_labels'] = vector_labels
data['embeddings'] = list(embeddings)
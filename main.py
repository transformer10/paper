import tensorflow_hub as hub
import numpy as np
from datasets import load_dataset
import lightgbm as lgb

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print('use loaded')

raw_dataset = load_dataset('imdb')
raw_dataset = raw_dataset.shuffle()
print('imdb loaded')

train_text = raw_dataset['train']['text'][:10]
train_label = np.array(raw_dataset['train']['label'][:10], dtype='uint8')
test_text = raw_dataset['test']['text']
test_label = raw_dataset['test']['label']

train_embeddings = embed(train_text)
train_embeddings_np = np.array(train_embeddings, dtype='float64')
print('embeded finished')

model = lgb.LGBMClassifier()
model.fit(train_embeddings_np, train_label)

print('train finished')